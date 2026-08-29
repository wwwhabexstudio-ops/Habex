"""
Your own AI Video Generation Tool
----------------------------------
This is the frontend + orchestration layer. It doesn't run the video model
itself (that needs a big GPU) — instead it calls a free, public Hugging Face
Space that already hosts an open-source text-to-video model (LTX-Video).

You can swap BACKEND_SPACE for any other text-to-video Space later
(e.g. a CogVideoX or HunyuanVideo Space) without changing your UI.

Deploy this file for free on Hugging Face Spaces (CPU basic tier, $0/mo):
1. Go to huggingface.co/new-space
2. Choose "Gradio" as the SDK
3. Upload app.py and requirements.txt from this folder
4. It builds automatically and gives you a public URL
"""

import asyncio
import json
import math
import os
import re
import tempfile
import uuid
from datetime import datetime, timezone

import edge_tts
import gradio as gr
import requests
from gradio_client import Client
from huggingface_hub import InferenceClient
from moviepy import (
    AudioFileClip,
    CompositeVideoClip,
    ImageClip,
    TextClip,
    VideoFileClip,
    concatenate_videoclips,
)

# The public Space doing the actual video generation (free, shared GPU queue).
# Check its "Use via API" page (bottom of the Space) if the function
# signature below ever changes — Spaces occasionally update their API.
BACKEND_SPACE = "Lightricks/ltx-video-distilled"
BACKEND_IMAGE_SPACE = "black-forest-labs/FLUX.1-schnell"

# Style choices map to prompt suffixes sent to the video model — this is
# how "video type" works with a single backend model (no separate model
# per style needed).
STYLE_PROMPT_SUFFIXES = {
    "Cinematic": "cinematic film still, dramatic lighting, shallow depth of field, film grain, 24fps look",
    "Realistic": "photorealistic, natural lighting, lifelike detail, 4k",
    "3D Animation": "3D render, CGI, smooth shading, animated film style",
    "Other / no style": "",
}

# Free-tier Hugging Face model used for script writing. Swappable —
# any instruct/chat model available on HF Inference works here.
SCRIPT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
HF_TOKEN = os.environ.get("HF_TOKEN")

YOUTUBE_CLIENT_ID = os.environ.get("YOUTUBE_CLIENT_ID")
YOUTUBE_CLIENT_SECRET = os.environ.get("YOUTUBE_CLIENT_SECRET")
YOUTUBE_REFRESH_TOKEN = os.environ.get("YOUTUBE_REFRESH_TOKEN")


def get_youtube_access_token():
    if not (YOUTUBE_CLIENT_ID and YOUTUBE_CLIENT_SECRET and YOUTUBE_REFRESH_TOKEN):
        raise RuntimeError("YouTube isn't connected — missing credentials on the server.")
    resp = requests.post(
        "https://oauth2.googleapis.com/token",
        data={
            "client_id": YOUTUBE_CLIENT_ID,
            "client_secret": YOUTUBE_CLIENT_SECRET,
            "refresh_token": YOUTUBE_REFRESH_TOKEN,
            "grant_type": "refresh_token",
        },
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def upload_to_youtube(video_path, title, description, tags):
    """Upload a video via the YouTube Data API. NOTE: until this project
    passes YouTube's own compliance audit, every upload lands as Private
    regardless of what privacyStatus is requested here — that's enforced
    by YouTube, not something this code controls. Flip to Public manually
    in YouTube Studio, or complete the audit for automatic public posting."""
    access_token = get_youtube_access_token()
    metadata = {
        "snippet": {
            "title": (title or "Untitled")[:100],
            "description": description or "",
            "tags": tags or [],
            "categoryId": "22",
        },
        "status": {"privacyStatus": "private"},
    }
    with open(video_path, "rb") as f:
        files = {
            "metadata": (None, json.dumps(metadata), "application/json; charset=UTF-8"),
            "file": (os.path.basename(video_path), f, "video/mp4"),
        }
        resp = requests.post(
            "https://www.googleapis.com/upload/youtube/v3/videos"
            "?uploadType=multipart&part=snippet,status",
            headers={"Authorization": f"Bearer {access_token}"},
            files=files,
            timeout=600,
        )
    resp.raise_for_status()
    return resp.json()["id"]

# Supabase logging (dashboard: total uploaded / success / failed).
# Set these as environment variables wherever you deploy this app.
# If unset, the app still works — it just won't log to the dashboard.
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")


def _supabase_headers():
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }


def log_video_started(script, duration_seconds, video_type=None, form=None):
    """Insert a 'generating' row and return its id, or None if logging is off."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        return None
    payload = {
        "script": script,
        "duration_seconds": duration_seconds,
        "status": "generating",
    }
    if video_type:
        payload["video_type"] = video_type
    if form:
        payload["form"] = form
    try:
        resp = requests.post(
            f"{SUPABASE_URL}/rest/v1/videos",
            headers=_supabase_headers(),
            json=payload,
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()[0]["id"]
    except Exception:
        # Logging failures should never break video generation itself.
        return None


def log_video_finished(video_id, status, video_url=None, error_message=None):
    if not video_id or not SUPABASE_URL or not SUPABASE_KEY:
        return
    payload = {"status": status, "completed_at": datetime.now(timezone.utc).isoformat()}
    if video_url:
        payload["video_url"] = str(video_url)
    if error_message:
        payload["error_message"] = str(error_message)[:2000]
    try:
        requests.patch(
            f"{SUPABASE_URL}/rest/v1/videos?id=eq.{video_id}",
            headers=_supabase_headers(),
            json=payload,
            timeout=10,
        )
    except Exception:
        pass


_client = None


def _count_rows(filter_query=""):
    """Get a row count from a Supabase table using PostgREST's exact-count header."""
    headers = _supabase_headers()
    headers["Prefer"] = "count=exact"
    resp = requests.get(
        f"{SUPABASE_URL}/rest/v1/{filter_query}",
        headers=headers,
        timeout=10,
    )
    resp.raise_for_status()
    content_range = resp.headers.get("Content-Range", "0/0")
    return int(content_range.split("/")[-1])


def get_backend_api_info():
    """Diagnostic: shows the REAL parameter list for each backend Space,
    straight from the Space itself — no guessing. Use this whenever a
    generation call fails with an api_name/parameter error."""
    sections = []
    for label, space in [("Video backend (LTX)", BACKEND_SPACE), ("Image backend (FLUX)", BACKEND_IMAGE_SPACE)]:
        try:
            client = Client(space)
            info = client.view_api(print_info=False, return_format="dict")
            endpoints = info.get("named_endpoints", {}) or {}
            lines = [f"### {label} — `{space}`"]
            if not endpoints:
                lines.append("_No named endpoints found._")
            for name, details in endpoints.items():
                params = details.get("parameters", [])
                if params:
                    param_lines = [
                        f"  - `{p.get('parameter_name', p.get('label', '?'))}` "
                        f"({p.get('python_type', {}).get('type', p.get('type', '?'))})"
                        + ("" if p.get("parameter_has_default", True) else " **required**")
                        for p in params
                    ]
                    lines.append(f"- **`{name}`** ({len(params)} params):\n" + "\n".join(param_lines))
                else:
                    lines.append(f"- **`{name}`**: (no parameters)")
            sections.append("\n".join(lines))
        except Exception as e:
            sections.append(f"### {label} — `{space}`\nCould not load API info: {e}")
    return "\n\n---\n\n".join(sections)


def get_dashboard_stats():
    if not SUPABASE_URL or not SUPABASE_KEY:
        return "Dashboard isn't connected — SUPABASE_URL/SUPABASE_KEY aren't set."
    try:
        total = _count_rows("videos?select=id")
        completed = _count_rows("videos?select=id&status=eq.completed")
        failed = _count_rows("videos?select=id&status=eq.failed")
        generating = _count_rows("videos?select=id&status=eq.generating")
        upload_success = _count_rows("platform_uploads?select=id&status=eq.success")
        upload_failed = _count_rows("platform_uploads?select=id&status=eq.failed")
        upload_pending = _count_rows("platform_uploads?select=id&status=eq.pending")
        return (
            f"## Videos\n"
            f"- **Total generated:** {total}\n"
            f"- ✅ Completed: {completed}\n"
            f"- ❌ Failed: {failed}\n"
            f"- ⏳ In progress: {generating}\n\n"
            f"## Platform uploads\n"
            f"- ✅ Successful: {upload_success}\n"
            f"- ❌ Failed: {upload_failed}\n"
            f"- ⏳ Pending/scheduled: {upload_pending}\n\n"
            f"*Per-channel view analytics will appear here once platform "
            f"publishing is connected — that data comes from each "
            f"platform's own analytics API.*"
        )
    except Exception as e:
        return f"Could not load dashboard stats: {e}"


def get_client():
    """Lazily connect to the backend Space so the app starts instantly."""
    global _client
    if _client is None:
        _client = Client(BACKEND_SPACE)
    return _client


def call_space(client, args, preferred_api_name, expected_num_params):
    """Call a Space endpoint, self-healing if the guessed api_name is
    wrong. Free community Spaces rename or restructure their endpoints
    without notice, so instead of failing outright, this inspects the
    Space's actual API and calls whichever endpoint takes the right
    number of parameters. If that also fails, the error message includes
    every real endpoint name/parameter count so the next fix is exact
    instead of another guess."""
    try:
        return client.predict(*args, api_name=preferred_api_name)
    except Exception as first_error:
        try:
            api_info = client.view_api(print_info=False, return_format="dict")
            endpoints = api_info.get("named_endpoints", {}) or api_info.get("unnamed_endpoints", {})
            for name, info in endpoints.items():
                if len(info.get("parameters", [])) == expected_num_params:
                    return client.predict(*args, api_name=name)
            available = {n: len(i.get("parameters", [])) for n, i in endpoints.items()}
            raise RuntimeError(
                f"'{preferred_api_name}' not found. This Space's real "
                f"endpoints are: {available}"
            )
        except Exception as second_error:
            raise RuntimeError(f"{first_error} | Auto-discovery also failed: {second_error}")


def generate_script(topic, video_form, tone):
    if not topic or not topic.strip():
        raise gr.Error("Please enter a topic first.")
    if not HF_TOKEN:
        raise gr.Error(
            "No Hugging Face token configured. Create one at "
            "huggingface.co/settings/tokens and add it as the HF_TOKEN "
            "environment variable on your host."
        )

    length_hint = {
        "Short (under 60 sec)": (
            "Keep it under 100 words total."
        ),
        "Long (several minutes)": (
            "Write 300-500 words, broken into a hook, 2-3 main points, and a closing line."
        ),
    }[video_form]

    system_prompt = (
        "You are a retention specialist who writes scripts for short-form "
        "video (TikTok/Reels/Shorts style pacing, even for longer videos). "
        "Your only job is to keep viewers watching and stop them from "
        "swiping away. Rules you always follow:\n"
        "1. The first line must be a scroll-stopping hook — a bold claim, "
        "a surprising fact, a question, or tension that creates a curiosity "
        "gap. Never open with a greeting, an introduction, or 'today we're "
        "talking about'.\n"
        "2. Never let the viewer feel like they already know where it's "
        "going — use open loops (teasing what's coming) and pattern "
        "interrupts (a turn, a twist, a 'but here's the thing') every "
        "few lines so there's no natural point to swipe away.\n"
        "3. Short, punchy sentences. No filler, no throat-clearing, no "
        "restating the topic before delivering value.\n"
        "4. Build toward a payoff or a satisfying ending line that "
        "rewards watching to the end — never trail off.\n"
        "Write natural, spoken-style narration only — no stage directions, "
        "no headers, no markdown, no camera notes."
    )
    user_prompt = (
        f"Topic: {topic}\nTone: {tone}\n{length_hint}\n"
        "Write the script now, as plain narration text only, following "
        "every rule above — especially the opening hook."
    )

    try:
        client = InferenceClient(token=HF_TOKEN)
        response = client.chat_completion(
            model=SCRIPT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=700,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise gr.Error(
            f"Script generation failed (the free model may be busy or "
            f"unavailable right now): {e}"
        )


STYLE_DB_VALUES = {
    "Cinematic": "cinematic",
    "Realistic": "realistic",
    "3D Animation": "3d",
    "Other / no style": "other",
}
FORM_DB_VALUES = {"Short": "short", "Long": "long"}

# The backend model generates short clips only. For "Long" form, we
# generate multiple clips back-to-back with the same prompt/style and
# stitch them into one file. This is an approximation of the requested
# frame count's real-world seconds — LTX-Video's exact fps isn't exposed
# through this API, so treat this as a rough per-clip duration estimate.
SECONDS_PER_CLIP = 4


ASPECT_RATIOS = {
    "Native (no cropping)": None,
    "9:16 (Vertical — TikTok/Reels/Shorts)": (9, 16),
    "16:9 (Horizontal — YouTube)": (16, 9),
    "1:1 (Square)": (1, 1),
}

# Optional path to a .ttf/.otf font file for nicer-looking captions. If unset,
# moviepy falls back to Pillow's built-in default font — plain but reliable,
# with no extra system dependencies needed.
CAPTION_FONT_PATH = os.environ.get("CAPTION_FONT_PATH")


def apply_aspect_ratio(video_path, aspect_ratio_key):
    """Center-crop the video to the target aspect ratio. Returns the (possibly
    unchanged) video path."""
    ratio = ASPECT_RATIOS.get(aspect_ratio_key)
    if not ratio:
        return video_path

    target_w, target_h = ratio
    target_ratio = target_w / target_h
    clip = VideoFileClip(video_path)
    try:
        current_ratio = clip.w / clip.h
        if abs(current_ratio - target_ratio) < 0.01:
            return video_path
        if current_ratio > target_ratio:
            new_w = int(clip.h * target_ratio)
            cropped = clip.cropped(x_center=clip.w / 2, width=new_w)
        else:
            new_h = int(clip.w / target_ratio)
            cropped = clip.cropped(y_center=clip.h / 2, height=new_h)
        out_path = os.path.join(tempfile.gettempdir(), "aspect_cropped.mp4")
        cropped.write_videofile(out_path, codec="libx264", audio_codec="aac", logger=None)
        cropped.close()
        return out_path
    finally:
        clip.close()


def _chunk_sentences(sentences, num_chunks):
    num_chunks = max(1, min(num_chunks, len(sentences)))
    k, m = divmod(len(sentences), num_chunks)
    return [
        sentences[i * k + min(i, m): (i + 1) * k + min(i + 1, m)]
        for i in range(num_chunks)
    ]


def add_burned_in_captions(video_path, caption_text):
    """Overlay caption_text across the video, split into timed chunks."""
    if not caption_text or not caption_text.strip():
        return video_path

    clip = VideoFileClip(video_path)
    try:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", caption_text.strip()) if s.strip()]
        if not sentences:
            return video_path

        # Roughly one caption chunk per 3 seconds of video.
        num_chunks = max(1, round(clip.duration / 3))
        chunks = _chunk_sentences(sentences, num_chunks)
        seg_duration = clip.duration / len(chunks)

        text_layers = []
        for i, chunk in enumerate(chunks):
            txt_clip = (
                TextClip(
                    font=CAPTION_FONT_PATH,
                    text=" ".join(chunk),
                    font_size=max(24, int(clip.w * 0.045)),
                    color="white",
                    bg_color="black",
                    method="caption",
                    size=(int(clip.w * 0.9), None),
                )
                .with_position(("center", "bottom"))
                .with_start(i * seg_duration)
                .with_duration(seg_duration)
            )
            text_layers.append(txt_clip)

        composite = CompositeVideoClip([clip, *text_layers])
        out_path = os.path.join(tempfile.gettempdir(), "captioned_output.mp4")
        composite.write_videofile(out_path, codec="libx264", audio_codec="aac", logger=None)
        return out_path
    finally:
        clip.close()


EDGE_VOICES = {"Female": "en-US-JennyNeural", "Male": "en-US-GuyNeural"}


def generate_voiceover(text, voice_choice):
    """Generate natural-sounding narration audio with Microsoft's free
    Edge TTS engine (no GPU, no API key, genuinely natural — not the
    robotic offline TTS engines like eSpeak)."""
    voice = EDGE_VOICES.get(voice_choice, EDGE_VOICES["Female"])
    out_path = os.path.join(tempfile.gettempdir(), f"voiceover_{uuid.uuid4().hex}.mp3")

    async def _run():
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(out_path)

    asyncio.run(_run())
    return out_path


def loop_video_to_duration(clip, target_duration):
    """Repeat a (silent) video clip until it covers target_duration, then
    trim to exactly that length — used to stretch short generated clips to
    match narration length."""
    if clip.duration >= target_duration:
        return clip.subclipped(0, target_duration)
    repeats = math.ceil(target_duration / clip.duration)
    looped = concatenate_videoclips([clip] * repeats)
    return looped.subclipped(0, target_duration)


SCENE_ASPECT_SIZES = {
    "9:16 (Vertical — TikTok/Reels/Shorts)": (720, 1280),
    "16:9 (Horizontal — YouTube)": (1280, 720),
    "1:1 (Square)": (960, 960),
}

_image_client = None


def get_image_client():
    global _image_client
    if _image_client is None:
        _image_client = Client(BACKEND_IMAGE_SPACE)
    return _image_client


def generate_scene_plan(script):
    """Ask the LLM to act as an AI Script Director: break a script into
    scenes, each with its own narration slice and a concrete visual
    description an image model can render."""
    if not HF_TOKEN:
        raise gr.Error(
            "No Hugging Face token configured. Add HF_TOKEN as an "
            "environment variable on your host."
        )
    system_prompt = (
        "You are an AI Script Director for short video. Break the given "
        "script into 4-8 scenes (hook, build, payoff). For each scene "
        "give: the exact narration text for that scene (a verbatim slice "
        "of the original script — do not paraphrase or rewrite it), and "
        "a vivid, concrete visual description an image generator can "
        "render (specific subject, setting, action — never an abstract "
        "concept like 'financial stress', always a literal visual scene). "
        "Respond with ONLY a JSON array, no commentary, no markdown "
        'fences, in exactly this shape: '
        '[{"narration": "...", "visual": "..."}, ...]'
    )
    user_prompt = f"Script:\n{script}\n\nBreak it into scenes now."
    try:
        client = InferenceClient(token=HF_TOKEN)
        response = client.chat_completion(
            model=SCRIPT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=1500,
        )
        raw = response.choices[0].message.content.strip()
    except Exception as e:
        raise gr.Error(f"Scene planning failed (free model may be busy): {e}")

    start, end = raw.find("["), raw.rfind("]")
    if start == -1 or end == -1:
        raise gr.Error("Scene Director couldn't find a scene list in the model's reply.")
    try:
        scenes = json.loads(raw[start:end + 1])
    except Exception as e:
        raise gr.Error(f"Scene Director got malformed JSON from the model: {e}")
    if not scenes:
        raise gr.Error("Scene Director produced zero scenes.")
    return scenes


def generate_scene_image(visual_prompt, style):
    style_suffix = STYLE_PROMPT_SUFFIXES.get(style, "")
    full_prompt = f"{visual_prompt}, {style_suffix}" if style_suffix else visual_prompt
    client = get_image_client()
    # This Space's real signature may take more than just a prompt (seed,
    # width, height, steps are common on FLUX demos) — call_space will
    # surface the real parameter count in the error if this guess is wrong,
    # rather than failing with no useful information.
    result = call_space(client, (full_prompt,), preferred_api_name="/infer", expected_num_params=1)
    return result[0] if isinstance(result, (list, tuple)) else result


def image_to_kenburns_clip(image_path, duration, size):
    """Turn a static image into a short clip with a slow zoom-in — real
    distinct AI visuals with motion, no second video-model call needed."""
    base = ImageClip(image_path).with_duration(duration)
    zoomed = base.resized(lambda t: 1 + 0.06 * (t / max(duration, 0.01)))
    return CompositeVideoClip([zoomed.with_position("center")], size=size).with_duration(duration)


def generate_scene_director_video(script, style, aspect_ratio, voice, add_captions, progress=gr.Progress()):
    if not script or not script.strip():
        raise gr.Error("Paste or write a script first — Tab 1 can generate one.")

    size = SCENE_ASPECT_SIZES.get(aspect_ratio, (720, 1280))
    video_id = log_video_started(script, 0, video_type=STYLE_DB_VALUES.get(style), form="long")

    progress(0.05, desc="Planning scenes...")
    try:
        scenes = generate_scene_plan(script)
    except Exception as e:
        log_video_finished(video_id, "failed", error_message=e)
        raise

    scene_clips = []
    try:
        for i, scene in enumerate(scenes):
            pct = 0.1 + 0.75 * (i / len(scenes))
            progress(pct, desc=f"Scene {i + 1}/{len(scenes)}: generating AI image...")
            image_path = generate_scene_image(
                scene.get("visual") or scene.get("narration", ""), style
            )

            progress(pct, desc=f"Scene {i + 1}/{len(scenes)}: generating narration...")
            audio_path = generate_voiceover(scene.get("narration", ""), voice)
            audio_clip = AudioFileClip(audio_path)

            clip = image_to_kenburns_clip(image_path, audio_clip.duration, size).with_audio(audio_clip)
            scene_clips.append(clip)
    except Exception as e:
        for c in scene_clips:
            c.close()
        log_video_finished(video_id, "failed", error_message=e)
        raise gr.Error(f"Scene Director failed partway through (scene {len(scene_clips) + 1}): {e}")

    progress(0.9, desc="Assembling final video...")
    try:
        final = concatenate_videoclips(scene_clips, method="compose")
        out_path = os.path.join(tempfile.gettempdir(), "scene_director_output.mp4")
        final.write_videofile(out_path, codec="libx264", audio_codec="aac", fps=24, logger=None)
    except Exception as e:
        log_video_finished(video_id, "failed", error_message=e)
        raise gr.Error(f"Assembling scenes together failed: {e}")
    finally:
        for c in scene_clips:
            c.close()

    result = out_path
    if add_captions:
        try:
            progress(0.97, desc="Burning in captions...")
            result = add_burned_in_captions(result, script)
        except Exception as e:
            log_video_finished(video_id, "completed", video_url=result, error_message=f"captions skipped: {e}")
            gr.Warning(f"Video assembled, but captions failed: {e}")
            progress(1.0, desc="Done (captions skipped).")
            return result

    log_video_finished(video_id, "completed", video_url=result)
    progress(1.0, desc="Done!")
    return result


def stitch_clips(clip_paths):
    """Concatenate several video files into one and return the output path."""
    clips = [VideoFileClip(p) for p in clip_paths]
    try:
        final = concatenate_videoclips(clips, method="compose")
        out_path = os.path.join(tempfile.gettempdir(), "stitched_output.mp4")
        final.write_videofile(out_path, codec="libx264", audio_codec="aac", logger=None)
        return out_path
    finally:
        for c in clips:
            c.close()


PLATFORM_META_RULES = {
    "YouTube": (
        "Title under 100 characters, front-load the keyword, avoid clickbait "
        "that isn't paid off. Description: 2-3 sentences summarizing the "
        "video plus 3-5 relevant hashtags at the end. Tags: 8-12 comma-"
        "separated search terms."
    ),
    "TikTok": (
        "Caption under 150 characters, punchy and hook-driven, matches the "
        "video's opening line. 3-5 hashtags woven naturally into the "
        "caption, mixing broad and niche tags."
    ),
    "Instagram": (
        "Caption's first line must hook on its own since Instagram truncates "
        "after ~125 characters before 'more'. Up to 3-4 short sentences "
        "total. 5-8 relevant hashtags grouped at the end."
    ),
    "Facebook": (
        "Conversational caption, 1-3 sentences, phrased to invite comments "
        "or shares. 0-2 hashtags at most — Facebook audiences respond "
        "better to plain language than hashtag-heavy captions."
    ),
}


def parse_metadata_for_publish(metadata_markdown):
    """Pull the first platform's title/description/tags out of the
    generated metadata block to auto-fill the Publish tab. If multiple
    platforms were generated, only the first is used — the others stay
    visible in the Metadata tab for manual copy if needed."""
    if not metadata_markdown:
        return "", "", ""
    title_match = re.search(r"TITLE:\s*(.+)", metadata_markdown)
    desc_match = re.search(r"DESCRIPTION:\s*(.+)", metadata_markdown)
    tags_match = re.search(r"TAGS:\s*(.+)", metadata_markdown)
    title = title_match.group(1).strip() if title_match else ""
    description = desc_match.group(1).strip() if desc_match else ""
    tags = tags_match.group(1).strip() if tags_match else ""
    return title, description, tags


def generate_metadata(content, platforms):
    if not content or not content.strip():
        raise gr.Error("Paste your script or topic first.")
    if not platforms:
        raise gr.Error("Select at least one platform.")
    if not HF_TOKEN:
        raise gr.Error(
            "No Hugging Face token configured. Create one at "
            "huggingface.co/settings/tokens and add it as the HF_TOKEN "
            "environment variable on your host."
        )

    client = InferenceClient(token=HF_TOKEN)
    sections = []
    for platform in platforms:
        rules = PLATFORM_META_RULES[platform]
        system_prompt = (
            "You write high-converting video metadata for social platforms. "
            "Follow the given platform's conventions exactly. Never use "
            "generic filler like 'check out this video'."
        )
        user_prompt = (
            f"Platform: {platform}\nConventions to follow: {rules}\n\n"
            f"Video content (script or topic):\n{content}\n\n"
            "Respond in exactly this format, nothing else:\n"
            "TITLE: <title>\nDESCRIPTION: <description>\nTAGS: <comma-separated tags/hashtags>"
        )
        try:
            response = client.chat_completion(
                model=SCRIPT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=400,
            )
            sections.append(f"### {platform}\n{response.choices[0].message.content.strip()}")
        except Exception as e:
            sections.append(f"### {platform}\n(generation failed: {e})")

    return "\n\n".join(sections)


def generate_video(
    prompt,
    negative_prompt,
    style,
    form,
    aspect_ratio,
    add_captions,
    caption_text,
    add_narration,
    narration_text,
    voice,
    target_seconds,
    num_frames,
    progress=gr.Progress(),
):
    if not prompt or not prompt.strip():
        raise gr.Error("Please enter a prompt describing the video you want.")

    style_suffix = STYLE_PROMPT_SUFFIXES.get(style, "")
    full_prompt = f"{prompt}, {style_suffix}" if style_suffix else prompt

    video_id = log_video_started(
        prompt,
        target_seconds if form == "Long" else round(num_frames / 24),
        video_type=STYLE_DB_VALUES.get(style),
        form=FORM_DB_VALUES.get(form),
    )

    progress(0.1, desc="Connecting to video model...")
    try:
        client = get_client()
    except Exception as e:
        log_video_finished(video_id, "failed", error_message=e)
        raise gr.Error(f"Could not connect to the model backend: {e}")

    num_clips = 1
    if form == "Long":
        num_clips = max(1, math.ceil(target_seconds / SECONDS_PER_CLIP))

    clip_paths = []
    try:
        for i in range(num_clips):
            pct = 0.1 + 0.8 * (i / num_clips)
            progress(
                pct,
                desc=f"Generating clip {i + 1}/{num_clips} "
                f"(free queue — 1-3 min per clip)...",
            )
            # If the guessed api_name is wrong, call_space auto-discovers
            # the Space's real endpoint instead of failing outright.
            clip = call_space(
                client,
                (full_prompt, negative_prompt or "worst quality, blurry, distorted", num_frames),
                preferred_api_name="/generate",
                expected_num_params=3,
            )
            clip_paths.append(clip)
    except Exception as e:
        log_video_finished(video_id, "failed", error_message=e)
        raise gr.Error(
            "Generation failed partway through. The free backend Space may "
            f"be busy, asleep, or its API changed. Details: {e}"
        )

    if num_clips == 1:
        result = clip_paths[0]
    else:
        progress(0.95, desc=f"Stitching {num_clips} clips into one video...")
        try:
            result = stitch_clips(clip_paths)
        except Exception as e:
            log_video_finished(video_id, "failed", error_message=e)
            raise gr.Error(
                f"All {num_clips} clips generated, but stitching them "
                f"together failed: {e}"
            )

    try:
        if aspect_ratio and aspect_ratio != "Native (no cropping)":
            progress(0.95, desc="Cropping to target aspect ratio...")
            result = apply_aspect_ratio(result, aspect_ratio)
        if add_captions and caption_text and caption_text.strip():
            progress(0.97, desc="Burning in captions...")
            result = add_burned_in_captions(result, caption_text)
        if add_narration and narration_text and narration_text.strip():
            progress(0.99, desc="Generating natural voiceover narration...")
            audio_path = generate_voiceover(narration_text, voice)
            video_clip = VideoFileClip(result)
            audio_clip = AudioFileClip(audio_path)
            matched_video = loop_video_to_duration(video_clip, audio_clip.duration)
            final = matched_video.with_audio(audio_clip)
            narrated_path = os.path.join(tempfile.gettempdir(), "final_with_voiceover.mp4")
            final.write_videofile(narrated_path, codec="libx264", audio_codec="aac", logger=None)
            result = narrated_path
    except Exception as e:
        # The core video already generated successfully — don't fail the
        # whole job over a post-processing step. Return what we have and
        # surface the issue.
        log_video_finished(video_id, "completed", video_url=result, error_message=f"post-processing skipped: {e}")
        gr.Warning(f"Video generated, but a finishing step failed: {e}")
        progress(1.0, desc="Done (some post-processing was skipped).")
        return result

    log_video_finished(video_id, "completed", video_url=result)
    progress(1.0, desc="Done!")
    return result


def submit_publish_request(uploaded_file, platforms, title, description, tags, caption, schedule_dt):
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise gr.Error("Scheduling database isn't connected.")
    if not platforms:
        raise gr.Error("Select at least one platform.")
    if not uploaded_file:
        raise gr.Error(
            "Upload a video file — either one you made elsewhere, or the "
            "file you downloaded after generating one in Tab 2."
        )

    tags_list = [t.strip() for t in (tags or "").split(",") if t.strip()]
    scheduled_iso = schedule_dt.strip() if schedule_dt and schedule_dt.strip() else None

    try:
        video_resp = requests.post(
            f"{SUPABASE_URL}/rest/v1/videos",
            headers=_supabase_headers(),
            json={
                "script": title or "manually uploaded video",
                "duration_seconds": 0,
                "status": "completed",
                "video_url": str(uploaded_file),
            },
            timeout=10,
        )
        video_resp.raise_for_status()
        video_id = video_resp.json()[0]["id"]
    except Exception as e:
        raise gr.Error(f"Could not save video record: {e}")

    results = []
    for platform in platforms:
        payload = {
            "video_id": video_id,
            "platform": platform.lower(),
            "status": "pending",
            "title": title,
            "description": description,
            "tags": tags_list,
            "caption": caption,
            "source_video_path": str(uploaded_file),
        }
        if scheduled_iso:
            payload["scheduled_at"] = scheduled_iso

        if platform == "YouTube" and not scheduled_iso:
            # Real upload — the only platform actually wired up so far.
            try:
                yt_video_id = upload_to_youtube(str(uploaded_file), title, description, tags_list)
                payload["status"] = "success"
                payload["platform_post_id"] = yt_video_id
                payload["published_at"] = datetime.now(timezone.utc).isoformat()
                results.append(
                    f"✅ **YouTube**: uploaded as **Private** (video ID `{yt_video_id}`) "
                    f"— YouTube enforces Private on all API uploads until this "
                    f"project passes their compliance audit. Flip it to Public "
                    f"in YouTube Studio to publish it."
                )
            except Exception as e:
                payload["status"] = "failed"
                payload["error_message"] = str(e)[:2000]
                results.append(f"❌ **YouTube** upload failed: {e}")
        elif platform == "YouTube" and scheduled_iso:
            results.append(
                f"⏳ **YouTube** logged for {scheduled_iso} — automatic "
                f"posting at a scheduled time isn't built yet (needs a "
                f"background job runner); this just records the request."
            )
        else:
            when = f"for {scheduled_iso}" if scheduled_iso else "immediately"
            results.append(
                f"📝 **{platform}** logged ({when}) — real posting isn't "
                f"connected for this platform yet."
            )

        try:
            requests.post(
                f"{SUPABASE_URL}/rest/v1/platform_uploads",
                headers=_supabase_headers(),
                json=payload,
                timeout=10,
            )
        except Exception as e:
            results.append(f"⚠️ Could not log {platform} to the dashboard: {e}")

    return "\n\n".join(results)


CUSTOM_CSS = """
:root {
    --body-background-fill: #0D0F12;
    --background-fill-primary: #15181D;
    --background-fill-secondary: #1B1F26;
    --border-color-primary: #262B33;
    --border-color-accent: #E63946;
    --body-text-color: #EDEAE3;
    --body-text-color-subdued: #8B9099;
    --block-background-fill: #15181D;
    --block-border-color: #262B33;
    --block-label-background-fill: #1B1F26;
    --block-label-text-color: #9AA0AB;
    --block-title-text-color: #EDEAE3;
    --input-background-fill: #0F1216;
    --input-border-color: #262B33;
    --button-primary-background-fill: #E63946;
    --button-primary-background-fill-hover: #F0505C;
    --button-primary-text-color: #0D0F12;
    --button-secondary-background-fill: #1B1F26;
    --button-secondary-border-color: #333944;
    --button-secondary-text-color: #EDEAE3;
    --panel-background-fill: #15181D;
}

.gradio-container {
    font-family: 'Inter', ui-sans-serif, sans-serif !important;
    background: #0D0F12 !important;
    max-width: 1180px !important;
}

/* Header */
#hs-header {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 18px 4px 22px 4px;
    border-bottom: 1px solid #262B33;
    margin-bottom: 18px;
}
#hs-header .hs-dot {
    width: 11px;
    height: 11px;
    border-radius: 50%;
    background: #E63946;
    box-shadow: 0 0 10px 2px rgba(230, 57, 70, 0.65);
    animation: hs-pulse 1.8s ease-in-out infinite;
    flex-shrink: 0;
}
@keyframes hs-pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.45; }
}
#hs-header .hs-title {
    font-family: 'Space Grotesk', ui-sans-serif, sans-serif;
    font-size: 22px;
    font-weight: 600;
    letter-spacing: -0.01em;
    color: #EDEAE3;
    margin: 0;
}
#hs-header .hs-tagline {
    font-family: 'Inter', ui-sans-serif, sans-serif;
    font-size: 13px;
    color: #8B9099;
    margin: 2px 0 0 0;
}
#hs-header .hs-rec-label {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 11px;
    letter-spacing: 0.08em;
    color: #E63946;
    margin-left: auto;
    border: 1px solid #3A2224;
    background: #1C1013;
    padding: 4px 10px;
    border-radius: 5px;
}

/* Tabs — pill style */
.tabs > .tab-nav {
    border-bottom: 1px solid #262B33 !important;
    gap: 4px;
}
.tabs > .tab-nav button {
    font-family: 'Inter', ui-sans-serif, sans-serif !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    color: #8B9099 !important;
    border-radius: 8px 8px 0 0 !important;
    padding: 9px 16px !important;
}
.tabs > .tab-nav button.selected {
    color: #EDEAE3 !important;
    background: #15181D !important;
    border-bottom: 2px solid #E63946 !important;
}

/* Numbers/data get the mono treatment */
input[type="number"], .gr-slider input {
    font-family: 'JetBrains Mono', ui-monospace, monospace !important;
}

/* Panels */
.block {
    border-radius: 10px !important;
}
button.primary {
    border-radius: 8px !important;
    font-weight: 600 !important;
}
"""

HEADER_HTML = """
<div id="hs-header">
    <span class="hs-dot"></span>
    <div>
        <p class="hs-title">Habex Studio</p>
        <p class="hs-tagline">Script to published video — one pipeline.</p>
    </div>
    <span class="hs-rec-label">● LIVE</span>
</div>
"""

theme = gr.themes.Base(
    primary_hue=gr.themes.colors.red,
    secondary_hue=gr.themes.colors.slate,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
)

with gr.Blocks(title="Habex Studio", theme=theme, css=CUSTOM_CSS) as demo:
    gr.HTML(HEADER_HTML)

    with gr.Tabs() as main_tabs:
        with gr.Tab("1. Write Script", id="tab_script"):
            with gr.Row():
                with gr.Column():
                    topic = gr.Textbox(
                        label="Topic",
                        placeholder="Why cats sleep so much",
                        lines=1,
                    )
                    video_form = gr.Radio(
                        label="Video length",
                        choices=["Short (under 60 sec)", "Long (several minutes)"],
                        value="Short (under 60 sec)",
                    )
                    tone = gr.Dropdown(
                        label="Tone",
                        choices=["casual", "energetic", "calm/documentary", "funny", "dramatic"],
                        value="casual",
                    )
                    script_btn = gr.Button("Generate Script", variant="primary")
                with gr.Column():
                    script_output = gr.Textbox(
                        label="Generated script (editable — tweak anything before generating video)",
                        lines=12,
                    )
                    send_to_video_btn = gr.Button("Use this script →  Go to Video tab")

        with gr.Tab("2. Generate Video", id="tab_video"):
            with gr.Row():
                with gr.Column():
                    prompt = gr.Textbox(
                        label="Video prompt (from your script, or write your own)",
                        placeholder="A golden retriever running on a beach at sunset, cinematic",
                        lines=3,
                    )
                    negative_prompt = gr.Textbox(
                        label="What to avoid (optional)",
                        placeholder="blurry, low quality, distorted",
                        lines=1,
                    )
                    with gr.Row():
                        style = gr.Dropdown(
                            label="Video style",
                            choices=list(STYLE_PROMPT_SUFFIXES.keys()),
                            value="Cinematic",
                        )
                        form_choice = gr.Radio(
                            label="Form", choices=["Short", "Long"], value="Short"
                        )
                    aspect_ratio = gr.Dropdown(
                        label="Aspect ratio",
                        choices=list(ASPECT_RATIOS.keys()),
                        value="9:16 (Vertical — TikTok/Reels/Shorts)",
                    )
                    add_captions = gr.Checkbox(label="Add burned-in captions", value=False)
                    caption_text = gr.Textbox(
                        label="Caption text (usually your script — paste or edit)",
                        lines=4,
                        visible=False,
                    )
                    with gr.Row():
                        add_narration = gr.Checkbox(label="Add AI voiceover narration", value=False)
                        voice = gr.Dropdown(
                            label="Voice", choices=["Female", "Male"], value="Female"
                        )
                    narration_text = gr.Textbox(
                        label="Narration text (usually your script — paste or edit)",
                        lines=4,
                        visible=False,
                    )
                    target_seconds = gr.Slider(
                        label="Target total duration (seconds) — Long form only",
                        minimum=20,
                        maximum=900,
                        step=10,
                        value=60,
                        visible=False,
                        info=(
                            "Generated as multiple short clips stitched together "
                            "(free model limitation). Longer targets mean more "
                            "clips, more free-queue wait time, and more chances "
                            "one clip fails and needs a retry."
                        ),
                    )
                    num_frames = gr.Slider(
                        label="Per-clip length (frames)", minimum=9, maximum=97, step=8, value=49
                    )
                    generate_btn = gr.Button("Generate Video", variant="primary")

                with gr.Column():
                    video_output = gr.Video(label="Your generated video")
                    video_send_btn = gr.Button("Send to Publish →")

        with gr.Tab("3. Scene Director (Beta)", id="tab_scene"):
            gr.Markdown(
                "The real AI Director pipeline: breaks your script into "
                "scenes, generates a **distinct AI image for every scene** "
                "(no repeated visuals), narrates each one, and assembles "
                "them with cinematic pan/zoom motion. Slower than Tab 2 — "
                "one image + one narration call per scene — but much "
                "closer to a true production pipeline."
            )
            with gr.Row():
                with gr.Column():
                    sd_script = gr.Textbox(
                        label="Script (from Tab 1, or paste your own)", lines=10
                    )
                    sd_style = gr.Dropdown(
                        label="Visual style",
                        choices=list(STYLE_PROMPT_SUFFIXES.keys()),
                        value="Cinematic",
                    )
                    sd_aspect = gr.Dropdown(
                        label="Aspect ratio",
                        choices=list(SCENE_ASPECT_SIZES.keys()),
                        value="9:16 (Vertical — TikTok/Reels/Shorts)",
                    )
                    sd_voice = gr.Dropdown(
                        label="Voice", choices=["Female", "Male"], value="Female"
                    )
                    sd_captions = gr.Checkbox(label="Add burned-in captions", value=False)
                    sd_btn = gr.Button("Generate Scene-by-Scene Video", variant="primary")
                with gr.Column():
                    sd_output = gr.Video(label="Your generated video")
                    sd_send_btn = gr.Button("Send to Publish →")

        with gr.Tab("4. Metadata", id="tab_metadata"):
            gr.Markdown("Generate a title, description, and tags/hashtags matched to each platform's conventions.")
            with gr.Row():
                with gr.Column():
                    meta_content = gr.Textbox(
                        label="Your script or topic",
                        lines=8,
                    )
                    meta_platforms = gr.CheckboxGroup(
                        label="Platforms",
                        choices=["YouTube", "TikTok", "Instagram", "Facebook"],
                        value=["YouTube"],
                    )
                    meta_btn = gr.Button("Generate Metadata", variant="primary")
                with gr.Column():
                    meta_output = gr.Markdown(label="Generated metadata")
                    meta_send_btn = gr.Button("Send to Publish →")

        with gr.Tab("5. Publish & Schedule", id="tab_publish"):
            gr.Markdown(
                "**YouTube is live** — uploads for real (lands as Private "
                "until this project passes YouTube's compliance audit; "
                "flip to Public in YouTube Studio). TikTok/Instagram/"
                "Facebook aren't connected yet — selecting them just logs "
                "the request for when they are. Scheduling for a future "
                "time also just logs it for now (no background job runner "
                "built yet)."
            )
            with gr.Row():
                with gr.Column():
                    publish_file = gr.File(
                        label="Video file (upload your own, or the one you generated in Tab 2)",
                        file_types=["video"],
                    )
                    publish_platforms = gr.CheckboxGroup(
                        label="Platforms",
                        choices=["YouTube", "TikTok", "Instagram", "Facebook"],
                    )
                    publish_title = gr.Textbox(label="Title")
                    publish_description = gr.Textbox(label="Description", lines=3)
                    publish_tags = gr.Textbox(label="Tags (comma-separated)")
                    publish_caption = gr.Textbox(label="Caption", lines=2)
                    publish_schedule = gr.Textbox(
                        label="Schedule for (optional — leave blank to queue for 'now')",
                        placeholder="2026-09-01T14:00:00",
                        info="ISO format, in your server's timezone (UTC on Render).",
                    )
                    publish_btn = gr.Button("Queue Publish Request", variant="primary")
                with gr.Column():
                    publish_output = gr.Markdown()

        with gr.Tab("6. Dashboard", id="tab_dashboard"):
            gr.Markdown("Live totals pulled from your Supabase database.")
            dashboard_output = gr.Markdown("Click refresh to load stats.")
            dashboard_btn = gr.Button("Refresh Stats")

            gr.Markdown(
                "---\n### Backend API Diagnostic\n"
                "If video or image generation errors mention `api_name` or "
                "a parameter mismatch, click below — it shows the real, "
                "current parameter list straight from each backend Space "
                "(no guessing)."
            )
            api_debug_output = gr.Markdown()
            api_debug_btn = gr.Button("Check Backend API Info")

    form_choice.change(
        fn=lambda f: gr.update(visible=(f == "Long")),
        inputs=form_choice,
        outputs=target_seconds,
    )
    add_captions.change(
        fn=lambda checked: gr.update(visible=checked),
        inputs=add_captions,
        outputs=caption_text,
    )
    add_narration.change(
        fn=lambda checked: gr.update(visible=checked),
        inputs=add_narration,
        outputs=narration_text,
    )

    script_btn.click(
        fn=generate_script,
        inputs=[topic, video_form, tone],
        outputs=script_output,
    )
    def send_script_to_video_tab(script):
        return script, script, script, script, gr.Tabs(selected="tab_video")

    send_to_video_btn.click(
        fn=send_script_to_video_tab,
        inputs=script_output,
        outputs=[prompt, caption_text, narration_text, meta_content, main_tabs],
    )
    # Scene Director gets the script too, but doesn't switch tabs to it —
    # the button says "Go to Video tab", so that's where it should land.
    send_to_video_btn.click(fn=lambda s: s, inputs=script_output, outputs=sd_script)

    generate_btn.click(
        fn=generate_video,
        inputs=[
            prompt,
            negative_prompt,
            style,
            form_choice,
            aspect_ratio,
            add_captions,
            caption_text,
            add_narration,
            narration_text,
            voice,
            target_seconds,
            num_frames,
        ],
        outputs=video_output,
    )
    # Send the finished video straight to Publish, ready to queue/upload.
    def send_video_to_publish_tab(video_path):
        return video_path, gr.Tabs(selected="tab_publish")

    video_send_btn.click(
        fn=send_video_to_publish_tab,
        inputs=video_output,
        outputs=[publish_file, main_tabs],
    )

    meta_btn.click(
        fn=generate_metadata,
        inputs=[meta_content, meta_platforms],
        outputs=meta_output,
    )
    # Auto-fill the Publish tab's title/description/tags from the first
    # generated platform's metadata, then jump straight there.
    def send_metadata_to_publish_tab(metadata_markdown):
        title, description, tags = parse_metadata_for_publish(metadata_markdown)
        return title, description, tags, gr.Tabs(selected="tab_publish")

    meta_send_btn.click(
        fn=send_metadata_to_publish_tab,
        inputs=meta_output,
        outputs=[publish_title, publish_description, publish_tags, main_tabs],
    )

    sd_btn.click(
        fn=generate_scene_director_video,
        inputs=[sd_script, sd_style, sd_aspect, sd_voice, sd_captions],
        outputs=sd_output,
    )
    sd_send_btn.click(
        fn=send_video_to_publish_tab,
        inputs=sd_output,
        outputs=[publish_file, main_tabs],
    )

    dashboard_btn.click(fn=get_dashboard_stats, outputs=dashboard_output)
    api_debug_btn.click(fn=get_backend_api_info, outputs=api_debug_output)


    publish_btn.click(
        fn=submit_publish_request,
        inputs=[
            publish_file,
            publish_platforms,
            publish_title,
            publish_description,
            publish_tags,
            publish_caption,
            publish_schedule,
        ],
        outputs=publish_output,
    )

if __name__ == "__main__":
    # Render (and most cloud hosts) assign a port via $PORT and require
    # binding to 0.0.0.0. Locally this just falls back to Gradio's default.
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
