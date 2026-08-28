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

import math
import os
import re
import tempfile
from datetime import datetime, timezone

import gradio as gr
import requests
from gradio_client import Client
from huggingface_hub import InferenceClient
from moviepy import CompositeVideoClip, TextClip, VideoFileClip, concatenate_videoclips

# The public Space doing the actual video generation (free, shared GPU queue).
# Check its "Use via API" page (bottom of the Space) if the function
# signature below ever changes — Spaces occasionally update their API.
BACKEND_SPACE = "Lightricks/LTX-Video-Playground"

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
            # NOTE: exact parameter names depend on the backend Space's API.
            # Visit https://huggingface.co/spaces/Lightricks/LTX-Video-Playground
            # and click "Use via API" (bottom of page) to confirm these names,
            # then adjust the call below to match.
            clip = client.predict(
                full_prompt,
                negative_prompt or "worst quality, blurry, distorted",
                num_frames,
                api_name="/generate",
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
            progress(0.97, desc="Cropping to target aspect ratio...")
            result = apply_aspect_ratio(result, aspect_ratio)
        if add_captions and caption_text and caption_text.strip():
            progress(0.98, desc="Burning in captions...")
            result = add_burned_in_captions(result, caption_text)
    except Exception as e:
        # The core video already generated successfully — don't fail the
        # whole job over a post-processing step. Return the unprocessed
        # video and surface the issue.
        log_video_finished(video_id, "completed", video_url=result, error_message=f"post-processing skipped: {e}")
        gr.Warning(f"Video generated, but aspect ratio/captions step failed: {e}")
        progress(1.0, desc="Done (post-processing partially skipped).")
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
        try:
            requests.post(
                f"{SUPABASE_URL}/rest/v1/platform_uploads",
                headers=_supabase_headers(),
                json=payload,
                timeout=10,
            )
            when = f"for {scheduled_iso}" if scheduled_iso else "immediately"
            results.append(f"✅ Queued for **{platform}** ({when})")
        except Exception as e:
            results.append(f"❌ **{platform}** failed to queue: {e}")

    results.append(
        "\n\n⚠️ This logs the request in your dashboard (Tab 4 will show it "
        "as pending). It does **not** actually post yet — each platform "
        "needs its own API credentials connected first, added one at a "
        "time as you get developer access."
    )
    return "\n".join(results)


with gr.Blocks(title="My AI Video Generator") as demo:
    gr.Markdown(
        """
        # 🎬 My AI Video Generator
        Write a script from a topic, then turn it into a video —
        powered entirely by free, open-source models.
        """
    )

    with gr.Tabs():
        with gr.Tab("1. Write Script"):
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

        with gr.Tab("2. Generate Video"):
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

        with gr.Tab("3. Metadata"):
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

        with gr.Tab("4. Publish & Schedule"):
            gr.Markdown(
                "Queue a video to go out to one or more platforms — now or "
                "at a scheduled time. **Actual posting isn't connected yet** "
                "(needs each platform's API credentials); this logs the "
                "request so it's ready the moment publishing goes live."
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

        with gr.Tab("5. Dashboard"):
            gr.Markdown("Live totals pulled from your Supabase database.")
            dashboard_output = gr.Markdown("Click refresh to load stats.")
            dashboard_btn = gr.Button("Refresh Stats")

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

    script_btn.click(
        fn=generate_script,
        inputs=[topic, video_form, tone],
        outputs=script_output,
    )
    send_to_video_btn.click(fn=lambda s: s, inputs=script_output, outputs=prompt)
    # Also drop the script straight into the caption box and metadata box,
    # since that's the most common thing someone wants to reuse it for.
    send_to_video_btn.click(fn=lambda s: s, inputs=script_output, outputs=caption_text)
    send_to_video_btn.click(fn=lambda s: s, inputs=script_output, outputs=meta_content)

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
            target_seconds,
            num_frames,
        ],
        outputs=video_output,
    )

    meta_btn.click(
        fn=generate_metadata,
        inputs=[meta_content, meta_platforms],
        outputs=meta_output,
    )

    dashboard_btn.click(fn=get_dashboard_stats, outputs=dashboard_output)

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
