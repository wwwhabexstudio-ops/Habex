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

import os
from datetime import datetime, timezone

import gradio as gr
import requests
from gradio_client import Client
from huggingface_hub import InferenceClient

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
            "Keep it under 100 words. One clear hook in the very first line."
        ),
        "Long (several minutes)": (
            "Write 300-500 words: a hook, 2-3 main points, and a closing line."
        ),
    }[video_form]

    system_prompt = (
        "You are a scriptwriter for online video. Write natural, spoken-style "
        "narration only — no stage directions, no headers, no markdown, no "
        "camera notes."
    )
    user_prompt = (
        f"Topic: {topic}\nTone: {tone}\n{length_hint}\n"
        "Write the script now, as plain narration text only."
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


def generate_video(prompt, negative_prompt, style, form, num_frames, progress=gr.Progress()):
    if not prompt or not prompt.strip():
        raise gr.Error("Please enter a prompt describing the video you want.")

    style_suffix = STYLE_PROMPT_SUFFIXES.get(style, "")
    full_prompt = f"{prompt}, {style_suffix}" if style_suffix else prompt

    video_id = log_video_started(
        prompt,
        num_frames,
        video_type=STYLE_DB_VALUES.get(style),
        form=FORM_DB_VALUES.get(form),
    )

    progress(0.1, desc="Connecting to video model...")
    try:
        client = get_client()
    except Exception as e:
        log_video_finished(video_id, "failed", error_message=e)
        raise gr.Error(f"Could not connect to the model backend: {e}")

    progress(0.3, desc="Generating your video (this can take 1-3 minutes on the free queue)...")
    try:
        # NOTE: exact parameter names depend on the backend Space's API.
        # Visit https://huggingface.co/spaces/Lightricks/LTX-Video-Playground
        # and click "Use via API" (bottom of page) to confirm these names,
        # then adjust the call below to match.
        result = client.predict(
            full_prompt,
            negative_prompt or "worst quality, blurry, distorted",
            num_frames,
            api_name="/generate",
        )
    except Exception as e:
        log_video_finished(video_id, "failed", error_message=e)
        raise gr.Error(
            "Generation failed. The free backend Space may be busy, "
            f"asleep, or its API changed. Details: {e}"
        )

    log_video_finished(video_id, "completed", video_url=result)
    progress(1.0, desc="Done!")
    return result


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
                    num_frames = gr.Slider(
                        label="Length (frames)", minimum=9, maximum=97, step=8, value=49
                    )
                    generate_btn = gr.Button("Generate Video", variant="primary")

                with gr.Column():
                    video_output = gr.Video(label="Your generated video")

    script_btn.click(
        fn=generate_script,
        inputs=[topic, video_form, tone],
        outputs=script_output,
    )
    send_to_video_btn.click(fn=lambda s: s, inputs=script_output, outputs=prompt)

    generate_btn.click(
        fn=generate_video,
        inputs=[prompt, negative_prompt, style, form_choice, num_frames],
        outputs=video_output,
    )

if __name__ == "__main__":
    # Render (and most cloud hosts) assign a port via $PORT and require
    # binding to 0.0.0.0. Locally this just falls back to Gradio's default.
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
