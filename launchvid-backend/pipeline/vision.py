import google.generativeai as genai
import json, base64

genai.configure(api_key="YOUR_GEMINI_KEY")  # free at aistudio.google.com

async def analyze_frame(frame: dict) -> dict:
    model = genai.GenerativeModel("gemini-2.5-flash")

    image_bytes = base64.b64decode(frame["fullPngBase64"])

    prompt = f"""
    You are building an animated app launch video. Analyze this UI screen.
    Layer tree: {json.dumps(frame["layers"], indent=2)[:3000]}

    Return ONLY valid JSON (no markdown):
    {{
      "screen_purpose": "onboarding|dashboard|feature|checkout|landing",
      "headline": "the most prominent text on screen",
      "narration": "one sentence for voiceover, max 15 words",
      "animation_sequence": [
        {{ "layer_id": "...", "layer_name": "...", "animation": "fade_in|slide_up|slide_right|scale_in|pulse", "delay_ms": 0, "duration_ms": 400 }}
      ],
      "bg_color": "#hexcode of dominant background color"
    }}
    """

    response = model.generate_content([
        {"mime_type": "image/png", "data": image_bytes},
        prompt
    ])

    return json.loads(response.text)