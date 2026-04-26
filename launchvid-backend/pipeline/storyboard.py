import json
import os
from typing import Any

from groq import Groq


def generate_storyboard(prompt: str, vision_notes: str) -> dict[str, Any]:
    """
    Generate scene-by-scene storyboard and voiceover script with Groq.
    """
    api_key = os.getenv("GROQ_KEY")
    if not api_key:
        # Sensible fallback so local testing still works without keys.
        return {
            "title": "LaunchVid Draft",
            "scenes": [
                {"scene": 1, "visual": f"Opening shot inspired by: {prompt}", "duration": 3},
                {"scene": 2, "visual": "Product/value highlight", "duration": 4},
                {"scene": 3, "visual": "Call to action", "duration": 3},
            ],
            "voiceover": f"{prompt}. Start with a strong hook, highlight benefits, and end with a CTA.",
        }

    client = Groq(api_key=api_key)
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        temperature=0.6,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a video storyboard assistant. Respond only in valid JSON with keys: "
                    "title (string), scenes (array of {scene, visual, duration}), voiceover (string)."
                ),
            },
            {
                "role": "user",
                "content": f"Prompt: {prompt}\nVision notes: {vision_notes}",
            },
        ],
    )
    content = completion.choices[0].message.content or "{}"
    return json.loads(content)
