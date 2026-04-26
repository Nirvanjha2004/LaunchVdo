"""
pipeline/tts.py
───────────────
Generates MP3 voiceover files for each scene using gTTS (Google TTS).
Completely free, no API key needed.

To upgrade to higher quality voices later, swap gTTS for OpenAI TTS:
    pip install openai
    client = OpenAI(api_key=OPENAI_KEY)
    response = client.audio.speech.create(model="tts-1", voice="nova", input=text)
"""

import os
import asyncio
from pathlib import Path
from gtts import gTTS
from config import OUTPUT_DIR


async def generate_voiceover(text: str, output_path: str) -> bool:
    """
    Generate a single MP3 voiceover file.

    Args:
        text:        The narration text to speak
        output_path: Full path where the MP3 should be saved

    Returns:
        True on success, False on failure
    """
    if not text or not text.strip():
        print(f"[tts] Skipping empty narration for {output_path}")
        return False

    try:
        # gTTS is synchronous — run in executor to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _generate_sync, text.strip(), output_path)
        print(f"[tts] Generated: {output_path}")
        return True
    except Exception as e:
        print(f"[tts] Failed for text '{text[:40]}...': {e}")
        return False


def _generate_sync(text: str, output_path: str):
    """Synchronous gTTS call (runs in thread executor)."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tts = gTTS(text=text, lang="en", slow=False, tld="com")
    tts.save(output_path)


async def generate_all_voiceovers(scenes: list[dict], job_id: str) -> list[dict]:
    """
    Generate voiceover MP3s for all scenes that have narration.

    Args:
        scenes:  list of scene dicts from storyboard.generate_storyboard()
        job_id:  used to namespace the output files

    Returns:
        The same scenes list with "audio_path" added to each scene
        (None if no narration or generation failed)
    """
    job_audio_dir = os.path.join(OUTPUT_DIR, job_id, "audio")
    os.makedirs(job_audio_dir, exist_ok=True)

    enriched = []
    for scene in scenes:
        narration = scene.get("narration", "").strip()
        audio_path = None

        if narration:
            file_path = os.path.join(job_audio_dir, f"scene_{scene['scene_index']:03d}.mp3")
            success   = await generate_voiceover(narration, file_path)
            audio_path = file_path if success else None

        enriched.append({**scene, "audio_path": audio_path})

    successful = sum(1 for s in enriched if s.get("audio_path"))
    print(f"[tts] Generated {successful}/{len(scenes)} voiceovers for job {job_id}")
    return enriched