from pathlib import Path

from gtts import gTTS


def create_voiceover(script: str, output_dir: str = "outputs") -> str:
    """
    Convert storyboard voiceover text to MP3.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir) / "voiceover.mp3"

    final_script = script.strip() or "Welcome to LaunchVid."
    tts = gTTS(text=final_script, lang="en")
    tts.save(str(output_path))
    return str(output_path)
