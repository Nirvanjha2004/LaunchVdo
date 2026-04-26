import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY", "")
GROQ_API_KEY    = os.getenv("GROQ_API_KEY", "")
SUPABASE_URL    = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY    = os.getenv("SUPABASE_KEY", "")
OUTPUT_DIR      = os.getenv("OUTPUT_DIR", "/tmp/launchvid_outputs")
REMOTION_DIR    = os.getenv("REMOTION_DIR", "../launchvid-remotion")
PORT            = int(os.getenv("PORT", "8000"))

# Create output dir if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)