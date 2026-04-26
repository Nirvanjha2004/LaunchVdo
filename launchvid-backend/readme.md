# LaunchVid Backend

FastAPI backend for the LaunchVid pipeline.
Accepts Figma plugin exports → returns animated MP4 launch videos.

---

## Setup (5 minutes)

### 1. Install Python dependencies
```bash
cd launchvid-backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Paste your API keys into .env
```
GEMINI_API_KEY   → https://aistudio.google.com/app/apikey       (free)
GROQ_API_KEY     → https://console.groq.com/keys                (free)
SUPABASE_URL     → https://supabase.com → project → Settings → API
SUPABASE_KEY     → same page, "anon public" key
```

### 3. Set up Supabase

**Run this SQL once** in your Supabase project → SQL Editor:
```sql
-- Jobs table (tracks pipeline status)
create table if not exists jobs (
  id           text primary key,
  status       text not null default 'queued',
  frame_count  int,
  video_url    text,
  error        text,
  created_at   timestamptz default now(),
  updated_at   timestamptz default now()
);

-- Storage bucket for output videos
insert into storage.buckets (id, name, public)
values ('launchvid-outputs', 'launchvid-outputs', true)
on conflict do nothing;

-- Allow public reads on the bucket
create policy "Public read" on storage.objects
  for select using (bucket_id = 'launchvid-outputs');

-- Allow service role to insert
create policy "Service insert" on storage.objects
  for insert with check (bucket_id = 'launchvid-outputs');
```

### 4. Run the server
```bash
python main.py
# → http://localhost:8000
# → http://localhost:8000/docs  (Swagger UI — test all endpoints here)
```

---

## Testing without Remotion (stages 1–3 only)

Before setting up Remotion, you can test the first 3 pipeline stages by
temporarily commenting out the renderer call in main.py:

```python
# In run_pipeline(), comment out stages 4 and 5:
# mp4_path = await render_video(...)
# video_url = await upload_video(...)
await update_job(job_id, status="done", video_url="RENDER_NOT_YET_SET_UP")
```

Then send a test request:
```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"frames": [{"frameId":"test","frameName":"Test","width":390,"height":844,"fullPngBase64":"","layers":{}}], "appDescription": "a fintech app"}'
```

Check the job:
```bash
curl http://localhost:8000/job/{job_id_from_above}
```

---

## Pipeline stages

| Stage | Status value | What runs |
|---|---|---|
| 0 | `queued` | Job created |
| 1 | `analyzing` | Gemini 2.5 Flash vision per frame |
| 2 | `storyboarding` | Groq llama-3.3-70b storyboard |
| 3 | `generating_audio` | gTTS voiceover MP3s |
| 4 | `rendering` | Remotion MP4 render |
| ✅ | `done` | `video_url` is populated |
| ❌ | `failed` | `error` field has details |

---

## Project structure

```
launchvid-backend/
├── main.py               ← FastAPI app + pipeline orchestrator
├── config.py             ← loads .env
├── requirements.txt
├── .env                  ← YOUR API KEYS GO HERE
├── pipeline/
│   ├── vision.py         ← Gemini frame analysis
│   ├── storyboard.py     ← Groq storyboard
│   ├── tts.py            ← gTTS voiceovers
│   └── renderer.py       ← Remotion subprocess
└── db/
    └── supabase.py       ← job CRUD + video upload
```

---

## Cost (per video generated)

| Component | Cost |
|---|---|
| Gemini 2.5 Flash | Free (500 req/day) |
| Groq llama-3.3-70b | Free (14,400 req/day) |
| gTTS | Free (unlimited) |
| Supabase | Free tier (500MB storage) |
| Remotion | Free for individuals |
| **Total** | **₹0** |