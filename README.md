---
title: YangtzeDelta Proposal Analyser
emoji: 📋
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# YangtzeDelta Proposal Analyser

AI-powered proposal analysis pipeline built with FastAPI.

✅ 1. Open project
cd C:\Users\ubcas\Desktop\YangtzeDeltaApplication
code .
python -c "import fastapi" #If no error → everything is ready
🎯 ✅ 2. Make sure environment is ready (ONE TIME CHECK)

👉 .env exists and contains:

OPENAI_API_KEY=your_real_key
PROVIDER=openai
OPENAI_MODEL=gpt-4o-mini
ENABLE_VISION=false

👉 You already fixed UTF8 in server.py ✅
👉 So NO need for set ... anymore

🎯 ✅ 3. Run project
uvicorn src.api.server:app --reload
🎯 ✅ 4. Open browser

👉 Normal UI:

http://127.0.0.1:8000/

👉 Debug UI (backup):

http://127.0.0.1:8000/docs
🎯 ✅ 5. Demo flow (what you do)
Upload PDF
Click/run pipeline
Show progress
Show final report
⚠️ If something goes wrong (backup)

👉 Restart server:

Ctrl + C
uvicorn src.api.server:app --reload

👉 If API key issue:
→ check .env

🧠 One-line memory

👉 “open folder → run uvicorn → open browser → demo”

🔥 Optional (1-click demo)

Create run.bat:

uvicorn src.api.server:app --reload

Double click → done ✅

You’re fully ready now 👍
If you want, I can give you a 30-second speaking script for professor (very impactful)
