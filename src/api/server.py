# -*- coding: utf-8 -*-
"""
FastAPI server for YangtzeDelta Proposal Analysis
"""

import os
os.environ["PYTHONUTF8"] = "1"
import sys
import json
import uuid
import asyncio
import smtplib
import subprocess
from email.message import EmailMessage
from pathlib import Path
from datetime import datetime, timezone


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import (
    StreamingResponse, FileResponse, HTMLResponse, JSONResponse
)
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

# ─── paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parents[2]
SRC_DIR    = BASE_DIR / "src"
DATA_DIR   = SRC_DIR  / "data"
PROPOSALS  = DATA_DIR / "proposals"
REPORTS    = DATA_DIR / "reports"
FRONTEND   = BASE_DIR / "frontend"

PIPELINE_STEPS = [
    ("prepare_proposal_text",       "Extracting text from document"),
    ("extract_facts_by_chunk",      "Extracting key facts"),
    ("verify_facts",                "Verifying extracted facts"),
    ("build_dimensions_from_facts", "Building analysis dimensions"),
    ("generate_questions",          "Generating evaluation questions"),
    ("llm_answering",               "Running LLM analysis"),
    ("post_processing",             "Post-processing answers"),
    ("ai_expert_opinion",           "Generating expert opinion"),
    ("generate_final_report",       "Compiling final report"),
]

# ─── in-memory job store ───────────────────────────────────────────────────────
# job_id → {"status": str, "pid": str, "steps": [...], "error": str|None}
jobs: dict[str, dict] = {}

# ─── app ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="YangtzeDelta Proposal Analyser")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def send_report_email(to: str, report_path: str, original_filename: str) -> None:
    """Send the finished report — tries Resend API first, falls back to SMTP."""
    try:
        report_data = Path(report_path).read_bytes()
    except FileNotFoundError:
        print(f"[email] Report file not found: {report_path}", flush=True)
        return

    report_name = Path(report_path).name

    # ── Option 1: Resend API (zero SMTP config needed) ────────────────────────
    resend_key = os.getenv("RESEND_API_KEY", "")
    if resend_key:
        import base64
        import requests as _requests
        try:
            resp = _requests.post(
                "https://api.resend.com/emails",
                headers={"Authorization": f"Bearer {resend_key}"},
                json={
                    "from":    "Yangtze Delta <onboarding@resend.dev>",
                    "to":      [to],
                    "subject": f"✅ Your Proposal Analysis is Ready — {original_filename}",
                    "text":    (
                        f"Hello,\n\n"
                        f"Great news! Your proposal \"{original_filename}\" has been fully analysed by the Yangtze Delta AI system.\n\n"
                        f"📎 The complete report is attached to this email as a Markdown (.md) file. "
                        f"You can open it with any Markdown viewer, Notion, Obsidian, VS Code, or a plain text editor.\n\n"
                        f"The report covers:\n"
                        f"  • Team & Background\n"
                        f"  • Objectives & Vision\n"
                        f"  • Market Strategy\n"
                        f"  • Innovation & Technology\n"
                        f"  • Feasibility & Risk\n"
                        f"  • Overall Expert Opinion\n\n"
                        f"If you have any questions or would like to run another analysis, visit the platform and upload a new proposal.\n\n"
                        f"Best regards,\n"
                        f"Yangtze Delta AI Analysis System"
                    ),
                    "attachments": [{
                        "filename": report_name,
                        "content":  base64.b64encode(report_data).decode(),
                    }],
                },
                timeout=15,
            )
            if resp.status_code == 200 or resp.status_code == 201:
                print(f"[email] Report sent to {to} via Resend ✓", flush=True)
                return
            else:
                print(f"[email] Resend error {resp.status_code}: {resp.text}", flush=True)
        except Exception as exc:
            print(f"[email] Resend exception: {exc}", flush=True)

    # ── Option 2: SMTP fallback ───────────────────────────────────────────────
    host  = os.getenv("SMTP_HOST", "")
    port  = int(os.getenv("SMTP_PORT", "587"))
    user  = os.getenv("SMTP_USER", "")
    pwd   = os.getenv("SMTP_PASS", "")
    from_ = os.getenv("SMTP_FROM", user)

    if not all([host, user, pwd]):
        print("[email] No email provider configured (set RESEND_API_KEY or SMTP_* vars) — skipping", flush=True)
        return

    msg = EmailMessage()
    msg["Subject"] = f"✅ Your Proposal Analysis is Ready — {original_filename}"
    msg["From"]    = from_
    msg["To"]      = to
    msg.set_content(
        f"Hello,\n\n"
        f"Great news! Your proposal \"{original_filename}\" has been fully analysed by the Yangtze Delta AI system.\n\n"
        f"📎 The complete report is attached to this email as a Markdown (.md) file. "
        f"You can open it with any Markdown viewer, Notion, Obsidian, VS Code, or a plain text editor.\n\n"
        f"The report covers:\n"
        f"  • Team & Background\n"
        f"  • Objectives & Vision\n"
        f"  • Market Strategy\n"
        f"  • Innovation & Technology\n"
        f"  • Feasibility & Risk\n"
        f"  • Overall Expert Opinion\n\n"
        f"If you have any questions or would like to run another analysis, visit the platform and upload a new proposal.\n\n"
        f"Best regards,\n"
        f"Yangtze Delta AI Analysis System"
    )
    msg.add_attachment(report_data, maintype="text", subtype="markdown",
                       filename=report_name)

    try:
        with smtplib.SMTP(host, port) as smtp:
            smtp.starttls()
            smtp.login(user, pwd)
            smtp.send_message(msg)
        print(f"[email] Report sent to {to} via SMTP", flush=True)
    except Exception as exc:
        print(f"[email] SMTP failed: {exc}", flush=True)


def python_bin() -> str:
    """Return the venv python that is running this server."""
    return sys.executable


def _build_step_cmd(script_name: str, pid: str, upload_path: str) -> list[str]:
    """Build the full command for a pipeline script with explicit pid args."""
    qs_file = str(DATA_DIR / "questions" / pid / "generated_questions.json")
    args_map = {
        "prepare_proposal_text":       ["--file", upload_path, "--proposal_id", pid],
        "extract_facts_by_chunk":      ["--proposal_id", pid],
        "verify_facts":                ["--proposal_id", pid],
        "build_dimensions_from_facts": ["--proposal_id", pid],
        "generate_questions":          ["--proposal_id", pid],
        "llm_answering":               ["--proposal_id", pid, "--qs_file", qs_file],
        "post_processing":             ["--pid", pid],
        "ai_expert_opinion":           ["--pid", pid],
        "generate_final_report":       ["--pid", pid],
    }
    extra = args_map.get(script_name, [])
    return [python_bin(), f"src/tools/{script_name}.py"] + extra


def _run_step(script_name: str, pid: str, upload_path: str) -> tuple[bool, str]:
    """Run one pipeline script, stream output to terminal, return (ok, tail)."""
    cmd = _build_step_cmd(script_name, pid, upload_path)
    print(f"\n{'='*60}", flush=True)
    print(f"▶ STEP: {script_name}  [pid={pid}]", flush=True)
    print(f"{'='*60}", flush=True)
    proc = subprocess.Popen(
        cmd,
        cwd=str(BASE_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    lines: list[str] = []
    for line in proc.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    proc.wait()
    ok = proc.returncode == 0
    tail = "".join(lines[-50:])
    if not ok:
        print(f"✗ STEP FAILED (exit {proc.returncode}): {script_name}", flush=True)
    else:
        print(f"✓ STEP DONE: {script_name}", flush=True)
    return ok, tail


async def _run_pipeline(job_id: str, pid: str):
    """Background coroutine that drives the pipeline and updates job state."""
    job = jobs[job_id]
    job["status"] = "running"
    upload_path = job.get("upload_path", "")
    progress_file = DATA_DIR / f"step_progress_{pid}.json"

    for i, (script, label) in enumerate(PIPELINE_STEPS):
        job["current_step"] = i
        job["steps"][i]["status"] = "running"
        job["steps"][i]["started_at"] = _utcnow()

        # reset sub-step progress for the new step
        try:
            progress_file.write_text('{"done":0,"total":0}', encoding="utf-8")
        except Exception:
            pass

        # run in thread so we don't block the event loop
        loop = asyncio.get_running_loop()
        ok, tail = await loop.run_in_executor(
            None, _run_step, script, pid, upload_path
        )

        job["steps"][i]["finished_at"] = _utcnow()
        if ok:
            job["steps"][i]["status"] = "done"
        else:
            job["steps"][i]["status"] = "error"
            job["steps"][i]["error"] = tail
            job["status"] = "error"
            job["error"] = f"Step '{label}' failed:\n{tail}"
            return

    job["status"] = "done"
    job["current_step"] = len(PIPELINE_STEPS)

    # clean up per-pid progress file
    try:
        progress_file.unlink(missing_ok=True)
    except Exception:
        pass

    # locate the report
    report_path = REPORTS / f"{pid}_final_report.md"
    if report_path.exists():
        job["report_path"] = str(report_path)
    else:
        matches = list(REPORTS.glob(f"{pid}*.md"))
        job["report_path"] = str(matches[0]) if matches else ""

    # send email if address was provided
    recipient = job.get("email", "")
    if recipient and job.get("report_path"):
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None, send_report_email, recipient, job["report_path"], job["filename"]
        )


# ─── routes ───────────────────────────────────────────────────────────────────

@app.post("/api/upload")
async def upload_proposal(
    file: UploadFile = File(...),
    email: str = Form(""),
):
    """Accept a proposal file, save it, return a job_id."""
    allowed = {".pdf", ".docx", ".doc", ".txt", ".md", ".pptx", ".ppt"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(400, f"Unsupported file type: {suffix}")

    PROPOSALS.mkdir(parents=True, exist_ok=True)

    # derive a clean proposal id from filename — append short job_id for uniqueness
    job_id = str(uuid.uuid4())
    stem = Path(file.filename).stem
    safe_stem = "".join(c if (c.isalnum() or c in "-_") else "_" for c in stem)
    pid = f"{safe_stem or 'proposal'}_{job_id[:8]}"

    dest = PROPOSALS / f"{pid}{suffix}"
    content = await file.read()
    dest.write_bytes(content)

    jobs[job_id] = {
        "job_id":       job_id,
        "pid":          pid,
        "filename":     file.filename,
        "upload_path":  str(dest),
        "email":        email.strip(),
        "status":       "queued",
        "current_step": -1,
        "steps": [
            {"name": script, "label": label, "status": "pending",
             "started_at": None, "finished_at": None, "error": None}
            for script, label in PIPELINE_STEPS
        ],
        "error":       None,
        "report_path": "",
        "created_at":  _utcnow(),
    }
    return {"job_id": job_id, "pid": pid}


@app.post("/api/run/{job_id}")
async def run_pipeline(job_id: str):
    """Start the pipeline for an already-uploaded job."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    job = jobs[job_id]
    if job["status"] not in ("queued",):
        raise HTTPException(400, f"Job is already in state: {job['status']}")

    asyncio.create_task(_run_pipeline(job_id, job["pid"]))
    return {"started": True}


@app.patch("/api/jobs/{job_id}/email")
async def update_email(job_id: str, payload: dict):
    """Update or add a notification email for an existing job."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    email = (payload.get("email") or "").strip()
    jobs[job_id]["email"] = email
    return {"ok": True, "email": email}


@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    """Poll job status."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    return jobs[job_id]


@app.get("/api/events/{job_id}")
async def sse_events(job_id: str):
    """
    Server-Sent Events stream — pushes job state every 1 s until done/error.
    """
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")

    async def generator():
        prog_file = DATA_DIR / f"step_progress_{jobs[job_id]['pid']}.json"
        while True:
            job = jobs.get(job_id, {})
            job_copy = {**job, "steps": [dict(s) for s in job.get("steps", [])]}
            if prog_file.exists():
                try:
                    prog = json.loads(prog_file.read_text(encoding="utf-8"))
                    for step in job_copy["steps"]:
                        if step["status"] == "running":
                            step["sub_progress"] = prog
                            break
                except Exception:
                    pass
            data = json.dumps(job_copy)
            yield f"data: {data}\n\n"
            if job.get("status") in ("done", "error"):
                break
            await asyncio.sleep(1)

    return StreamingResponse(generator(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.get("/api/report/{job_id}")
async def get_report(job_id: str):
    """Return the final Markdown report as plain text."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    job = jobs[job_id]
    if job["status"] != "done":
        raise HTTPException(400, "Pipeline not finished yet")
    rp = job.get("report_path", "")
    if not rp or not Path(rp).exists():
        raise HTTPException(404, "Report file not found")
    return JSONResponse({"markdown": Path(rp).read_text(encoding="utf-8"),
                         "pid": job["pid"]})


@app.get("/api/download/{job_id}")
async def download_report(job_id: str):
    """Download the final Markdown report as a file."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    job = jobs[job_id]
    if job["status"] != "done":
        raise HTTPException(400, "Pipeline not finished yet")
    rp = job.get("report_path", "")
    if not rp or not Path(rp).exists():
        raise HTTPException(404, "Report file not found")
    return FileResponse(
        path=rp,
        media_type="text/markdown",
        filename=f"{job['pid']}_final_report.md",
    )


# ─── frontend ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    index = FRONTEND / "index.html"
    return HTMLResponse(index.read_text(encoding="utf-8"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.server:app", host="0.0.0.0", port=8000, reload=False)
