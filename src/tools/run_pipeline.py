# src/tools/run_pipeline.py
# -*- coding: utf-8 -*-
import subprocess
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

PIPELINE_STAGES = [
    "prepare_proposal_text",
    "extract_facts_by_chunk",
    "verify_facts",
    "build_dimensions_from_facts",
    "generate_questions",
    "llm_answering",
    "post_processing",
    "ai_expert_opinion",
    "generate_final_report",
]


def run_cmd(cmd: list):
    """Execute a subprocess command from the project root directory."""
    print("🚀 Running:", " ".join(cmd))
    r = subprocess.run(cmd, cwd=BASE_DIR)
    if r.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def run_full_pipeline(resume: bool = False):
    """
    Run the complete pipeline. If resume=True, skip already-completed stages
    using the checkpoint system.

    Pipeline stages:
      1) prepare_proposal_text.py
      2) extract_facts_by_chunk.py
      2.5) verify_facts.py
      3) build_dimensions_from_facts.py
      4) generate_questions.py
      5) llm_answering.py
      6) post_processing.py
      7) ai_expert_opinion.py
      8) generate_final_report.py
    """
    ckpt = None
    if resume:
        try:
            from src.tools.checkpoint import PipelineCheckpoint
            prepared_dir = BASE_DIR / "src" / "data" / "prepared"
            if prepared_dir.exists():
                candidates = [
                    (d.stat().st_mtime, d.name)
                    for d in prepared_dir.iterdir() if d.is_dir()
                ]
                if candidates:
                    pid = max(candidates, key=lambda x: x[0])[1]
                    ckpt = PipelineCheckpoint(pid)
                    last = ckpt.get_last_completed_stage()
                    if last:
                        print(f"📋 Resuming pipeline for {pid}. Last completed: {last}")
        except Exception as e:
            print(f"[WARN] Could not load checkpoint: {e}")

    for stage in PIPELINE_STAGES:
        if ckpt and ckpt.is_stage_complete(stage):
            print(f"⏭️  Skipping {stage} (already complete)")
            continue

        run_cmd(["python", f"src/tools/{stage}.py"])

        if ckpt:
            ckpt.mark_complete(stage)

    print("🎯 Full pipeline finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full review pipeline")
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from the last completed stage (skip completed stages)"
    )
    args = parser.parse_args()
    run_full_pipeline(resume=args.resume)
