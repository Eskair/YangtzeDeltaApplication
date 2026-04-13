# src/tools/run_pipeline.py
# -*- coding: utf-8 -*-
import subprocess
from pathlib import Path

# 项目根目录（包含 src/）
BASE_DIR = Path(__file__).resolve().parents[2]


def run_cmd(cmd: list):
    """在项目根目录下执行一个子命令，并在失败时直接抛出异常。"""
    print("🚀 Running:", " ".join(cmd))
    r = subprocess.run(cmd, cwd=BASE_DIR)
    if r.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def run_full_pipeline():
    """
    一键跑完整流程（自动提案模式）：

      1) prepare_proposal_text.py
      2) extract_facts_by_chunk.py
      3) build_dimensions_from_facts.py
      4) generate_questions.py
      5) llm_answering.py
      6) post_processing.py
      7) ai_expert_opinion.py
      8) generate_final_report.py

    前提：每个脚本内部都实现了“自动检测最新 proposal”的逻辑，
    即在不传 --file / --proposal_id / --pid 的情况下能自己找到最新项目。
    """

    # 1) 准备文本
    run_cmd(["python", "src/tools/prepare_proposal_text.py"])

    # 2) facts 抽取
    run_cmd(["python", "src/tools/extract_facts_by_chunk.py"])

    # 2.5) fact verification against source text
    run_cmd(["python", "src/tools/verify_facts.py"])

    # 3) 由 facts 构建维度
    run_cmd(["python", "src/tools/build_dimensions_from_facts.py"])

    # 4) 生成问题
    run_cmd(["python", "src/tools/generate_questions.py"])

    # 5) LLM 回答
    run_cmd(["python", "src/tools/llm_answering.py"])

    # 6) post-processing
    run_cmd(["python", "src/tools/post_processing.py"])

    # 7) AI 专家意见
    run_cmd(["python", "src/tools/ai_expert_opinion.py"])

    # 8) 最终报告
    run_cmd(["python", "src/tools/generate_final_report.py"])

    print("🎯 Full pipeline finished.")


if __name__ == "__main__":
    # 不再需要任何命令行参数，直接按照“最新提案”自动跑一遍
    run_full_pipeline()
