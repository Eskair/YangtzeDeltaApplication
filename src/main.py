from dotenv import load_dotenv
load_dotenv()

import os
import json
import argparse

from backend.chains.base_chain import BaseChain
from backend.chains.orchestrator import run_all, save_full_report

def parse_args():
    p = argparse.ArgumentParser(description="RAG-6View CLI")
    p.add_argument("--mode", choices=["single", "all"], default="all",
                   help="single=只跑一个维度; all=并行跑所有维度")
    p.add_argument("--dimension", default="team", help="在 single 模式下指定维度")
    p.add_argument("--question", default=None, help="自定义问题(可选)")
    p.add_argument("--max_workers", type=int, default=3, help="并行线程数")
    return p.parse_args()

def run_single(dim: str, question: str = None):
    print("🚀 Starting RAG Demo (Single)...")
    chain = BaseChain(dim)
    q = question or "Does the core team have strong research capability?"
    result = chain.run(q)
    os.makedirs("data/results", exist_ok=True)
    out = "data/results/single_result.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n✅ 单维度分析完成！结果已保存到: {out}")
    print(json.dumps(result, indent=2, ensure_ascii=False))

def run_all_dims(max_workers: int):
    print("🚀 Starting RAG Demo (ALL Dimensions, parallel)...")
    report = run_all(max_workers=max_workers)
    out_path = save_full_report(report)
    print(f"\n✅ 全维度分析完成！报告已保存到: {out_path}")
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))

def main():
    args = parse_args()
    if args.mode == "single":
        run_single(args.dimension, args.question)
    else:
        run_all_dims(args.max_workers)

if __name__ == "__main__":
    main()
