import os
import subprocess
import sys
import argparse

def run_step(step_name, script_name):
    print(f"\n{'='*60}\nRunning {step_name} ({script_name})\n{'='*60}")
    result = subprocess.run([sys.executable, script_name])
    if result.returncode != 0:
        print(f"❌ {step_name} failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    print(f"✅ {step_name} completed successfully.")

def main():
    parser = argparse.ArgumentParser(description="Run all lab steps.")
    parser.add_argument("--step", type=int, choices=[1, 2, 3, 4], help="Run a specific step")
    args = parser.parse_args()

    steps = [
        (1, "Step 1: LangSmith RAG Pipeline", "01_langsmith_rag_pipeline.py"),
        (2, "Step 2: Prompt Hub & A/B Routing", "02_prompt_hub_ab_routing.py"),
        (3, "Step 3: RAGAS Evaluation", "03_ragas_evaluation.py"),
        (4, "Step 4: Guardrails AI Validator", "04_guardrails_validator.py")
    ]

    for step_num, step_name, script_name in steps:
        if args.step is None or args.step == step_num:
            run_step(step_name, script_name)

if __name__ == "__main__":
    main()
