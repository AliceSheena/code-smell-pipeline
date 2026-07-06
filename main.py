#!/usr/bin/env python3
"""Minimal CLI wrapper for common project tasks (prepare, smoke, quick-infer).

This is a lightweight convenience entrypoint to aid Colab runs and CI where
`main.py` is expected. It intentionally delegates to the existing scripts.
"""
import argparse
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd, check=True):
    print("$", " ".join(cmd))
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(res.stdout)
    if check and res.returncode != 0:
        raise SystemExit(res.returncode)


def prepare(input_path, output_dir):
    run_cmd([sys.executable, "data/prepare_dataset.py", "--input", input_path, "--output-dir", output_dir])


def smoke():
    run_cmd([sys.executable, "-m", "pytest", "-q", "tests/test_smoke.py"])


def quick_infer(n_examples: int = 2):
    # delegate to pipeline/end_to_end.py if available
    script = Path("pipeline/end_to_end.py")
    if script.exists():
        run_cmd([sys.executable, str(script), "--localise-only", "--input-examples", str(n_examples)])
    else:
        print("No pipeline/end_to_end.py found; skipping quick inference")


def main():
    parser = argparse.ArgumentParser(description="Lightweight project driver")
    sub = parser.add_subparsers(dest="cmd")

    p_prep = sub.add_parser("prepare")
    p_prep.add_argument("--input", required=True)
    p_prep.add_argument("--output-dir", default="./data")

    sub.add_parser("smoke")

    p_inf = sub.add_parser("quick-infer")
    p_inf.add_argument("--n", type=int, default=2)

    args = parser.parse_args()
    if args.cmd == "prepare":
        prepare(args.input, args.output_dir)
    elif args.cmd == "smoke":
        smoke()
    elif args.cmd == "quick-infer":
        quick_infer(args.n)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
