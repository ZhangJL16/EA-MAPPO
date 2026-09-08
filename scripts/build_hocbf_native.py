from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "review_bundle" / "safety" / "collision" / "_qp_native.c"
OUTPUT = ROOT / "review_bundle" / "safety" / "collision" / "_qp_native.so"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the optional native three-variable HOCBF QP kernel"
    )
    parser.add_argument("--compiler", default=os.environ.get("CC", "cc"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    command = [
        args.compiler,
        "-O3",
        "-std=c11",
        "-fPIC",
        "-shared",
        str(SOURCE),
        "-lm",
        "-o",
        str(OUTPUT),
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    if not OUTPUT.is_file() or OUTPUT.stat().st_size == 0:
        raise RuntimeError("native HOCBF QP build did not produce a shared library")
    print(OUTPUT)


if __name__ == "__main__":
    main()
