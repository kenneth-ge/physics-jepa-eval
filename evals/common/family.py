"""Shared CLI/loop harness so every eval family has the same shape.

A family declares its eval ids and a `generate(eval_id, out_dir, args)`
callback that renders A/B/C into out_dir (None in --check mode) and returns
a list of contract-failure strings (empty = OK). The harness handles
argument parsing, the --only subset filter, per-eval subdirectories, status
printing, and the exit code.
"""

import argparse
import pathlib
import sys


def family_cli(*, name, eval_ids, generate, subdir=str, description="",
               add_args=None):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--out-root", type=pathlib.Path, default=None)
    parser.add_argument("--check", action="store_true",
                        help="validate the contract only, no rendering")
    parser.add_argument("--only", nargs="+",
                        help="subset of eval ids (matched against str(id) or subdir name)")
    parser.add_argument("--rig", default="mono",
                        help="camera rig: mono (default, back-compat), stereo, tri")
    if add_args is not None:
        add_args(parser)
    args = parser.parse_args()
    if not args.check and args.out_root is None:
        parser.error("--out-root is required unless --check")

    selected = [e for e in eval_ids
                if args.only is None or str(e) in args.only or subdir(e) in args.only]
    if not selected:
        parser.error(f"--only matched no eval ids (available: {[str(e) for e in eval_ids]})")

    any_fail = False
    for e in selected:
        out_dir = None
        if not args.check:
            out_dir = args.out_root / subdir(e)
            out_dir.mkdir(parents=True, exist_ok=True)
        failures = generate(e, out_dir, args)
        print(f"[{name}:{subdir(e)}] contract {'OK' if not failures else 'FAILED'}")
        for f in failures:
            print(f"  - {f}")
        any_fail = any_fail or bool(failures)
    sys.exit(1 if any_fail else 0)
