import argparse
import json
import sys

from .pipeline import Recipe
from .stages.encoder import was_processed
from . import develop


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="silver-halide",
        description="Develops an AI image into a photograph-grade JPEG: sensor noise, optics, CFA, ISP, grain.",
    )
    parser.add_argument("input", help="image to develop (PNG/JPEG)")
    parser.add_argument("output", nargs="?", help="JPEG path to write")
    parser.add_argument("--strength", choices=["subtle", "natural", "harsh"], default="natural", help="sensor character (default: natural)")
    parser.add_argument("--seed", type=int, default=7, help="deterministic output (default: 7)")
    parser.add_argument("--audit", metavar="REPORT", help="write a generation-defect report (JSON) for INPUT")
    parser.add_argument("--json", action="store_true", help="machine-readable result")
    args = parser.parse_args(argv)

    if args.audit:
        from .audit import audit

        overlay = args.audit.rsplit(".", 1)[0] + "-overlay.png"
        print(json.dumps(audit(args.input, overlay)))
        return 0
    if not args.output:
        parser.error("output is required unless --audit is used")

    try:
        if was_processed(args.input):
            message = f"nothing to do: {args.input} was already developed by silver-halide"
            return _finish(2, message, args)
        result = develop(Recipe(strength=args.strength, seed=args.seed), args.input, args.output)
    except FileNotFoundError:
        return _finish(1, f"input not found: {args.input} — check the path and try again", args)
    except PermissionError:
        return _finish(1, f"cannot write: {args.output} — check permissions on the target folder", args)
    except OSError as error:
        hint = str(error)
        if "cannot identify image file" in hint:
            return _finish(1, f"not an image: {args.input} — give me a PNG or JPEG", args)
        return _finish(1, f"could not read {args.input}: {hint}", args)
    return _finish(0, f"{args.output} written ({result['width']}x{result['height']}, {result['strength']})", args, result)


def _finish(code: int, message: str, args: argparse.Namespace, result: dict | None = None) -> int:
    if args.json:
        print(json.dumps({"ok": code == 0, "code": code, "message": message, "result": result}))
    else:
        print(message)
    return code


if __name__ == "__main__":
    sys.exit(main())
