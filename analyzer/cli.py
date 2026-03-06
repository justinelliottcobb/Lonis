"""analyzer/cli.py"""
import argparse
import json
import sys
from pathlib import Path

from .pipeline import Pipeline

def main():
    parser = argparse.ArgumentParser(
        prog="lonis",
        description="Lonis — Bitmap-to-structured-design-data analyzer",
    )
    subparsers = parser.add_subparsers(dest="command")

    analyze = subparsers.add_parser("analyze", help="Analyze an image file")
    analyze.add_argument("image", help="Path to image file")
    analyze.add_argument("-o", "--output", help="Write JSON to file instead of stdout")
    analyze.add_argument("--only", help="Comma-separated list of analyzers to run (e.g., color,edge)")
    analyze.add_argument("-v", "--verbose", action="store_true", help="Print progress to stderr")

    args = parser.parse_args()

    if args.command != "analyze":
        parser.print_help()
        sys.exit(1)

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: file not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    analyzers = None
    if args.only:
        analyzers = [a.strip() for a in args.only.split(",")]

    if args.verbose:
        names = analyzers or ["color", "spatial", "edge", "gradient", "texture"]
        print(f"Analyzing {image_path.name} with: {', '.join(names)}", file=sys.stderr)

    try:
        pipe = Pipeline(analyzers=analyzers)
        result = pipe.run(str(image_path))
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    output = json.dumps(result, indent=2)

    if args.output:
        Path(args.output).write_text(output)
        if args.verbose:
            print(f"Written to {args.output}", file=sys.stderr)
    else:
        print(output)

if __name__ == "__main__":
    main()
