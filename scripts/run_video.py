import argparse
import json

from visualguard.ingestion.demo import create_demo
from visualguard.pipeline import analyze
from visualguard.settings import Settings, local_caches, project_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze one clip without the API or worker")
    parser.add_argument("--input", help="Existing video path inside this project")
    parser.add_argument("--output", default="runtime/cli-demo")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    local_caches()
    if args.demo and args.input:
        parser.error("--demo generates its own fixture; do not supply --input")
    if not args.demo and not args.input:
        parser.error("Supply --input or --demo")
    source = project_path("data/demo/synthetic.mp4" if args.demo else args.input)
    output = project_path(args.output)
    if args.demo:
        # Always regenerate: the synthetic detector must never analyze arbitrary real uploads.
        create_demo(source)
    report = analyze(source, output, "video", Settings(), demo=args.demo)
    (output / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"{len(report['events'])} events; result: {output / 'result.json'}")
