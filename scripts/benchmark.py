import argparse
import json
import platform
import statistics

from visualguard.ingestion.demo import create_demo
from visualguard.pipeline import analyze
from visualguard.settings import Settings, local_caches, project_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthetic workflow timing, NOT ML throughput")
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("runs must be positive")
    local_caches()
    source = project_path("data/demo/benchmark.mp4")
    create_demo(source)
    timings = [
        analyze(source, project_path(f"runtime/benchmark/{i}"), "video", Settings(), demo=True)[
            "processing_seconds"
        ]
        for i in range(args.runs)
    ]
    report = {
        "synthetic": True,
        "scope": "Fixture processing with encoding; no ML inference",
        "platform": platform.platform(),
        "processor": platform.processor(),
        "runs_seconds": timings,
        "median_seconds": statistics.median(timings),
    }
    output = project_path("reports/synthetic_benchmark.json")
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(output)
