import argparse
import json

from visualguard.settings import project_path
from visualguard.training.construction import audit_prepare

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit and prepare a provisional PPE baseline")
    parser.add_argument("--source", default="data/external/construction-ppe-source/extracted")
    parser.add_argument("--output", default="data/processed/construction-ppe-v1")
    args = parser.parse_args()
    report = audit_prepare(
        project_path(args.source), project_path(args.output), project_path("reports/baseline")
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k
                not in {
                    "cross_source_split_duplicate_pairs",
                    "exact_duplicates_removed",
                    "rejected_images",
                }
            },
            indent=2,
        )
    )
    print("Cross-split duplicate pairs:", len(report["cross_source_split_duplicate_pairs"]))
