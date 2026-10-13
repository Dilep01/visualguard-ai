"""Validation-only gates for selecting an experimental local checkpoint."""


def compare_candidate(baseline: dict, candidate: dict) -> dict:
    for report in (baseline, candidate):
        if report.get("split") != "val":
            raise ValueError("Candidate selection must use validation data only")
    for field in ("split_manifest_sha256", "dataset_config_sha256", "images", "imgsz"):
        if not baseline.get(field) or baseline[field] != candidate.get(field):
            raise ValueError(f"Comparison requires matching {field}")
    for field in ("confidence", "iou"):
        expected = 0.4 if field == "confidence" else 0.5
        if any(
            report["fixed_operating_point"][field] != expected for report in (baseline, candidate)
        ):
            raise ValueError(f"Predeclared comparison requires {field}={expected}")
    head_thresholds = [
        r["fixed_operating_point"].get("head_confidence", 0.4) for r in (baseline, candidate)
    ]
    if head_thresholds[0] != head_thresholds[1]:
        raise ValueError("Comparison requires matching head_confidence")
    old = baseline["fixed_operating_point"]["per_class"]
    new = candidate["fixed_operating_point"]["per_class"]
    gates = []
    for metric, minimum in (("precision", 0.70), ("recall", 0.30)):
        value = new["head"][metric]
        gates.append(
            {
                "criterion": f"head_{metric}",
                "minimum": minimum,
                "actual": value,
                "passed": value is not None and value >= minimum,
            }
        )
    for name in ("person", "helmet", "vest"):
        previous, current = old[name]["recall"], new[name]["recall"]
        delta = current - previous if current is not None and previous is not None else None
        gates.append(
            {
                "criterion": f"{name}_recall_change",
                "minimum": -0.05,
                "actual": delta,
                "passed": delta is not None and delta >= -0.05 - 1e-12,
            }
        )
    return {
        "split": "val",
        "images": candidate["images"],
        "imgsz": candidate["imgsz"],
        "confidence": 0.4,
        "head_confidence": head_thresholds[1],
        "iou": 0.5,
        "baseline_sha256": baseline["weights_sha256"],
        "candidate_sha256": candidate["weights_sha256"],
        "baseline_per_class": old,
        "candidate_per_class": new,
        "gates": gates,
        "eligible_for_experimental_use": all(gate["passed"] for gate in gates),
        "safety_validated": False,
        "scope": "Development selection only. Independent scene and video testing still required.",
    }
