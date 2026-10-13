from visualguard.domain import Detection, Observation


def associate(
    tracked: list[tuple[int, Detection]], detections: list[Detection], width: int, height: int
) -> list[Observation]:
    """Unique region containment. Ambiguous equipment never establishes absence."""
    matches: dict[int, dict[str, Detection]] = {key: {} for key, _ in tracked}
    ambiguous = set()
    for equipment in detections:
        if equipment.label == "person":
            continue
        owners = []
        for key, person in tracked:
            region = (
                person.box.region(0, 0.4)
                if equipment.label != "vest"
                else (person.box.region(0.15, 0.8))
            )
            if region.intersection(equipment.box) / max(equipment.box.area, 1e-9) >= 0.8:
                owners.append(key)
        if len(owners) == 1:
            existing = matches[owners[0]].get(equipment.label)
            if existing is None or existing.confidence < equipment.confidence:
                matches[owners[0]][equipment.label] = equipment
        elif len(owners) > 1:
            ambiguous.update(owners)

    observations = []
    for key, person in tracked:
        match = matches[key]
        box = person.box
        cropped = box.x1 <= 2 or box.y1 <= 2 or box.x2 >= width - 2 or box.y2 >= height - 2
        crowded = any(
            other_key != key and box.intersection(other.box) / max(box.area, 1e-9) > 0.2
            for other_key, other in tracked
        )
        helmet = "unknown"
        reason = "Head visibility is insufficient to infer helmet absence."
        evidence_confidence = person.confidence
        if key in ambiguous or cropped or crowded:
            reason = "Cropped, overlapping, or ambiguous person; helmet state is unknown."
        elif "helmet" in match:
            helmet = "present"
            evidence_confidence = min(person.confidence, match["helmet"].confidence)
            reason = "A helmet is uniquely associated with this person's upper region."
        elif "head" in match:
            helmet = "possible_missing"
            evidence_confidence = min(person.confidence, match["head"].confidence)
            reason = "Visible head detection with no associated helmet; human review required."
        vest = "present" if "vest" in match and key not in ambiguous and not crowded else "unknown"
        # Four object classes cannot establish that a torso is visible but unvested.
        reason += " Vest absence is not inferred from a missing detection."
        observations.append(Observation(key, box, helmet, vest, evidence_confidence, reason))
    return observations
