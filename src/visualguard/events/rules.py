from dataclasses import dataclass, field

from visualguard.domain import Observation


@dataclass
class Candidate:
    start: float
    last: float
    frames: int = 0
    confidence: float = 1.0
    evidence: dict = field(default_factory=dict)


class EventRules:
    """Only uninterrupted evidence opens an event. Unknown closes and resets it."""

    def __init__(self, min_frames=3, min_seconds=0.5, max_gap=0.75):
        self.min_frames = min_frames
        self.min_seconds = min_seconds
        self.max_gap = max_gap
        self.pending: dict[int, Candidate] = {}
        self.events: list[dict] = []

    def close(self, key: int) -> None:
        candidate = self.pending.pop(key)
        if candidate.frames >= self.min_frames and candidate.last - candidate.start >= (
            self.min_seconds
        ):
            self.events.append(
                {
                    "track_id": key,
                    "kind": "possible_missing_helmet",
                    "zone": "unassigned",
                    "start_seconds": candidate.start,
                    "end_seconds": candidate.last,
                    "evidence_frames": candidate.frames,
                    "confidence": candidate.confidence,
                    "evidence": candidate.evidence,
                    "rule": f"Visible head without matched helmet for >= {self.min_frames} "
                    f"consecutive samples spanning >= {self.min_seconds}s.",
                }
            )

    def update(self, observations: list[Observation], timestamp: float) -> None:
        missing = {
            item.track_id: item for item in observations if item.helmet == "possible_missing"
        }
        for key in list(self.pending):
            if key not in missing or timestamp - self.pending[key].last > self.max_gap:
                self.close(key)
        for key, observation in missing.items():
            if key not in self.pending:
                self.pending[key] = Candidate(timestamp, timestamp)
            candidate = self.pending[key]
            candidate.last = timestamp
            candidate.frames += 1
            candidate.confidence = min(candidate.confidence, observation.confidence)
            candidate.evidence = {
                "timestamp": timestamp,
                "box": vars(observation.box),
                "reason": observation.reason,
            }

    def finish(self) -> list[dict]:
        for key in list(self.pending):
            self.close(key)
        return self.events
