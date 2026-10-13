from visualguard.domain import Box, Detection


class IoUTracker:
    """Small baseline tracker, scoped to a single clip; no cross-camera identity."""

    def __init__(self, max_gap: float = 0.75, min_iou: float = 0.25):
        self.max_gap = max_gap
        self.min_iou = min_iou
        self.tracks: dict[int, tuple[Box, float]] = {}
        self.next_id = 1

    def update(self, people: list[Detection], timestamp: float) -> list[tuple[int, Detection]]:
        self.tracks = {
            key: value for key, value in self.tracks.items() if timestamp - value[1] <= self.max_gap
        }
        candidates = sorted(
            (
                (box.iou(person.box), key, index)
                for key, (box, _) in self.tracks.items()
                for index, person in enumerate(people)
            ),
            reverse=True,
        )
        assigned: dict[int, int] = {}
        used = set()
        for score, key, index in candidates:
            if score >= self.min_iou and key not in used and index not in assigned:
                assigned[index] = key
                used.add(key)
        result = []
        for index, person in enumerate(people):
            key = assigned.get(index)
            if key is None:
                key = self.next_id
                self.next_id += 1
            self.tracks[key] = person.box, timestamp
            result.append((key, person))
        return result
