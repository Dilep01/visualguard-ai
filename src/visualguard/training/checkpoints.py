"""Keep the checkpoint selected by uncovered-head validation AP50."""

import json
import shutil

from visualguard.settings import project_path


class HeadCheckpoint:
    def __init__(self, output):
        self.output = project_path(output)
        self.report = self.output / "head_selection.json"
        self.history = json.loads(self.report.read_text()) if self.report.exists() else []

    def __call__(self, trainer):
        metrics = trainer.validator.metrics.box
        classes = list(map(int, metrics.ap_class_index))
        head_ap = float(metrics.class_result(classes.index(1))[2]) if 1 in classes else 0.0
        previous_best = max((row["head_ap50"] for row in self.history), default=-1.0)
        selected = head_ap > previous_best
        if selected:
            temporary = self.output / "weights/best-head.pt.tmp"
            shutil.copy2(project_path(trainer.last), temporary)
            temporary.replace(self.output / "weights/best-head.pt")
        self.history.append(
            {"epoch": trainer.epoch + 1, "head_ap50": head_ap, "selected": selected}
        )
        temporary = self.report.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.history, indent=2), encoding="utf-8")
        temporary.replace(self.report)
