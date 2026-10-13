import json
from types import SimpleNamespace

from visualguard.training.checkpoints import HeadCheckpoint


def trainer(last, epoch, ap):
    box = SimpleNamespace(
        ap_class_index=[0, 1, 2, 3],
        class_result=lambda index: (0.8, 0.5, ap if index == 1 else 0.99, 0.2),
    )
    return SimpleNamespace(
        last=last, epoch=epoch, validator=SimpleNamespace(metrics=SimpleNamespace(box=box))
    )


def test_head_selection_keeps_best_head_model_across_resume(tmp_path):
    weights = tmp_path / "weights"
    weights.mkdir()
    last = weights / "last.pt"
    callback = HeadCheckpoint(tmp_path)
    last.write_bytes(b"first")
    callback(trainer(last, 0, 0.4))
    last.write_bytes(b"worse-head-despite-high-other-ap")
    callback(trainer(last, 1, 0.3))
    assert (weights / "best-head.pt").read_bytes() == b"first"
    restored = HeadCheckpoint(tmp_path)
    last.write_bytes(b"same-head")
    restored(trainer(last, 2, 0.4))
    assert (weights / "best-head.pt").read_bytes() == b"first"
    last.write_bytes(b"improved-head")
    restored(trainer(last, 3, 0.6))
    assert (weights / "best-head.pt").read_bytes() == b"improved-head"
    rows = json.loads((tmp_path / "head_selection.json").read_text())
    assert [row["selected"] for row in rows] == [True, False, False, True]
    assert not list(tmp_path.rglob("*.tmp"))
