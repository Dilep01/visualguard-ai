from visualguard.domain import Box, Detection
from visualguard.vision.matching import associate
from visualguard.vision.tracker import IoUTracker


def detection(label, coords):
    return Detection(label, 0.9, Box(*coords))


def test_helmet_and_vest_match_to_person():
    person = detection("person", (50, 50, 150, 350))
    items = [
        person,
        detection("helmet", (75, 60, 125, 100)),
        detection("vest", (70, 140, 130, 250)),
    ]
    observation = associate([(1, person)], items, 500, 500)[0]
    assert observation.helmet == "present"
    assert observation.vest == "present"


def test_no_head_or_vest_is_unknown():
    person = detection("person", (50, 50, 150, 350))
    observation = associate([(1, person)], [person], 500, 500)[0]
    assert observation.helmet == observation.vest == "unknown"


def test_head_is_required_for_possible_missing_helmet():
    person = detection("person", (50, 50, 150, 350))
    head = detection("head", (75, 60, 125, 100))
    assert associate([(1, person)], [person, head], 500, 500)[0].helmet == "possible_missing"


def test_equipment_cannot_be_shared_between_people():
    person1 = detection("person", (50, 50, 150, 350))
    person2 = detection("person", (60, 50, 160, 350))
    helmet = detection("helmet", (75, 60, 125, 100))
    observations = associate([(1, person1), (2, person2)], [helmet], 500, 500)
    assert all(item.helmet == "unknown" for item in observations)


def test_border_crop_does_not_create_missing_alert():
    person = detection("person", (0, 0, 100, 300))
    head = detection("head", (25, 10, 75, 50))
    assert associate([(1, person)], [head], 500, 500)[0].helmet == "unknown"


def test_tracker_preserves_id_for_small_motion_then_expires():
    tracker = IoUTracker()
    person = detection("person", (50, 50, 150, 350))
    moved = detection("person", (55, 50, 155, 350))
    first = tracker.update([person], 0)[0][0]
    assert tracker.update([moved], 0.25)[0][0] == first
    assert tracker.update([moved], 2.0)[0][0] != first


def test_tracker_assigns_each_old_id_at_most_once():
    tracker = IoUTracker()
    person = detection("person", (50, 50, 150, 350))
    tracker.update([person], 0)
    result = tracker.update([person, person], 0.25)
    assert len({key for key, _ in result}) == 2
