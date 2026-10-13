from visualguard.domain import Box, Observation
from visualguard.events.rules import EventRules


def observation(status="possible_missing", key=1):
    return Observation(key, Box(10, 10, 100, 200), status, "unknown", 0.8, "test evidence")


def test_persistence_opens_one_event_and_recovery_closes_it():
    rules = EventRules()
    for timestamp in (0, 0.25, 0.5, 0.75):
        rules.update([observation()], timestamp)
    rules.update([observation("present")], 1.0)
    events = rules.finish()
    assert len(events) == 1
    assert events[0]["start_seconds"] == 0
    assert events[0]["end_seconds"] == 0.75
    assert events[0]["evidence_frames"] == 4


def test_single_frame_flicker_does_not_alert():
    rules = EventRules()
    rules.update([observation()], 0)
    rules.update([observation("present")], 0.25)
    assert rules.finish() == []


def test_unknown_breaks_consecutive_evidence():
    rules = EventRules()
    rules.update([observation()], 0)
    rules.update([observation()], 0.25)
    rules.update([observation("unknown")], 0.5)
    rules.update([observation()], 0.75)
    assert rules.finish() == []


def test_gap_does_not_bridge_missing_evidence():
    rules = EventRules()
    rules.update([observation()], 0)
    rules.update([observation()], 0.25)
    rules.update([observation()], 2.0)
    assert rules.finish() == []


def test_disappearing_track_closes_event_at_last_evidence():
    rules = EventRules()
    for timestamp in (0, 0.25, 0.5):
        rules.update([observation()], timestamp)
    rules.update([], 0.75)
    assert rules.finish()[0]["end_seconds"] == 0.5


def test_two_people_are_not_merged():
    rules = EventRules()
    for timestamp in (0, 0.25, 0.5):
        rules.update([observation(key=1), observation(key=2)], timestamp)
    assert len(rules.finish()) == 2


def test_frame_count_alone_is_not_enough():
    rules = EventRules()
    for timestamp in (0, 0.01, 0.02):
        rules.update([observation()], timestamp)
    assert rules.finish() == []
