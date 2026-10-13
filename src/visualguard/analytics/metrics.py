from collections import Counter

from visualguard.persistence.store import Store


def summarize(store: Store, synthetic: bool = False) -> dict:
    completed = store.completed_results(synthetic)
    ids = {job["id"] for job in completed}
    events = [event for event in store.list_events() if event["job_id"] in ids]
    seconds = sum(job["result"]["observed_person_seconds"] for job in completed)
    unknown = sum(job["result"]["unknown_helmet_person_seconds"] for job in completed)
    minutes = seconds / 60
    return {
        "synthetic": synthetic,
        "completed_videos": len(completed),
        "candidate_events": len(events),
        "review_counts": dict(Counter(event["review"] for event in events)),
        "observed_person_minutes": round(minutes, 4),
        "candidate_events_per_person_minute": len(events) / minutes if minutes else None,
        "unknown_helmet_fraction": unknown / seconds if seconds else None,
        "events_by_video_minute": dict(
            sorted(
                Counter(int(event["payload"]["start_seconds"] // 60) for event in events).items()
            )
        ),
        "events_by_zone": dict(Counter(event["payload"]["zone"] for event in events)),
        "denominator": "Sum of detected people times each sampled interval, video only. "
        "Repeated uploads count again. Not unique people or exposure time.",
    }
