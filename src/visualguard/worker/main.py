import argparse
import logging
import time
from pathlib import Path

from visualguard.persistence.store import Store
from visualguard.pipeline import analyze
from visualguard.settings import Settings, local_caches

logger = logging.getLogger(__name__)


def process_job(store: Store, settings: Settings, job: dict) -> None:
    last_update = 0.0

    def progress(value):
        nonlocal last_update
        if time.monotonic() - last_update >= 2:
            store.heartbeat(job, value)
            last_update = time.monotonic()

    try:
        output = settings.storage_path / "artifacts" / job["id"] / job["token"]
        result = analyze(
            Path(job["source"]),
            output,
            job["kind"],
            settings,
            demo=job["synthetic"] == "yes",
            progress=progress,
        )
        store.complete(job, result, output)
    except Exception as exc:
        logger.exception("Job %s failed", job["id"])
        store.fail(job, str(exc)[:500])


def main():
    parser = argparse.ArgumentParser(description="VisualGuard durable video worker")
    parser.add_argument("--once", action="store_true", help="Process one queued video and exit")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    local_caches()
    settings = Settings()
    store = Store(settings)
    try:
        while True:
            job = store.claim()
            if job:
                logger.info("Processing %s", job["id"])
                process_job(store, settings, job)
            if args.once:
                break
            if not job:
                time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Worker stopped; interrupted jobs can be reclaimed after lease expiry.")
    finally:
        store.engine.dispose()


if __name__ == "__main__":
    main()
