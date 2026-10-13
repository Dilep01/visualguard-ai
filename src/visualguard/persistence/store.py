import time
import uuid

from sqlalchemy import (
    JSON,
    Column,
    Float,
    MetaData,
    String,
    Table,
    and_,
    create_engine,
    insert,
    or_,
    select,
    update,
)
from sqlalchemy import event as sql_event

from visualguard.settings import Settings

metadata = MetaData()
jobs = Table(
    "jobs",
    metadata,
    Column("id", String, primary_key=True),
    Column("kind", String, nullable=False),
    Column("status", String, nullable=False),
    Column("created_at", Float, nullable=False),
    Column("progress", Float, default=0),
    Column("source", String, nullable=False),
    Column("synthetic", String, nullable=False),
    Column("result", JSON),
    Column("error", String),
    Column("lease_until", Float, default=0),
    Column("token", String),
    Column("output", String),
)
events = Table(
    "events",
    metadata,
    Column("id", String, primary_key=True),
    Column("job_id", String, nullable=False, index=True),
    Column("payload", JSON, nullable=False),
    Column("review", String, default="pending"),
    Column("note", String, default=""),
)


class Store:
    def __init__(self, settings: Settings):
        url = settings.database_url or f"sqlite:///{settings.storage_path / 'visualguard.db'}"
        self.engine = create_engine(
            url, connect_args={"timeout": 30} if not (settings.database_url) else {}
        )
        if not settings.database_url:

            @sql_event.listens_for(self.engine, "connect")
            def configure_sqlite(connection, _):
                connection.execute("PRAGMA journal_mode=WAL")

        # Serialize first-start schema creation across API and worker processes.
        with self.engine.begin() as connection:
            if settings.database_url:
                connection.exec_driver_sql("SELECT pg_advisory_xact_lock(874291)")
            else:
                connection.exec_driver_sql("BEGIN IMMEDIATE")
            metadata.create_all(connection)

    def create(self, kind, source, synthetic=False) -> str:
        job_id = uuid.uuid4().hex
        with self.engine.begin() as connection:
            connection.execute(
                insert(jobs).values(
                    id=job_id,
                    kind=kind,
                    status="queued",
                    created_at=time.time(),
                    source=str(source),
                    progress=0,
                    synthetic="yes" if synthetic else "no",
                    lease_until=0,
                )
            )
        return job_id

    def get(self, job_id) -> dict | None:
        with self.engine.connect() as connection:
            row = connection.execute(select(jobs).where(jobs.c.id == job_id)).mappings().first()
            return dict(row) if row else None

    def list_jobs(self) -> list[dict]:
        with self.engine.connect() as connection:
            columns = [
                jobs.c.id,
                jobs.c.kind,
                jobs.c.status,
                jobs.c.created_at,
                jobs.c.progress,
                jobs.c.synthetic,
                jobs.c.error,
            ]
            return [
                dict(row)
                for row in connection.execute(
                    select(*columns).order_by(jobs.c.created_at.desc()).limit(100)
                ).mappings()
            ]

    def claim(self, job_id=None) -> dict | None:
        now = time.time()
        eligible = or_(
            jobs.c.status == "queued", and_(jobs.c.status == "running", jobs.c.lease_until < now)
        )
        query = select(jobs.c.id).where(eligible)
        query = query.where(jobs.c.id == job_id) if job_id else query.where(jobs.c.kind == "video")
        with self.engine.begin() as connection:
            candidate = connection.execute(query.order_by(jobs.c.created_at).limit(1)).scalar()
            if candidate is None:
                return None
            token = uuid.uuid4().hex
            result = connection.execute(
                update(jobs)
                .where(jobs.c.id == candidate, eligible)
                .values(
                    status="running", token=token, lease_until=now + 900, progress=0, error=None
                )
            )
            if result.rowcount != 1:
                return None
        return self.get(candidate)

    def heartbeat(self, job, progress):
        with self.engine.begin() as connection:
            result = connection.execute(
                update(jobs)
                .where(
                    jobs.c.id == job["id"], jobs.c.token == job["token"], jobs.c.status == "running"
                )
                .values(progress=progress, lease_until=time.time() + 900)
            )
            if result.rowcount != 1:
                raise RuntimeError("Job lease was lost")

    def complete(self, job, result, output):
        with self.engine.begin() as connection:
            changed = connection.execute(
                update(jobs)
                .where(
                    jobs.c.id == job["id"], jobs.c.token == job["token"], jobs.c.status == "running"
                )
                .values(status="completed", progress=1, result=result, output=str(output))
            )
            if changed.rowcount != 1:
                raise RuntimeError("Job lease was lost before completion")
            for payload in result["events"]:
                connection.execute(
                    insert(events).values(
                        id=uuid.uuid4().hex,
                        job_id=job["id"],
                        payload=payload,
                        review="pending",
                        note="",
                    )
                )

    def fail(self, job, error):
        with self.engine.begin() as connection:
            connection.execute(
                update(jobs)
                .where(jobs.c.id == job["id"], jobs.c.token == job["token"])
                .values(status="failed", error=error)
            )

    def list_events(self, job_id=None, review=None) -> list[dict]:
        query = select(events, jobs.c.synthetic).join(jobs, events.c.job_id == jobs.c.id)
        if job_id:
            query = query.where(events.c.job_id == job_id)
        if review:
            query = query.where(events.c.review == review)
        with self.engine.connect() as connection:
            return [dict(row) for row in connection.execute(query).mappings()]

    def review(self, event_id, status, note):
        with self.engine.begin() as connection:
            return (
                connection.execute(
                    update(events).where(events.c.id == event_id).values(review=status, note=note)
                ).rowcount
                == 1
            )

    def completed_results(self, synthetic: bool) -> list[dict]:
        with self.engine.connect() as connection:
            return [
                dict(row)
                for row in connection.execute(
                    select(jobs).where(
                        jobs.c.status == "completed",
                        jobs.c.kind == "video",
                        jobs.c.synthetic == ("yes" if synthetic else "no"),
                    )
                ).mappings()
            ]
