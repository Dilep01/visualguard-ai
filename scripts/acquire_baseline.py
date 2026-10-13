"""Fetch explicitly selected upstream assets into the project, with provenance."""

import hashlib
import json
import stat
import subprocess
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import requests

from visualguard.settings import local_caches, project_path


def download(url: str, destination: Path, limit: int) -> dict:
    destination = project_path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        temporary = destination.with_suffix(destination.suffix + ".part")
        try:
            print(f"Fetching {destination.name}", flush=True)
            subprocess.run(
                [
                    "curl.exe",
                    "--fail",
                    "--location",
                    "--silent",
                    "--show-error",
                    "--connect-timeout",
                    "20",
                    "--max-time",
                    "300",
                    "--retry",
                    "1",
                    "--max-filesize",
                    str(limit),
                    "--output",
                    str(temporary),
                    url,
                ],
                check=True,
            )
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    return {
        "url": url,
        "file": str(destination),
        "bytes": destination.stat().st_size,
        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
    }


def extract_checked(archive: Path, destination: Path) -> None:
    destination = project_path(destination)
    with zipfile.ZipFile(archive) as zipped:
        members = zipped.infolist()
        if sum(item.file_size for item in members) > 2_000_000_000:
            raise ValueError("Archive exceeds extraction limit")
        for item in members:
            target = (destination / item.filename).resolve()
            if not target.is_relative_to(destination) or stat.S_ISLNK(item.external_attr >> 16):
                raise ValueError("Unsafe archive member")
        destination.mkdir(parents=True, exist_ok=True)
        for item in members:
            zipped.extract(item, destination)


if __name__ == "__main__":
    local_caches()
    root = project_path("data/external/construction-ppe-source")
    root.mkdir(parents=True, exist_ok=True)
    assets = []
    for name, url, limit in [
        (
            "construction-ppe.zip",
            "https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip",
            400_000_000,
        ),
        (
            "upstream.yaml",
            "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/datasets/construction-ppe.yaml",
            100_000,
        ),
        (
            "UPSTREAM-LICENSE.txt",
            "https://raw.githubusercontent.com/ultralytics/ultralytics/main/LICENSE",
            100_000,
        ),
    ]:
        item = download(url, root / name, limit)
        assets.append(item)
        print(f"Downloaded {name}: {item['bytes']} bytes", flush=True)
    # Resolve the model from the publisher's release listing, not a third-party checkpoint.
    response = requests.get(
        "https://api.github.com/repos/ultralytics/assets/releases/tags/v8.3.0", timeout=30
    )
    response.raise_for_status()
    asset = next(item for item in response.json()["assets"] if item["name"] == "yolo11n.pt")
    assets.append(
        download(asset["browser_download_url"], project_path("models/yolo11n.pt"), 20_000_000)
    )
    extract_checked(root / "construction-ppe.zip", root / "extracted")
    record = {
        "retrieved_at_utc": datetime.now(UTC).isoformat(),
        "assets": assets,
        "dataset_documentation": "https://docs.ultralytics.com/datasets/detect/construction-ppe/",
        "publisher_stated_license": "AGPL-3.0",
        "model_release": "v8.3.0",
        "use": "Local experimental baseline; no redistribution or deployment performed",
    }
    (root / "acquisition.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))
