"""Download the NAB ``realAWSCloudwatch`` subset used for evaluation.

    python scripts/fetch_nab_dataset.py            # download if missing
    python scripts/fetch_nab_dataset.py --force    # re-download
    python scripts/fetch_nab_dataset.py --summary  # print what is cached

Files land in ``data/raw/nab/`` (git-ignored) together with a ``manifest.json``
recording the source URL and a sha256 per file, so a later run can prove the
data has not changed underneath published results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import PROJECT_ROOT
from src.data.timeseries import NAB_BRANCH, NAB_DATASET, NAB_REPO

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "nab"
DATA_DIR = RAW_DIR / NAB_DATASET
LABELS_FILE = RAW_DIR / "combined_windows.json"
MANIFEST_FILE = RAW_DIR / "manifest.json"
BASE_URL = f"https://raw.githubusercontent.com/{NAB_REPO}/{NAB_BRANCH}"
WINDOWS_URL = f"{BASE_URL}/labels/combined_windows.json"
DATASET_URL = f"{BASE_URL}/data/{NAB_DATASET}"
USER_AGENT = "ai-incident-response-system/1.0 (+dataset fetch)"

#: NAB publishes its data under https://github.com/numenta/NAB/blob/master/LICENSE
LICENSE_URL = "https://github.com/numenta/NAB/blob/master/LICENSE"


def fetch(url: str, timeout: float = 60.0) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code} fetching {url}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"cannot reach {url}: {exc.reason}") from exc


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def series_names() -> list[str]:
    """Series names of the dataset, read from the label file."""
    payload = json.loads(fetch(WINDOWS_URL).decode("utf-8"))
    prefix = f"{NAB_DATASET}/"
    return sorted(Path(key).stem for key in payload if key.startswith(prefix))


def download(force: bool = False) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    labels = fetch(WINDOWS_URL)
    LABELS_FILE.write_bytes(labels)

    manifest = {"dataset": NAB_DATASET, "source_url": DATASET_URL, "license": LICENSE_URL, "files": {}}
    for name in series_names():
        target = DATA_DIR / f"{name}.csv"
        if target.is_file() and not force:
            manifest["files"][target.name] = sha256(target.read_bytes())
            print(f"  cached  {target.name}")
            continue
        payload = fetch(f"{DATASET_URL}/{name}.csv")
        target.write_bytes(payload)
        manifest["files"][target.name] = sha256(payload)
        print(f"  fetched {target.name} ({len(payload) / 1024:.0f} KiB)")

    manifest["downloaded_at"] = ""
    MANIFEST_FILE.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return DATA_DIR


def verify() -> bool:
    if not MANIFEST_FILE.is_file():
        print("no manifest: run scripts/fetch_nab_dataset.py first", file=sys.stderr)
        return False
    manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    mismatches: list[str] = []
    for name, expected in manifest["files"].items():
        path = DATA_DIR / name
        if not path.is_file():
            mismatches.append(f"{name}: missing")
        elif sha256(path.read_bytes()) != expected:
            mismatches.append(f"{name}: checksum changed")
    for name in mismatches:
        print(f"  {name}", file=sys.stderr)
    return not mismatches


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch the NAB realAWSCloudwatch dataset")
    parser.add_argument("--force", action="store_true", help="re-download even when cached")
    parser.add_argument("--verify", action="store_true", help="only verify cached files against the manifest")
    parser.add_argument("--summary", action="store_true", help="print what is cached and exit")
    args = parser.parse_args(argv)

    if args.summary:
        for path in sorted(DATA_DIR.glob("*.csv")):
            print(f"{path.stat().st_size / 1024:8.0f} KiB  {path.name}")
        print(f"labels: {LABELS_FILE.is_file()}   manifest: {MANIFEST_FILE.is_file()}")
        return 0

    if args.verify:
        ok = verify()
        print("checksums ok" if ok else "checksum verification FAILED")
        return 0 if ok else 1

    print(f"fetching {NAB_DATASET} from {NAB_REPO}@{NAB_BRANCH} -> {DATA_DIR}")
    try:
        directory = download(force=args.force)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    total = sum(path.stat().st_size for path in directory.glob("*.csv"))
    print(f"{len(list(directory.glob('*.csv')))} series, {total / 1024:.0f} KiB total")
    print(f"manifest: {MANIFEST_FILE}")
    print(f"license:  {LICENSE_URL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
