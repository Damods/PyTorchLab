"""Fetch training data from its original source; no dataset is stored in Git."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from pathlib import Path
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
STRAWDI_URL = "https://drive.usercontent.google.com/download?id=1elFB-q9dgPbfnleA7qIrTb96Qsli8PZl&export=download&confirm=t"
STRAWDI_SIZE = 4_392_323_486
STRAWDI_SHA256 = "fc2736ffaf10959dd651a2caa25d1364b926dfc39e384f845f1402f22af9bbf8"
CHUNK = 8 * 1024**2


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024**2):
            digest.update(block)
    return digest.hexdigest()


def download_strawdi(data, workers):
    target = data / "StrawDI_Db1.zip"
    if target.exists():
        if target.stat().st_size != STRAWDI_SIZE or sha256_file(target) != STRAWDI_SHA256:
            raise RuntimeError(f"Existing archive checksum differs: {target}; inspect it before replacing it")
        print(f"Already downloaded and SHA-256 verified: {target}")
        return
    parts = data / "StrawDI_Db1.parts"
    parts.mkdir(parents=True, exist_ok=True)
    ranges = [(start, min(start + CHUNK, STRAWDI_SIZE) - 1) for start in range(0, STRAWDI_SIZE, CHUNK)]

    def fetch(bounds):
        start, end = bounds
        part = parts / f"{start:012d}.part"
        length = end - start + 1
        if part.exists() and part.stat().st_size == length:
            return length
        for attempt in range(5):
            try:
                request = urllib.request.Request(STRAWDI_URL, headers={
                    "Range": f"bytes={start}-{end}", "User-Agent": "PyTorchLab-data/1.0"})
                with urllib.request.urlopen(request, timeout=60) as response:
                    expected = f"bytes {start}-{end}/{STRAWDI_SIZE}"
                    if response.status != 206 or response.headers.get("Content-Range") != expected:
                        raise RuntimeError("Source did not return the requested range; use the official site for manual download")
                    block = response.read(length + 1)
                if len(block) != length:
                    raise RuntimeError("Incomplete download chunk")
                partial = part.with_suffix(".partial")
                partial.write_bytes(block)
                partial.replace(part)
                return length
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(min(2**attempt, 8))

    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch, item) for item in ranges]
        for future in as_completed(futures):
            completed += future.result()
            print(f"StrawDI: {completed / STRAWDI_SIZE:.1%}", flush=True)
    staging = target.with_suffix(".zip.partial")
    with staging.open("wb") as output:
        for start, _ in ranges:
            with (parts / f"{start:012d}.part").open("rb") as source:
                while block := source.read(8 * 1024**2):
                    output.write(block)
    if staging.stat().st_size != STRAWDI_SIZE or sha256_file(staging) != STRAWDI_SHA256:
        raise RuntimeError("StrawDI SHA-256 verification failed; target archive was not replaced")
    with zipfile.ZipFile(staging) as archive:
        corrupt = archive.testzip()
        if corrupt:
            raise RuntimeError(f"ZIP CRC error: {corrupt}")
    staging.replace(target)
    # Only our own numbered temporary chunks are removed after full verification.
    for start, _ in ranges:
        (parts / f"{start:012d}.part").unlink()
    print(f"Downloaded and verified: {target}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=["strawdi", "cifar10"])
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    args.data_dir.mkdir(parents=True, exist_ok=True)
    if args.dataset == "strawdi":
        print("Source and dataset terms: https://strawdi.github.io/", flush=True)
        download_strawdi(args.data_dir, args.workers)
    else:
        from torchvision.datasets import CIFAR10
        CIFAR10(str(args.data_dir), train=True, download=True)
        CIFAR10(str(args.data_dir), train=False, download=True)
        print(f"CIFAR-10 ready: {args.data_dir}")


if __name__ == "__main__":
    main()
