#!/usr/bin/env python3
"""Download a subset of the RadioML 2016.10a dataset from GitHub."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

import requests
from tqdm import tqdm

GITHUB_API_URL = "https://api.github.com/repos/dannis999/RML2016.10a/contents/2016.10a"
DEFAULT_SNRS: Sequence[int] = (-20, -10, 0, 10, 18)

def fetch_remote_listing() -> List[dict]:
    response = requests.get(GITHUB_API_URL, timeout=30)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise RuntimeError(
            "Unexpected response from GitHub API when listing dataset contents: "
            f"{payload!r}"
        )
    return payload


def parse_entry_name(name: str) -> tuple[str, int]:
    base, _, ext = name.rpartition(".")
    if ext.lower() != "txt":
        raise ValueError(f"Unexpected file extension in dataset listing: {name}")
    try:
        modulation, snr_str = base.rsplit(" ", 1)
    except ValueError as exc:  # pragma: no cover - guard for unusual filenames
        raise ValueError(f"Could not split filename into modulation/SNR parts: {name}") from exc
    try:
        snr = int(snr_str)
    except ValueError as exc:  # pragma: no cover - guard for unusual filenames
        raise ValueError(f"SNR component was not an integer in filename {name}") from exc
    return modulation, snr


def filter_entries(entries: Iterable[dict], *, snrs: Iterable[int] | None, modulations: Iterable[str] | None) -> List[dict]:
    snr_set = set(snrs) if snrs is not None else None
    modulation_set = {m.upper() for m in modulations} if modulations is not None else None

    filtered: List[dict] = []
    for entry in entries:
        name = entry.get("name")
        if not isinstance(name, str):
            continue
        modulation, snr = parse_entry_name(name)
        if snr_set is not None and snr not in snr_set:
            continue
        if modulation_set is not None and modulation.upper() not in modulation_set:
            continue
        entry = dict(entry)
        entry["modulation"] = modulation
        entry["snr"] = snr
        filtered.append(entry)
    return filtered


def download_entry(entry: dict, destination_root: Path, *, overwrite: bool = False) -> Path:
    modulation: str = entry["modulation"]
    snr: int = entry["snr"]
    download_url: str = entry["download_url"]

    output_dir = destination_root / modulation
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"snr_{snr}.txt"

    if output_path.exists() and not overwrite:
        return output_path

    with requests.get(download_url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1 << 15):
                if chunk:
                    f.write(chunk)
    return output_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("data/raw/RML2016.10a"),
        help="Directory where the dataset subset will be stored.",
    )
    parser.add_argument(
        "--snr",
        type=int,
        nargs="*",
        default=list(DEFAULT_SNRS),
        help=(
            "SNR levels to download. Defaults to a small representative subset: "
            f"{', '.join(map(str, DEFAULT_SNRS))}."
        ),
    )
    parser.add_argument(
        "--modulation",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of modulation classes to download (case-insensitive).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Redownload files even if they already exist locally.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    entries = fetch_remote_listing()
    filtered = filter_entries(entries, snrs=args.snr if args.snr else None, modulations=args.modulation)
    if not filtered:
        print("No dataset files matched the requested filters.", file=sys.stderr)
        return 1

    destination_root: Path = args.destination
    destination_root.mkdir(parents=True, exist_ok=True)

    with tqdm(filtered, desc="Downloading", unit="file") as pbar:
        for entry in pbar:
            path = download_entry(entry, destination_root, overwrite=args.overwrite)
            pbar.set_postfix({"mod": entry["modulation"], "snr": entry["snr"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
