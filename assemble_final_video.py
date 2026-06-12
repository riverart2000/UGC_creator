#!/usr/bin/env python3
"""
Assemble generated UGC clips into a single final video.

- Reads clip order and file paths from a render manifest.
- Concatenates clips in clip_index order.
- Optional --remove trims the first N opening frames from solution and
    product_hero clips (default N=2)
  before concatenation.

This script uses OpenCV to estimate per-clip FPS and ffmpeg to trim/concat while
keeping audio.

Usage:
    python assemble_final_video.py --manifest generated_clips/my-run/manifest.json
    python assemble_final_video.py --manifest generated_clips/my-run/manifest.json --remove
    python assemble_final_video.py --manifest generated_clips/my-run/manifest.json --remove --remove-frames 2
    python assemble_final_video.py --manifest generated_clips/my-run/manifest.json --remove --out generated_clips/my-run/final_removed.mp4
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

try:
    import cv2
except ImportError:  # pragma: no cover
    print("ERROR: opencv-python-headless not installed. Run: pip install -r requirements.txt", file=sys.stderr)
    sys.exit(1)


REMOVE_ROLES = {"solution", "product_hero"}


def fail(message: str, code: int = 1) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    sys.exit(code)


def run_cmd(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        stderr_tail = "\n".join((proc.stderr or "").splitlines()[-20:])
        fail(f"Command failed ({' '.join(cmd)}):\n{stderr_tail}")


def ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        fail("ffmpeg is required but not available in PATH")


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        fail(f"Manifest file not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fail(f"Manifest is not valid JSON: {exc}")


def resolve_clip_path(raw_path: str, manifest_path: Path) -> Path:
    p = Path(raw_path)
    candidates: list[Path] = []

    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append((manifest_path.parent / p).resolve())
        candidates.append((Path.cwd() / p).resolve())

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    fail(f"Clip file referenced in manifest was not found: {raw_path}")
    return Path()  # unreachable


def sorted_manifest_clips(manifest: dict[str, Any], manifest_path: Path) -> list[dict[str, Any]]:
    raw_clips = manifest.get("clips")
    if not isinstance(raw_clips, list) or not raw_clips:
        fail("Manifest has no clips array")

    clips: list[dict[str, Any]] = []
    for idx, item in enumerate(raw_clips, start=1):
        if not isinstance(item, dict):
            fail(f"Manifest clip #{idx} must be an object")
        if "clip_index" not in item or "role" not in item or "file" not in item:
            fail(f"Manifest clip #{idx} missing one of: clip_index, role, file")

        clip_index = int(item["clip_index"])
        role = str(item["role"]).strip()
        file_path = resolve_clip_path(str(item["file"]), manifest_path)
        clips.append({"clip_index": clip_index, "role": role, "file_path": file_path})

    clips.sort(key=lambda c: int(c["clip_index"]))
    return clips


def detect_fps(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        fail(f"OpenCV could not open clip: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()

    if fps <= 0:
        return 24.0
    return fps


def trim_first_n_frames(input_path: Path, output_path: Path, frames_to_remove: int) -> None:
    if frames_to_remove <= 0:
        fail("frames_to_remove must be >= 1 when trimming")

    fps = detect_fps(input_path)
    trim_seconds = float(frames_to_remove) / fps

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-ss",
        f"{trim_seconds:.6f}",
        "-vf",
        "setpts=PTS-STARTPTS",
        "-af",
        "asetpts=PTS-STARTPTS",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-x264-params",
        "bframes=0",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    run_cmd(cmd)


def write_concat_list(paths: list[Path], list_path: Path) -> None:
    lines: list[str] = []
    for p in paths:
        escaped = str(p).replace("'", "'\\''")
        lines.append(f"file '{escaped}'")
    list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def concat_clips(paths: list[Path], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="ugc_concat_") as tmp:
        list_path = Path(tmp) / "concat_list.txt"
        write_concat_list(paths, list_path)

        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_path),
            "-vf",
            "setpts=PTS-STARTPTS",
            "-af",
            "asetpts=PTS-STARTPTS",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-x264-params",
            "bframes=0",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        run_cmd(cmd)


def build_processed_paths(
    clips: list[dict[str, Any]],
    remove: bool,
    remove_frames: int,
    work_dir: Path,
) -> list[Path]:
    processed: list[Path] = []

    for clip in clips:
        clip_path = Path(clip["file_path"])
        role = str(clip["role"]).strip().lower()
        clip_index = int(clip["clip_index"])

        if remove and role in REMOVE_ROLES:
            out_name = f"clip_{clip_index:02d}_{role}_trimmed.mp4"
            out_path = work_dir / out_name
            print(f"[trim] Removing first {remove_frames} frame(s) from clip {clip_index} ({role})")
            trim_first_n_frames(clip_path, out_path, remove_frames)
            processed.append(out_path)
        else:
            processed.append(clip_path)

    return processed


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble generated clips into one final video")
    parser.add_argument("--manifest", required=True, help="Path to manifest.json from clip generation")
    parser.add_argument("--out", default=None, help="Output path for final assembled video")
    parser.add_argument(
        "--remove",
        action="store_true",
        help="Remove opening frames from solution and product_hero clips before concatenation",
    )
    parser.add_argument(
        "--remove-frames",
        type=int,
        default=2,
        help="How many opening frames to remove when --remove is set (default: 2)",
    )
    args = parser.parse_args()

    if args.remove and args.remove_frames < 1:
        fail("--remove-frames must be >= 1 when --remove is used")

    ensure_ffmpeg()

    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest = load_manifest(manifest_path)
    clips = sorted_manifest_clips(manifest, manifest_path)

    default_out = manifest_path.parent / ("final_assembled_removed.mp4" if args.remove else "final_assembled.mp4")
    output_path = Path(args.out).expanduser().resolve() if args.out else default_out

    with tempfile.TemporaryDirectory(prefix="ugc_assemble_") as tmp:
        work_dir = Path(tmp)
        input_paths = build_processed_paths(clips, args.remove, args.remove_frames, work_dir)

        print(f"[concat] Assembling {len(input_paths)} clips...")
        concat_clips(input_paths, output_path)

    print(f"Done. Final video: {output_path}")


if __name__ == "__main__":
    main()
