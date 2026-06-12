#!/usr/bin/env python3
"""
Generate video clips from a prompt-plan JSON file using Replicate.

Default behavior:
- Finds the newest JSON file in output/
- Reads video_generation settings and clips
- Builds the exact payload for xai/grok-imagine-video

Modes:
- --dry-run: print exactly what would be sent to the model (no API calls)
- live mode: call Replicate and save mp4 clips + manifest JSON

Usage:
    python3 generate_clips_from_json.py --dry-run
    python3 generate_clips_from_json.py
    python3 generate_clips_from_json.py --image-mode upload
    python3 generate_clips_from_json.py --image-mode url
    python3 generate_clips_from_json.py --image-usage skip-first
    python3 generate_clips_from_json.py --image-usage none
    python3 generate_clips_from_json.py --plan output/my-plan.json --out clips/my-plan
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from statistics import mean
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv
from PIL import Image, ImageFilter, UnidentifiedImageError

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

try:
    from pydantic import BaseModel, ConfigDict, Field, ValidationError
except ImportError:  # pragma: no cover
    print("ERROR: pydantic package not installed. Run: pip install -r requirements.txt", file=sys.stderr)
    sys.exit(1)

try:
    from tenacity import retry, retry_if_exception, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter
except ImportError:  # pragma: no cover
    print("ERROR: tenacity package not installed. Run: pip install -r requirements.txt", file=sys.stderr)
    sys.exit(1)


class PlanClipModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    clip_index: int = Field(ge=1)
    role: str = Field(min_length=1)
    duration_seconds: int = Field(ge=1, le=15)
    video_prompt: str = Field(min_length=20)
    audio_prompt: str = Field(min_length=10)
    use_image_reference: bool = True


class PlanInputDefaultsModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    aspect_ratio: str | None = None
    resolution: str | None = "480p"


class PlanVideoGenerationModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = Field(min_length=1)
    image_ref: str = Field(min_length=1)
    input_defaults: PlanInputDefaultsModel = Field(default_factory=PlanInputDefaultsModel)
    clips: list[PlanClipModel] = Field(min_length=1)


class PlanRootModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    video_generation: PlanVideoGenerationModel


PERSON_LED_REALISM_DIRECTIVE = (
    "Realism directive: documentary-style photorealism; natural skin texture and pores, "
    "accurate facial anatomy and hand/finger proportions, physically plausible body motion and eye gaze, "
    "true-to-life fabric and hair behavior, realistic exposure and shadow roll-off, handheld camera micro-movement, "
    "and subtle real-world imperfections without beauty-filter look."
)

PRODUCT_HERO_REALISM_DIRECTIVE = (
    "Realism directive: true-to-life commercial macro realism; physically accurate bottle geometry and label proportions, "
    "natural reflections/refractions and contact shadows, realistic material texture and micro-contrast, "
    "stable focus breathing and lens behavior, plausible camera motion, and authentic countertop/environment lighting "
    "with no CGI-like plastic sheen."
)

PRODUCT_ONLY_ROLES = {"product_hero", "soft_cta"}


def fail(message: str, code: int = 1) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    sys.exit(code)


@retry(
    reraise=True,
    stop=stop_after_attempt(4),
    wait=wait_exponential_jitter(initial=1, max=8),
    retry=retry_if_exception_type(requests.RequestException),
)
def http_get(url: str, **kwargs: Any) -> requests.Response:
    response = requests.get(url, **kwargs)
    response.raise_for_status()
    return response


def _is_transient_replicate_error(exc: BaseException) -> bool:
    low = str(exc).lower()
    transient_tokens = (
        "timeout",
        "tempor",
        "rate limit",
        "429",
        "500",
        "502",
        "503",
        "504",
        "service unavailable",
        "connection",
        "network",
        "try again",
    )
    return any(token in low for token in transient_tokens)


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential_jitter(initial=2, max=20),
    retry=retry_if_exception(_is_transient_replicate_error),
)
def run_replicate_request(
    client: Any,
    model: str,
    payload: dict[str, Any],
    *,
    file_encoding_strategy: str | None = None,
) -> Any:
    if file_encoding_strategy:
        return client.run(model, input=payload, file_encoding_strategy=file_encoding_strategy)
    return client.run(model, input=payload)


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential_jitter(initial=2, max=20),
    retry=retry_if_exception(_is_transient_replicate_error),
)
def run_replicate_upload_request(
    client: Any,
    model: str,
    clip: dict[str, Any],
    defaults: dict[str, Any],
    image_path: Path,
) -> Any:
    with image_path.open("rb") as image_file:
        payload = build_model_input(clip, image_file, defaults)
        return client.run(model, input=payload, file_encoding_strategy="base64")


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return slug or "clip"


def resolve_plan_path(plan: str | None, plans_dir: str) -> Path:
    if plan:
        p = Path(plan).expanduser().resolve()
        if not p.is_file():
            fail(f"Plan file not found: {p}")
        return p

    base = Path(plans_dir).expanduser().resolve()
    if not base.is_dir():
        fail(f"Plans directory not found: {base}")

    candidates = sorted(base.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
    if not candidates:
        fail(f"No JSON plan files found in: {base}")
    return candidates[0]


def load_plan(plan_path: Path) -> dict[str, Any]:
    try:
        return json.loads(plan_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fail(f"Invalid JSON in plan file {plan_path}: {exc}")


def parse_clip_indexes(raw: str | None) -> set[int] | None:
    if raw is None or not raw.strip():
        return None

    indexes: set[int] = set()
    for token in raw.split(","):
        part = token.strip()
        if not part:
            continue
        if not part.isdigit():
            fail(f"Invalid --clips value {part!r}. Use comma-separated integers, e.g. 1,2")
        value = int(part)
        if value <= 0:
            fail(f"Invalid clip index in --clips: {value}. Indexes must be >= 1")
        indexes.add(value)

    if not indexes:
        fail("--clips provided but no valid clip indexes were found")
    return indexes


def parse_aspect_ratio(aspect_ratio: str | None) -> tuple[int, int] | None:
    if not isinstance(aspect_ratio, str) or not aspect_ratio.strip():
        return None

    match = re.match(r"^\s*(\d+)\s*[:/xX]\s*(\d+)\s*$", aspect_ratio)
    if not match:
        return None

    width = int(match.group(1))
    height = int(match.group(2))
    if width <= 0 or height <= 0:
        return None

    return width, height


def _estimate_focus_center(image: Image.Image) -> tuple[float, float]:
    """Estimate focal point using edge energy with a center bias."""
    resample = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
    work = image.convert("L")
    w, h = work.size
    max_side = 512
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        work = work.resize((max(1, int(round(w * scale))), max(1, int(round(h * scale)))), resample=resample)

    ew, eh = work.size
    edges = work.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.GaussianBlur(radius=2))
    pixels = edges.load()

    cx = (ew - 1) / 2 if ew > 1 else 0.0
    cy = (eh - 1) / 2 if eh > 1 else 0.0

    x_energy = [0.0] * ew
    y_energy = [0.0] * eh
    total = 0.0
    for y in range(eh):
        row_sum = 0.0
        for x in range(ew):
            val = float(pixels[x, y])
            x_bias = 1.0 - 0.35 * (abs(x - cx) / max(cx, 1.0))
            y_bias = 1.0 - 0.35 * (abs(y - cy) / max(cy, 1.0))
            weighted = max(0.0, val * x_bias * y_bias)
            x_energy[x] += weighted
            row_sum += weighted
        y_energy[y] = row_sum
        total += row_sum

    if total <= 1e-6:
        return 0.5, 0.5

    x_total = sum(x_energy)
    x_center = sum(idx * val for idx, val in enumerate(x_energy)) / max(x_total, 1e-6)
    y_center = sum(idx * val for idx, val in enumerate(y_energy)) / total

    x_norm = x_center / max(ew - 1, 1)
    y_norm = y_center / max(eh - 1, 1)
    return x_norm, y_norm


def fit_image_to_aspect(image: Image.Image, aspect_ratio: tuple[int, int]) -> Image.Image:
    """Fit image to target aspect ratio using focus-aware cover crop (no blur/padding)."""
    src = image.convert("RGB")
    src_w, src_h = src.size
    target_w, target_h = aspect_ratio
    target_ratio = target_w / target_h
    src_ratio = src_w / src_h

    # Already close enough to target ratio.
    if abs(src_ratio - target_ratio) <= 1e-4:
        return src

    resample = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
    focus_x, focus_y = _estimate_focus_center(src)

    if src_ratio > target_ratio:
        crop_h = src_h
        crop_w = max(1, int(round(crop_h * target_ratio)))
    else:
        crop_w = src_w
        crop_h = max(1, int(round(crop_w / target_ratio)))

    focus_px = int(round(focus_x * max(src_w - 1, 1)))
    focus_py = int(round(focus_y * max(src_h - 1, 1)))

    left = max(0, min(focus_px - crop_w // 2, src_w - crop_w))
    top = max(0, min(focus_py - crop_h // 2, src_h - crop_h))
    cropped = src.crop((left, top, left + crop_w, top + crop_h))

    long_side = max(cropped.width, cropped.height)
    if long_side > 1920:
        scale = 1920 / long_side
        new_w = max(1, int(round(cropped.width * scale)))
        new_h = max(1, int(round(cropped.height * scale)))
        cropped = cropped.resize((new_w, new_h), resample=resample)

    return cropped


def prepare_image_asset(image_ref: str, cache_dir: Path, aspect_ratio: str | None = None) -> Path:
    """Resolve image_ref into a normalized local JPEG path for upload mode."""
    local_candidate = Path(image_ref).expanduser()
    source_path: Path

    if local_candidate.is_file():
        source_path = local_candidate.resolve()
    else:
        if not image_ref.startswith(("http://", "https://")):
            fail(
                "image_ref must be a local file path or an http(s) URL when using --image-mode upload. "
                f"Got: {image_ref}"
            )

        cache_dir.mkdir(parents=True, exist_ok=True)
        parsed = urlparse(image_ref)
        ext = Path(parsed.path).suffix.lower()
        if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
            ext = ".jpg"

        ref_hash = hashlib.sha1(image_ref.encode("utf-8")).hexdigest()[:12]
        source_path = cache_dir / f"source_image_raw_{ref_hash}{ext}"
        if not (source_path.exists() and source_path.stat().st_size > 0):
            try:
                resp = http_get(image_ref, timeout=120)
                source_path.write_bytes(resp.content)
            except requests.RequestException as exc:
                fail(f"Failed to download image_ref for upload mode: {exc}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    normalized_path = cache_dir / "source_image.jpg"
    parsed_aspect = parse_aspect_ratio(aspect_ratio)
    try:
        with Image.open(source_path) as img:
            prepared = fit_image_to_aspect(img, parsed_aspect) if parsed_aspect else img.convert("RGB")
            prepared.save(
                normalized_path,
                format="JPEG",
                quality=95,
                optimize=False,
                progressive=False,
            )
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        fail(f"Unsupported or unreadable image_ref for upload mode: {exc}")

    return normalized_path


def validate_plan(plan: dict[str, Any]) -> tuple[str, str, dict[str, Any], list[dict[str, Any]]]:
    try:
        validated = PlanRootModel.model_validate(plan)
    except ValidationError as exc:
        fail(f"Plan schema validation failed: {exc}")

    vg = validated.video_generation
    seen_indexes: set[int] = set()
    for clip in vg.clips:
        if clip.clip_index in seen_indexes:
            fail(f"Duplicate clip_index found in plan: {clip.clip_index}")
        seen_indexes.add(clip.clip_index)

    defaults = vg.input_defaults.model_dump(mode="python", exclude_none=True)
    clips = [clip.model_dump(mode="python") for clip in vg.clips]
    return vg.model, vg.image_ref, defaults, clips


def build_final_prompt(video_prompt: str, audio_prompt: str) -> str:
    audio = audio_prompt.strip()
    if audio.upper().startswith("AUDIO:"):
        return f"{video_prompt.strip()}\n{audio}"
    return f"{video_prompt.strip()}\nAUDIO: {audio}"


def role_realism_directive(role: str) -> str:
    role_lower = role.strip().lower()
    if role_lower in PRODUCT_ONLY_ROLES:
        return PRODUCT_HERO_REALISM_DIRECTIVE
    return PERSON_LED_REALISM_DIRECTIVE


def append_realism_directive(video_prompt: str, role: str) -> str:
    prompt = video_prompt.strip()
    if not prompt:
        return role_realism_directive(role)

    # Respect prompts that already include a custom realism section.
    if "realism directive:" in prompt.lower():
        return prompt

    return f"{prompt} {role_realism_directive(role)}"


def build_model_input(
    clip: dict[str, Any],
    image_value: Any,
    defaults: dict[str, Any],
) -> dict[str, Any]:
    enhanced_video_prompt = append_realism_directive(
        str(clip["video_prompt"]),
        str(clip.get("role", "")),
    )

    payload: dict[str, Any] = {
        "prompt": build_final_prompt(enhanced_video_prompt, str(clip["audio_prompt"])),
        "duration": int(clip["duration_seconds"]),
    }

    if image_value is not None:
        payload["image"] = image_value

    aspect = defaults.get("aspect_ratio")
    if isinstance(aspect, str) and aspect.strip():
        payload["aspect_ratio"] = aspect

    resolution = defaults.get("resolution", "480p")
    if isinstance(resolution, str) and resolution.strip():
        payload["resolution"] = resolution
    else:
        payload["resolution"] = "480p"

    return payload


def should_use_image_for_clip(image_usage: str, clip_index: int) -> bool:
    if image_usage == "all":
        return True
    if image_usage == "none":
        return False
    if image_usage == "skip-first":
        return clip_index != 1
    fail(f"Unsupported image_usage: {image_usage}")


def extract_output_url(output: Any) -> str | None:
    if isinstance(output, str) and output.startswith("http"):
        return output

    if isinstance(output, list) and output:
        return extract_output_url(output[0])

    if isinstance(output, dict):
        for key in ("url", "output", "file", "video"):
            value = output.get(key)
            if isinstance(value, str) and value.startswith("http"):
                return value

    if hasattr(output, "url"):
        try:
            maybe_url = output.url() if callable(output.url) else output.url
            if isinstance(maybe_url, str) and maybe_url.startswith("http"):
                return maybe_url
        except Exception:
            pass

    return None


def write_video_output(output: Any, out_path: Path) -> None:
    if isinstance(output, list) and output:
        output = output[0]

    if hasattr(output, "read"):
        try:
            data = output.read()
            if isinstance(data, (bytes, bytearray)):
                out_path.write_bytes(data)
                return
            if isinstance(data, str) and data.startswith("http"):
                download_url_to_file(data, out_path)
                return
        except Exception:
            pass

    if isinstance(output, (bytes, bytearray)):
        out_path.write_bytes(output)
        return

    url = extract_output_url(output)
    if url:
        download_url_to_file(url, out_path)
        return

    fail(f"Unsupported output type from Replicate for {out_path.name}: {type(output).__name__}")


def download_url_to_file(url: str, out_path: Path) -> None:
    resp = http_get(url, timeout=120)
    out_path.write_bytes(resp.content)


def evaluate_clip_quality(video_path: Path, role: str) -> dict[str, Any]:
    report: dict[str, Any] = {
        "tool": "opencv",
        "status": "ok",
        "passed": True,
        "issues": [],
        "metrics": {},
    }

    if cv2 is None:
        report["status"] = "skipped"
        report["issues"].append("opencv-python-headless not installed")
        return report

    if not video_path.exists() or video_path.stat().st_size <= 0:
        report["passed"] = False
        report["issues"].append("output video file missing or empty")
        return report

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        report["passed"] = False
        report["issues"].append("opencv could not open generated video")
        return report

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    duration = (frame_count / fps) if fps > 0 else 0.0
    sample_step = max(1, frame_count // 40) if frame_count > 0 else 1

    blur_values: list[float] = []
    frame_diff_values: list[float] = []
    previous_gray = None
    idx = 0
    sampled_frames = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if idx % sample_step != 0:
            idx += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur_values.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        if previous_gray is not None:
            frame_diff_values.append(float(cv2.absdiff(gray, previous_gray).mean()))
        previous_gray = gray
        sampled_frames += 1
        idx += 1

    cap.release()

    avg_blur = float(mean(blur_values)) if blur_values else 0.0
    avg_motion = float(mean(frame_diff_values)) if frame_diff_values else 0.0
    freeze_ratio = (
        sum(1 for d in frame_diff_values if d < 1.4) / len(frame_diff_values)
        if frame_diff_values
        else 1.0
    )

    report["metrics"] = {
        "file_size_bytes": int(video_path.stat().st_size),
        "frame_count": frame_count,
        "sampled_frames": sampled_frames,
        "fps": round(fps, 3),
        "duration_seconds": round(duration, 3),
        "avg_laplacian_variance": round(avg_blur, 3),
        "avg_frame_diff": round(avg_motion, 3),
        "freeze_ratio": round(float(freeze_ratio), 3),
    }

    role_lower = role.lower()
    is_product_only = "product_hero" in role_lower or "soft_cta" in role_lower
    min_blur = 18.0
    max_freeze_ratio = 0.9 if is_product_only else 0.75
    min_motion = 0.15 if is_product_only else 0.6

    if sampled_frames < 6:
        report["passed"] = False
        report["issues"].append("too few decodable frames")
    if avg_blur < min_blur:
        report["passed"] = False
        report["issues"].append("video appears too blurry")
    if freeze_ratio > max_freeze_ratio:
        report["passed"] = False
        report["issues"].append("video appears too static/frozen")
    if avg_motion < min_motion:
        report["passed"] = False
        report["issues"].append("insufficient motion for role")

    return report


def run_dry(
    model: str,
    image_ref: str,
    defaults: dict[str, Any],
    clips: list[dict[str, Any]],
    image_mode: str,
    image_usage: str,
    image_cache_dir: Path,
    selected_clip_indexes: set[int] | None,
) -> None:
    if image_mode not in {"upload", "url"}:
        fail(f"Unsupported image_mode: {image_mode}")
    if image_usage not in {"all", "none", "skip-first"}:
        fail(f"Unsupported image_usage: {image_usage}")

    image_for_payload: Any = None
    if image_mode == "upload" and image_usage != "none":
        image_path = prepare_image_asset(image_ref, image_cache_dir, defaults.get("aspect_ratio"))
        image_for_payload = f"<uploaded_file:{image_path}>"
        image_mode_note = f"upload (source: {image_path})"
    elif image_mode == "upload":
        image_mode_note = "upload (image disabled by image-usage)"
    elif image_usage != "none":
        image_for_payload = image_ref
        image_mode_note = "url"
    else:
        image_mode_note = "url (image disabled by image-usage)"

    print("DRY RUN: no API calls will be made.")
    print(f"Model: {model}")
    print(f"Image mode: {image_mode_note}")
    print(f"Image usage: {image_usage}")
    print(f"Resolved image_ref: {image_ref}")
    print(f"Image ref: {image_ref}")
    print("")

    clips_to_render = (
        [c for c in clips if int(c["clip_index"]) in selected_clip_indexes]
        if selected_clip_indexes is not None
        else clips
    )
    if not clips_to_render:
        fail("No clips matched --clips selection")

    for clip in clips_to_render:
        clip_id = int(clip["clip_index"])
        role = str(clip["role"])
        clip_prefers_image = bool(clip.get("use_image_reference", True))
        use_image = should_use_image_for_clip(image_usage, clip_id) and clip_prefers_image
        payload = build_model_input(clip, image_for_payload if use_image else None, defaults)
        print(f"--- clip_{clip_id:02d} ({role}) ---")
        print(json.dumps({"model": model, "input": payload}, indent=2, ensure_ascii=False))
        print("")


def run_live(
    model: str,
    image_ref: str,
    defaults: dict[str, Any],
    clips: list[dict[str, Any]],
    out_dir: Path,
    image_mode: str,
    image_usage: str,
    image_cache_dir: Path,
    selected_clip_indexes: set[int] | None,
    qc_enabled: bool,
    strict_qc: bool,
) -> None:
    token = os.getenv("REPLICATE_API_TOKEN")
    if not token:
        fail("REPLICATE_API_TOKEN not set. Add it to .env before live generation.")

    try:
        import replicate
    except ImportError:
        fail("replicate package not installed. Run: pip install -r requirements.txt")

    out_dir.mkdir(parents=True, exist_ok=True)
    client = replicate.Client(api_token=token)

    if image_mode not in {"upload", "url"}:
        fail(f"Unsupported image_mode: {image_mode}")
    if image_usage not in {"all", "none", "skip-first"}:
        fail(f"Unsupported image_usage: {image_usage}")

    uploaded_image_path: Path | None = None
    if image_mode == "upload" and image_usage != "none":
        uploaded_image_path = prepare_image_asset(image_ref, image_cache_dir, defaults.get("aspect_ratio"))
        print(f"Image mode: upload (using local file {uploaded_image_path})")
    elif image_mode == "upload":
        print("Image mode: upload (image disabled by image-usage)")
    else:
        print("Image mode: url")
    print(f"Image usage: {image_usage}")
    print(f"Resolved image_ref: {image_ref}")

    manifest: dict[str, Any] = {
        "model": model,
        "image_ref": image_ref,
        "image_mode": image_mode,
        "image_usage": image_usage,
        "qc_enabled": qc_enabled,
        "strict_qc": strict_qc,
        "uploaded_image_path": str(uploaded_image_path) if uploaded_image_path else None,
        "output_dir": str(out_dir),
        "clips": [],
    }

    clips_to_render = (
        [c for c in clips if int(c["clip_index"]) in selected_clip_indexes]
        if selected_clip_indexes is not None
        else clips
    )
    if not clips_to_render:
        fail("No clips matched --clips selection")

    qc_failures: list[str] = []

    total = len(clips_to_render)
    for idx, clip in enumerate(clips_to_render, start=1):
        clip_index = int(clip["clip_index"])
        role = str(clip["role"])
        role_slug = slugify(role)
        filename = f"clip_{clip_index:02d}_{role_slug}.mp4"
        out_path = out_dir / filename
        clip_prefers_image = bool(clip.get("use_image_reference", True))
        use_image = should_use_image_for_clip(image_usage, clip_index) and clip_prefers_image

        if use_image and image_mode == "upload":
            assert uploaded_image_path is not None
            print(f"[{idx}/{total}] Generating {filename}...")
            output = run_replicate_upload_request(client, model, clip, defaults, uploaded_image_path)
            write_video_output(output, out_path)
        elif use_image:
            payload = build_model_input(clip, image_ref, defaults)

            print(f"[{idx}/{total}] Generating {filename}...")
            output = run_replicate_request(client, model, payload)
            write_video_output(output, out_path)
        else:
            payload = build_model_input(clip, None, defaults)
            print(f"[{idx}/{total}] Generating {filename}... (no image)")
            output = run_replicate_request(client, model, payload)
            write_video_output(output, out_path)

        if not use_image:
            manifest_image_value = None
        elif image_mode == "url":
            manifest_image_value = image_ref
        else:
            manifest_image_value = f"uploaded_file:{uploaded_image_path}"
        manifest_payload = build_model_input(clip, manifest_image_value, defaults)

        manifest["clips"].append(
            {
                "clip_index": clip_index,
                "role": role,
                "use_image_reference": use_image,
                "file": str(out_path),
                "input": manifest_payload,
            }
        )

        if qc_enabled:
            qc_report = evaluate_clip_quality(out_path, role)
            manifest["clips"][-1]["quality_check"] = qc_report
            qc_status = "PASS" if qc_report.get("passed") else "FAIL"
            print(f"      QC: {qc_status}")
            if qc_report.get("issues"):
                print(f"      QC issues: {', '.join(str(x) for x in qc_report['issues'])}")
            if not bool(qc_report.get("passed")):
                qc_failures.append(filename)

        print(f"      Saved: {out_path}")

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Done. Manifest: {manifest_path}")

    if qc_enabled and strict_qc and qc_failures:
        fail(
            "QC strict mode failed for: "
            + ", ".join(qc_failures)
            + ". No auto-regeneration was performed. Re-run selected clips manually using --clips."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate clips from prompt-plan JSON via Replicate")
    parser.add_argument("--plan", default=None, help="Path to a specific JSON plan file")
    parser.add_argument("--plans-dir", default="output", help="Directory containing JSON plan files")
    parser.add_argument("--out", default=None, help="Output directory for generated clips")
    parser.add_argument(
        "--image-mode",
        choices=["upload", "url"],
        default="upload",
        help="How to send image input to Replicate. Default 'upload' avoids remote SSL URL fetch issues.",
    )
    parser.add_argument(
        "--image-usage",
        choices=["all", "none", "skip-first"],
        default="all",
        help="Control which clips receive image input: all, none, or skip-first.",
    )
    parser.add_argument(
        "--clips",
        default=None,
        help="Optional comma-separated clip indexes to render (e.g. 1,2).",
    )
    parser.add_argument(
        "--qc",
        action="store_true",
        help="Run OpenCV quality checks and include results in manifest (no auto-regeneration).",
    )
    parser.add_argument(
        "--strict-qc",
        action="store_true",
        help="With --qc, exit non-zero when any clip fails quality checks. Never auto-regenerates clips.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print exact payloads and exit")
    args = parser.parse_args()

    load_dotenv()

    plan_path = resolve_plan_path(args.plan, args.plans_dir)
    plan = load_plan(plan_path)
    model, image_ref, defaults, clips = validate_plan(plan)
    selected_clip_indexes = parse_clip_indexes(args.clips)

    plan_stem = plan_path.stem
    out_dir = Path(args.out).expanduser().resolve() if args.out else Path("generated_clips") / plan_stem
    image_cache_dir = plan_path.parent / "_assets"

    print(f"Plan file: {plan_path}")

    if args.dry_run:
        run_dry(
            model,
            image_ref,
            defaults,
            clips,
            args.image_mode,
            args.image_usage,
            image_cache_dir,
            selected_clip_indexes,
        )
        return

    run_live(
        model,
        image_ref,
        defaults,
        clips,
        out_dir,
        args.image_mode,
        args.image_usage,
        image_cache_dir,
        selected_clip_indexes,
        args.qc,
        args.strict_qc,
    )


if __name__ == "__main__":
    main()
