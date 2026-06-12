#!/usr/bin/env python3
"""
generate_ugc_script.py

Generate a detailed UGC ad JSON PROMPT SCRIPT from a product blog URL + a config JSON.

This tool ONLY authors the script file. It does NOT call Replicate and does NOT
generate any clips or audio. A separate downstream script consumes the output
`storyboard.json` to actually create:
  - video clips via Replicate `xai/grok-imagine-video` (image-to-video)
  - voiceover via Replicate `inworld/realtime-tts-2`

Output:
    - output/<url-slug>.json : machine-readable, downstream-consumable prompt plan

Only OPENAI_API_KEY (from .env) is required.

Usage:
    python generate_ugc_script.py --url <blog_url> --config config.json
    python generate_ugc_script.py --url <blog_url> --config config.json --out output
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import re
import sys
import textwrap
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote, urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from PIL import Image

try:
    from pydantic import BaseModel, ConfigDict, Field, ValidationError
except ImportError:  # pragma: no cover
    print("ERROR: pydantic package not installed. Run: pip install -r requirements.txt", file=sys.stderr)
    sys.exit(1)

try:
    from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter
except ImportError:  # pragma: no cover
    print("ERROR: tenacity package not installed. Run: pip install -r requirements.txt", file=sys.stderr)
    sys.exit(1)

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    print("ERROR: openai package not installed. Run: pip install -r requirements.txt", file=sys.stderr)
    sys.exit(1)

OPENAI_RETRY_EXCEPTIONS: tuple[type[BaseException], ...] = (Exception,)
try:
    from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError

    OPENAI_RETRY_EXCEPTIONS = (APIConnectionError, APITimeoutError, InternalServerError, RateLimitError)
except Exception:
    # Older SDKs may not expose all typed transient exceptions.
    OPENAI_RETRY_EXCEPTIONS = (Exception,)


# --------------------------------------------------------------------------- #
# Model reference metadata (embedded into storyboard.json so the downstream
# script / Copilot can build the generator with zero extra context).
# --------------------------------------------------------------------------- #

MODELS_REFERENCE: dict[str, Any] = {
    "video": {
        "replicate_model": "xai/grok-imagine-video",
        "mode": "image-to-video",
        "pricing": "$0.05 per second of output video",
        "inputs": {
            "prompt": "Text prompt. Motion-focused for i2v: Subject + Action + Setting + Camera + Lighting/Mood.",
            "image": "Input image to animate (image-to-video). The downstream script must supply this (see clip.image_ref).",
            "duration": "Integer seconds 1-15. We use 5-8 for stability.",
            "aspect_ratio": "16:9 | 9:16 | 1:1 | 4:3 | 3:4 | 3:2 | 2:3.",
            "resolution": "480p | 720p.",
        },
        "notes": [
            "Generates its own synchronized audio; downstream should MUTE/ignore it (Inworld voiceover is the narration).",
            "i2v prompts: describe MOTION only, do not re-describe the image, never contradict it, no negative prompts.",
            "Keep to one subject + one action + one camera move; use specific verbs + intensity modifiers.",
            "Reuse the SAME product image across clips for visual consistency.",
        ],
    },
    "voiceover": {
        "replicate_model": "inworld/realtime-tts-2",
        "pricing": "$0.035 per 1,000 input characters",
        "inputs": {
            "text": "Text to speak. Max 2,000 characters. Supports bracketed steering tags placed BEFORE the text they apply to.",
            "voice_id": "Preset voice (Ashley, Dennis, Alex, Darlene) or a custom cloned voice ID.",
            "language": "Language code (e.g. en) or 'auto'.",
            "speaking_rate": "0-1.5; 0 = normal (1.0).",
            "temperature": "0-2; 0 = model default (1.1). Higher = more expressive.",
            "audio_format": "Output format, e.g. mp3.",
            "sample_rate": "Hz, e.g. 48000.",
            "text_normalization": "auto | on | off.",
        },
        "notes": [
            "Steering: wrap natural-language direction in [brackets] BEFORE the relevant text, e.g. [calm, trustworthy expert tone].",
            "Inline non-verbals like [breathe] can be placed where the sound should occur.",
            "Use CAPS to emphasize specific benefit words.",
            "Match the steering instruction to the text; avoid conflicting tags.",
        ],
    },
}


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #


@dataclass
class BlogContent:
    url: str
    title: str
    text: str
    image_candidates: list[str]


@dataclass
class SelectedImage:
    url: str
    width: int
    height: int
    reason: str


@dataclass
class ClipSpec:
    index: int
    role: str
    role_label: str
    min_seconds: int
    max_seconds: int
    purpose: str


class StoryboardSourceModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    blog_url: str = Field(min_length=1)
    blog_title: str = Field(min_length=1)


class StoryboardAdModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    platform: str = Field(min_length=1)
    aspect_ratio: str = Field(min_length=3)
    resolution: str = Field(min_length=3)
    target_seconds: int = Field(ge=1)
    total_clip_seconds: int = Field(ge=1)
    style: str = Field(default="")


class StoryboardIdealCustomerProfileModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    person_name: str = Field(min_length=1, max_length=80)
    age: int = Field(ge=18, le=85)
    career: str = Field(min_length=1)
    hobbies: list[str] = Field(min_length=2, max_length=6)
    life_context: str = Field(min_length=1)
    purchase_motivation: str = Field(min_length=1)


class StoryboardStoryModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ideal_customer_profile: StoryboardIdealCustomerProfileModel
    primary_problem: str = Field(min_length=1)
    resolution_path: str = Field(min_length=1)
    concept: str = Field(min_length=1)
    call_to_action: str = Field(min_length=1)


class StoryboardClipModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    clip_index: int = Field(ge=1)
    role: str = Field(min_length=1)
    duration_seconds: int = Field(ge=1, le=15)
    use_image_reference: bool
    video_prompt: str = Field(min_length=20)
    audio_prompt: str = Field(min_length=10)


class StoryboardInputDefaultsModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    aspect_ratio: str = Field(min_length=3)
    resolution: str = Field(min_length=3)


class StoryboardVideoGenerationModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1)
    image_ref: str = Field(min_length=1)
    input_defaults: StoryboardInputDefaultsModel
    prompt_assembly: str = Field(min_length=1)
    clips: list[StoryboardClipModel] = Field(min_length=1)


class StoryboardModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(min_length=1)
    generated_by: str = Field(min_length=1)
    source: StoryboardSourceModel
    structure_source: str = Field(min_length=1)
    ad: StoryboardAdModel
    brand: dict[str, Any]
    story: StoryboardStoryModel
    video_generation: StoryboardVideoGenerationModel


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


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


@retry(
    reraise=True,
    stop=stop_after_attempt(4),
    wait=wait_exponential_jitter(initial=1, max=8),
    retry=retry_if_exception_type(requests.RequestException),
)
def http_post(url: str, **kwargs: Any) -> requests.Response:
    response = requests.post(url, **kwargs)
    response.raise_for_status()
    return response


def load_config(path: str) -> dict[str, Any]:
    if not os.path.isfile(path):
        fail(f"Config file not found: {path}")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError as exc:
        fail(f"Config file is not valid JSON: {exc}")
    return {}  # unreachable


def require(d: dict[str, Any], *keys: str) -> Any:
    cur: Any = d
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            fail(f"Missing required config key: {'.'.join(keys)}")
        cur = cur[key]
    return cur


def slug_from_url(url: str) -> str:
    parsed = urlparse(url)
    slug = parsed.path.rstrip("/").split("/")[-1]
    if not slug:
        slug = parsed.netloc or "ugc-script"
    slug = re.sub(r"\.html?$", "", slug, flags=re.IGNORECASE)
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", slug).strip("-").lower()
    return slug or "ugc-script"


def next_available_json_path(out_dir: str, base_slug: str) -> str:
    candidate = os.path.join(out_dir, f"{base_slug}.json")
    if not os.path.exists(candidate):
        return candidate
    suffix = 2
    while True:
        candidate = os.path.join(out_dir, f"{base_slug}-{suffix}.json")
        if not os.path.exists(candidate):
            return candidate
        suffix += 1


def canonical_role(role_label: str) -> str:
    lower = role_label.lower()
    if "story" in lower and "arc" in lower:
        return "story_arc"
    if "hero" in lower and "product" in lower:
        return "product_hero"
    if "hook" in lower:
        return "hook"
    if "problem" in lower:
        return "problem"
    if "solution" in lower or "resolution" in lower:
        return "solution"
    if "education" in lower:
        return "education"
    if "product" in lower:
        return "product"
    if "soft" in lower and "cta" in lower:
        return "soft_cta"
    if "cta" in lower:
        return "soft_cta"
    return re.sub(r"[^a-z0-9]+", "_", lower).strip("_")


def load_clip_structure(structure_path: str) -> list[ClipSpec]:
    if not os.path.isfile(structure_path):
        fail(f"Required structure file not found: {structure_path}")

    pattern = re.compile(
        r"^\s*clip\s*(\d+)\s*:\s*(\d+)\s*-\s*(\d+)\s*sec(?:s)?\s*[—-]\s*(.+?)\s*$",
        re.IGNORECASE,
    )
    specs: list[ClipSpec] = []
    with open(structure_path, "r", encoding="utf-8") as fh:
        for line in fh:
            m = pattern.match(line.strip())
            if not m:
                continue
            idx = int(m.group(1))
            min_s = int(m.group(2))
            max_s = int(m.group(3))
            label = m.group(4).strip()
            role = canonical_role(label)
            specs.append(
                ClipSpec(
                    index=idx,
                    role=role,
                    role_label=label,
                    min_seconds=min_s,
                    max_seconds=max_s,
                    purpose="",
                )
            )

    if not specs:
        fail(f"No clip definitions found in {structure_path}")

    specs.sort(key=lambda s: s.index)

    expected_indexes = list(range(1, len(specs) + 1))
    actual_indexes = [s.index for s in specs]
    if actual_indexes != expected_indexes:
        fail(
            "structure.md clip indexes must be contiguous and start at 1. "
            f"Got: {actual_indexes}"
        )

    if len(specs) < 2:
        fail("structure.md must define at least 2 clips")

    for s in specs:
        if s.min_seconds > s.max_seconds:
            fail(f"Invalid duration range for clip {s.index}: min > max")
        if s.max_seconds > 15:
            fail(
                f"Invalid duration range for clip {s.index}: max {s.max_seconds}s exceeds model limit of 15s"
            )

    return specs


# --------------------------------------------------------------------------- #
# Step 2: Blog extraction
# --------------------------------------------------------------------------- #

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/123.0 Safari/537.36"
)
SHOPIFY_API_VERSION = os.getenv("SHOPIFY_API_VERSION", "2025-01")


def _extract_blog_handles(blog_url: str) -> tuple[str, str] | None:
    parsed = urlparse(blog_url)
    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) < 3 or parts[0].lower() != "blogs":
        return None
    return parts[1], parts[2]


def _normalize_shop_domain(domain: str) -> str:
    return domain.strip().replace("https://", "").replace("http://", "").strip("/")


def _ensure_shopify_env_loaded() -> None:
    has_domain = bool(os.getenv("MYSHOPIFY_DOMAIN") or os.getenv("SHOPIFY_MYSHOPIFY_DOMAIN"))
    has_token = bool(os.getenv("SHOPIFY_ADMIN_ACCESS_TOKEN") or os.getenv("SHOPIFY_ACCESS_TOKEN"))
    has_client = bool(os.getenv("SHOPIFY_CLIENT_ID") or os.getenv("SHOPIFY_API_KEY"))
    has_secret = bool(
        os.getenv("SHOPIFY_CLIENT_SECRET") or os.getenv("SHOPIFY_SECRET") or os.getenv("SHOPIFY_API_SECRET")
    )
    if has_domain and (has_token or (has_client and has_secret)):
        return

    fallback_env_paths = [
        os.getenv("SHOPIFY_ENV_PATH", "").strip(),
        "/Users/joebains/shopify-ai-blog-system/.env",
    ]
    for env_path in fallback_env_paths:
        if env_path and os.path.isfile(env_path):
            load_dotenv(dotenv_path=env_path, override=False)
            break


def _shopify_get_admin_token(myshopify_domain: str) -> str | None:
    static_token = os.getenv("SHOPIFY_ADMIN_ACCESS_TOKEN") or os.getenv("SHOPIFY_ACCESS_TOKEN")
    if static_token:
        return static_token.strip()

    client_id = os.getenv("SHOPIFY_CLIENT_ID") or os.getenv("SHOPIFY_API_KEY")
    client_secret = (
        os.getenv("SHOPIFY_CLIENT_SECRET")
        or os.getenv("SHOPIFY_SECRET")
        or os.getenv("SHOPIFY_API_SECRET")
    )
    if not client_id or not client_secret:
        return None

    token_url = f"https://{myshopify_domain}/admin/oauth/access_token"
    payload = {
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret,
    }
    try:
        resp = http_post(
            token_url,
            headers={"Content-Type": "application/x-www-form-urlencoded", "User-Agent": USER_AGENT},
            data=payload,
            timeout=30,
        )
        token = resp.json().get("access_token")
        if isinstance(token, str) and token.strip():
            return token.strip()
    except requests.RequestException as exc:
        print(f"WARN: Shopify Admin token fetch failed; falling back to page image scraping ({exc})", file=sys.stderr)
    except ValueError:
        print("WARN: Shopify Admin token response was not valid JSON; falling back to page image scraping", file=sys.stderr)
    return None


def fetch_shopify_admin_article_image(blog_url: str) -> str | None:
    _ensure_shopify_env_loaded()

    handles = _extract_blog_handles(blog_url)
    if handles is None:
        return None

    blog_handle, article_handle = handles
    myshopify_domain = _normalize_shop_domain(
        os.getenv("MYSHOPIFY_DOMAIN") or os.getenv("SHOPIFY_MYSHOPIFY_DOMAIN") or ""
    )
    if not myshopify_domain:
        return None

    token = _shopify_get_admin_token(myshopify_domain)
    if not token:
        return None

    headers = {
        "X-Shopify-Access-Token": token,
        "Content-Type": "application/json",
        "User-Agent": USER_AGENT,
    }
    base_url = f"https://{myshopify_domain}/admin/api/{SHOPIFY_API_VERSION}"

    try:
        blogs_resp = http_get(f"{base_url}/blogs.json?fields=id,handle&limit=250", headers=headers, timeout=30)
        blogs = blogs_resp.json().get("blogs", [])
        blog_id = next((b.get("id") for b in blogs if b.get("handle") == blog_handle), None)
        if not blog_id:
            return None

        # Try direct handle query first, then fallback to scanning the blog article list.
        article_endpoint = (
            f"{base_url}/blogs/{blog_id}/articles.json"
            f"?limit=1&fields=id,handle,image&handle={quote(article_handle)}"
        )
        article_resp = http_get(article_endpoint, headers=headers, timeout=30)
        articles = article_resp.json().get("articles", [])

        if not articles:
            list_endpoint = f"{base_url}/blogs/{blog_id}/articles.json?limit=250&fields=id,handle,image"
            list_resp = http_get(list_endpoint, headers=headers, timeout=30)
            articles = list_resp.json().get("articles", [])

        match = next((a for a in articles if a.get("handle") == article_handle), None)
        if not match:
            return None

        image_obj = match.get("image") or {}
        src = image_obj.get("src")
        if isinstance(src, str) and src.strip():
            return _normalize_shopify_url(src.strip())
    except requests.RequestException as exc:
        print(f"WARN: Shopify Admin article image fetch failed; falling back to page scraping ({exc})", file=sys.stderr)
    except ValueError:
        print("WARN: Shopify Admin article response was not valid JSON; falling back to page scraping", file=sys.stderr)

    return None


def _normalize_compare_url(url: str) -> str:
    parsed = urlparse(url.strip())
    scheme = (parsed.scheme or "https").lower()
    netloc = parsed.netloc.lower()
    path = parsed.path.rstrip("/")
    return f"{scheme}://{netloc}{path}"


def _extract_product_handles_from_soup(blog_url: str, soup: BeautifulSoup) -> list[str]:
    handles: list[str] = []
    seen: set[str] = set()
    for a in soup.find_all("a", href=True):
        href = urljoin(blog_url, str(a.get("href") or "").strip())
        parsed = urlparse(href)
        parts = [p for p in parsed.path.split("/") if p]
        if len(parts) < 2 or parts[0].lower() != "products":
            continue
        handle = parts[1].strip()
        if not handle or handle in seen:
            continue
        seen.add(handle)
        handles.append(handle)
    return handles


def fetch_shopify_admin_product_image(blog_url: str, soup: BeautifulSoup) -> str | None:
    """Prefer a product image linked from the blog, validated via product->blog backlink."""
    _ensure_shopify_env_loaded()

    product_handles = _extract_product_handles_from_soup(blog_url, soup)
    if not product_handles:
        return None

    myshopify_domain = _normalize_shop_domain(
        os.getenv("MYSHOPIFY_DOMAIN") or os.getenv("SHOPIFY_MYSHOPIFY_DOMAIN") or ""
    )
    if not myshopify_domain:
        return None

    token = _shopify_get_admin_token(myshopify_domain)
    if not token:
        return None

    headers = {
        "X-Shopify-Access-Token": token,
        "Content-Type": "application/json",
        "User-Agent": USER_AGENT,
    }
    base_url = f"https://{myshopify_domain}/admin/api/{SHOPIFY_API_VERSION}"

    normalized_blog_url = _normalize_compare_url(blog_url).lower()
    blog_path = urlparse(blog_url).path.rstrip("/").lower()
    fallback_image: str | None = None

    for handle in product_handles:
        endpoint = (
            f"{base_url}/products.json"
            f"?handle={quote(handle)}&limit=1&fields=id,title,handle,body_html,images"
        )
        try:
            resp = http_get(endpoint, headers=headers, timeout=30)
            products = resp.json().get("products", [])
            if not products:
                continue
            product = products[0]
            images = product.get("images") or []
            if not images or not isinstance(images[0], dict):
                continue

            src = images[0].get("src")
            if not isinstance(src, str) or not src.strip():
                continue
            image_src = _normalize_shopify_url(src.strip())

            if fallback_image is None:
                fallback_image = image_src

            body_html = str(product.get("body_html") or "").lower()
            if normalized_blog_url in body_html:
                return image_src
            if blog_path and blog_path in body_html:
                return image_src
        except requests.RequestException:
            continue
        except ValueError:
            continue

    return fallback_image


def fetch_blog(url: str) -> BlogContent:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        fail(f"Invalid URL scheme: {url!r}. Provide an http(s) URL.")

    try:
        resp = http_get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    except requests.RequestException as exc:
        fail(f"Failed to fetch blog URL: {exc}")

    soup = BeautifulSoup(resp.text, "html.parser")

    # Title
    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        title = h1.get_text(strip=True)

    # Body text: prefer <article> / <main>, fall back to all paragraphs.
    container = soup.find("article") or soup.find("main") or soup.body or soup
    paragraphs = [p.get_text(" ", strip=True) for p in container.find_all(["p", "li", "h2", "h3"])]
    text = "\n".join(p for p in paragraphs if p)
    text = text.strip()

    if not text:
        # Last resort: whole-page visible text.
        text = container.get_text("\n", strip=True)

    # Image candidates: Admin product image first, then Admin article image,
    # then page-derived candidates.
    candidates: list[str] = []

    admin_product_image = fetch_shopify_admin_product_image(url, soup)
    if admin_product_image:
        candidates.append(admin_product_image)

    admin_image = fetch_shopify_admin_article_image(url)
    if admin_image:
        candidates.append(admin_image)

    og = soup.find("meta", property="og:image")
    if og and og.get("content"):
        candidates.append(urljoin(url, og["content"]))

    for img in container.find_all("img"):
        src = img.get("src") or img.get("data-src") or img.get("data-original")
        # Shopify often uses srcset; grab the largest entry.
        srcset = img.get("srcset") or img.get("data-srcset")
        if srcset:
            largest = _largest_from_srcset(srcset)
            if largest:
                src = largest
        if src:
            candidates.append(urljoin(url, src))

    # De-duplicate, preserve order, drop obvious non-product assets.
    seen: set[str] = set()
    cleaned: list[str] = []
    for c in candidates:
        c = _normalize_shopify_url(c)
        if c in seen:
            continue
        seen.add(c)
        if _looks_like_asset_noise(c):
            continue
        cleaned.append(c)

    return BlogContent(url=url, title=title, text=text, image_candidates=cleaned)


def _largest_from_srcset(srcset: str) -> str | None:
    best_url = None
    best_w = -1
    for part in srcset.split(","):
        bits = part.strip().split()
        if not bits:
            continue
        candidate_url = bits[0]
        width = 0
        if len(bits) > 1 and bits[1].endswith("w"):
            try:
                width = int(bits[1][:-1])
            except ValueError:
                width = 0
        if width > best_w:
            best_w = width
            best_url = candidate_url
    return best_url


def _normalize_shopify_url(url: str) -> str:
    # Strip protocol-relative leading // and ensure https.
    if url.startswith("//"):
        url = "https:" + url
    if url.startswith("http://"):
        url = "https://" + url[len("http://") :]
    return url


def _looks_like_asset_noise(url: str) -> bool:
    low = url.lower()
    noise = ("logo", "icon", "sprite", "favicon", "placeholder", "pixel", "spacer", ".svg", "loading")
    return any(n in low for n in noise)


# --------------------------------------------------------------------------- #
# Step 3: Image selection
# --------------------------------------------------------------------------- #


def measure_image(url: str) -> tuple[int, int] | None:
    try:
        resp = http_get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
        with Image.open(io.BytesIO(resp.content)) as im:
            return im.size  # (width, height)
    except Exception:
        return None


def select_image(
    blog: BlogContent,
    cfg: dict[str, Any],
    client: OpenAI | None,
) -> SelectedImage:
    img_cfg = cfg.get("image_selection", {})
    min_w = int(img_cfg.get("min_width", 500))
    min_h = int(img_cfg.get("min_height", 500))
    max_candidates = int(img_cfg.get("max_candidates", 12))

    # Measure candidates and keep ones large enough to be real product shots.
    measured: list[SelectedImage] = []
    for url in blog.image_candidates[: max_candidates * 2]:
        size = measure_image(url)
        if not size:
            continue
        w, h = size
        if w >= min_w and h >= min_h:
            measured.append(SelectedImage(url=url, width=w, height=h, reason="meets size threshold"))
        if len(measured) >= max_candidates:
            break

    if not measured:
        fail(
            "No usable product image found on the blog page (image-to-video requires one). "
            "The downstream generator cannot run without a product image. Stopping."
        )

    # Optional vision-based ranking.
    use_vision = bool(cfg.get("openai", {}).get("use_vision_for_image_ranking", True))
    if use_vision and client is not None and len(measured) > 1:
        ranked = _rank_images_with_vision(client, cfg, blog, measured)
        if ranked is not None:
            return ranked

    # Fallback: largest area wins.
    best = max(measured, key=lambda s: s.width * s.height)
    best.reason = "largest image meeting size threshold (heuristic)"
    return best


def _rank_images_with_vision(
    client: OpenAI,
    cfg: dict[str, Any],
    blog: BlogContent,
    measured: list[SelectedImage],
) -> SelectedImage | None:
    model = cfg.get("openai", {}).get("model", "gpt-5.4")
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "You are selecting the single best PRODUCT image for a premium UGC ad. "
                "Prefer a clean, well-lit shot of the actual product (packaging/bottle/jar) "
                "that would animate well as cinematic product b-roll. Avoid banners, logos, "
                "infographics, charts, people-only lifestyle shots without the product, and decorative images.\n"
                f"Blog title: {blog.title}\n"
                "Reply with ONLY the integer index (0-based) of the best image."
            ),
        }
    ]
    for idx, img in enumerate(measured):
        content.append({"type": "text", "text": f"Index {idx} ({img.width}x{img.height}):"})
        content.append({"type": "image_url", "image_url": {"url": img.url}})

    try:
        resp = _chat_create(
            client,
            model=model,
            messages=[{"role": "user", "content": content}],
            temperature=0,
            max_output_tokens=10,
        )
        raw = (resp.choices[0].message.content or "").strip()
        digits = "".join(ch for ch in raw if ch.isdigit())
        if digits == "":
            return None
        idx = int(digits)
        if 0 <= idx < len(measured):
            chosen = measured[idx]
            chosen.reason = "selected by OpenAI vision as best product shot"
            return chosen
    except Exception as exc:  # pragma: no cover
        print(f"WARN: vision ranking failed, falling back to heuristic ({exc})", file=sys.stderr)
    return None


# --------------------------------------------------------------------------- #
# Step 4: Storyboard generation
# --------------------------------------------------------------------------- #


def _is_gpt5(model: str) -> bool:
    return model.lower().startswith("gpt-5")


@retry(
    reraise=True,
    stop=stop_after_attempt(4),
    wait=wait_exponential_jitter(initial=1, max=12),
    retry=retry_if_exception_type(OPENAI_RETRY_EXCEPTIONS),
)
def _chat_create(client: OpenAI, *, model: str, messages: list, temperature: float | None = None,
                 response_format: dict | None = None, max_output_tokens: int | None = None):
    """chat.completions.create wrapper that adapts params for GPT-5 models.

    GPT-5 chat models only accept the default temperature and use
    `max_completion_tokens` instead of `max_tokens`.
    """
    kwargs: dict[str, Any] = {"model": model, "messages": messages}
    if response_format is not None:
        kwargs["response_format"] = response_format
    if _is_gpt5(model):
        if max_output_tokens is not None:
            kwargs["max_completion_tokens"] = max_output_tokens
    else:
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_output_tokens is not None:
            kwargs["max_tokens"] = max_output_tokens
    return client.chat.completions.create(**kwargs)


def build_storyboard_schema(clip_specs: list[ClipSpec]) -> dict[str, Any]:
    """JSON schema optimized for downstream generation prompts."""
    role_enum = [s.role for s in clip_specs]
    clip_count = len(clip_specs)

    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "ideal_customer_profile",
            "primary_problem",
            "resolution_path",
            "concept",
            "call_to_action",
            "clips",
        ],
        "properties": {
            "ideal_customer_profile": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "person_name",
                    "age",
                    "career",
                    "hobbies",
                    "life_context",
                    "purchase_motivation",
                ],
                "properties": {
                    "person_name": {
                        "type": "string",
                        "description": "A single specific fictional first name for the target customer persona.",
                    },
                    "age": {
                        "type": "integer",
                        "minimum": 18,
                        "maximum": 85,
                        "description": "Exact age for the persona.",
                    },
                    "career": {
                        "type": "string",
                        "description": "Specific occupation/career of this person.",
                    },
                    "hobbies": {
                        "type": "array",
                        "minItems": 2,
                        "maxItems": 6,
                        "items": {"type": "string"},
                        "description": "2-6 concrete hobbies/interests this person has.",
                    },
                    "life_context": {
                        "type": "string",
                        "description": "Concise daily routine/life situation context.",
                    },
                    "purchase_motivation": {
                        "type": "string",
                        "description": "Why this specific person would buy this product now.",
                    },
                },
            },
            "primary_problem": {
                "type": "string",
                "description": "LLM-inferred main customer pain/problem this ad should dramatize.",
            },
            "resolution_path": {
                "type": "string",
                "description": "LLM-inferred problem-to-resolution arc tied to product usage.",
            },
            "concept": {"type": "string"},
            "call_to_action": {"type": "string"},
            "clips": {
                "type": "array",
                "minItems": clip_count,
                "maxItems": clip_count,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "role",
                        "duration_seconds",
                        "use_image_reference",
                        "video_prompt",
                        "audio_prompt",
                    ],
                    "properties": {
                        "role": {"type": "string", "enum": role_enum},
                        "duration_seconds": {"type": "integer"},
                        "use_image_reference": {
                            "type": "boolean",
                            "description": "Whether this clip should be conditioned on the product image reference.",
                        },
                        "video_prompt": {
                            "type": "string",
                            "description": "Detailed visual direction for image-to-video generation.",
                        },
                        "audio_prompt": {
                            "type": "string",
                            "description": "Audio direction for Grok AUDIO section: narration + music + SFX + ambience + optional short dialogue.",
                        },
                    },
                },
            },
        },
    }


def generate_storyboard(
    client: OpenAI,
    cfg: dict[str, Any],
    blog: BlogContent,
    clip_specs: list[ClipSpec],
) -> dict[str, Any]:
    ad = cfg.get("ad", {})
    vo = cfg.get("voiceover", {})
    brand = cfg.get("brand", {})
    product = cfg.get("product", {})
    compliance = cfg.get("compliance", {})

    target_seconds = int(ad.get("target_seconds", 30))

    model = cfg.get("openai", {}).get("model", "gpt-5.4")
    temperature = float(cfg.get("openai", {}).get("temperature", 0.8))
    max_output_tokens = int(cfg.get("openai", {}).get("max_output_tokens", 7000))

    compliance_rules = "\n".join(f"- {r}" for r in compliance.get("rules", []))
    structure_lines = [
        f"- clip {s.index}: role={s.role}, duration={s.min_seconds}-{s.max_seconds}s"
        for s in clip_specs
    ]
    structure_text = "\n".join(structure_lines)

    # Trim blog text to keep token usage reasonable.
    blog_text = blog.text[:8000]

    system_prompt = textwrap.dedent(
        f"""
        You are an expert direct-response UGC ad creative director and copywriter.
        You write person-led UGC scripts optimized for xai/grok-imagine-video generation.

        CREATIVE DECISION AUTHORITY:
        - You must infer ALL key creative decisions per product from the source blog content.
        - Do not ask for missing persona details. Infer the most plausible and specific customer profile.
        - You must produce: ideal_customer_profile, primary_problem, and resolution_path.
        - ideal_customer_profile MUST represent one specific fictional person and include: person_name, exact age, career, 2-6 hobbies, life_context, purchase_motivation.

        PERSON-LED UGC REQUIREMENTS:
        - Build one cohesive narrative around a single on-camera creator persona.
        - Use the same named persona consistently across all person-led clips.
        - Keep the person visible in all person-led clips.
        - Start each person-led clip with person-only framing for the first 1.8 to 2.0 seconds.
        - For hook/problem and solution clips, keep strict continuity: same person, same age, same general appearance, and coherent wardrobe/location progression.
        - In solution clips, show practical product use and an early believable improvement.
        - Product hero / soft CTA clips must be product-only with no person on camera.

        VIDEO MODEL — xai/grok-imagine-video (image-to-video):
        - For each clip decide `use_image_reference` = true/false.
        - Set `use_image_reference=true` when product identity continuity is important.
        - Set `use_image_reference=false` when person/problem realism should dominate.
        - If `use_image_reference=true`, integrate the product naturally into a human scene.
        - No negative prompts. No on-screen text.
        - Produce EXACTLY the clips defined by structure.md in the same order:
    {structure_text}
                - Each clip must include one detailed visual prompt (`video_prompt`) and one audio prompt (`audio_prompt`).
            - `video_prompt` should be detailed and precise (roughly 90-160 words) with camera + motion + lighting direction.
        - Keep each short clip focused on one primary beat and one dominant camera move.
                - Use the Grok readme audio guidance: include background music, sound effects, ambient audio, and short dialogue when useful.
                - `audio_prompt` should be written to fit an `AUDIO:` section and include:
                on-camera dialogue cues + narration delivery + background music + SFX + ambient audio.
                - Keep spoken narration concise so it can be delivered naturally without rushing (target around 2.1 words/sec or less).
                - Audio and visual intent must be synchronized.
                - Keep each clip duration inside its specified band from structure.md.

        COMPLIANCE (moderate but safe):
        {compliance_rules or '- Use wellness language only; avoid medical/disease claims.'}
        - Strong wellness framing is allowed, but do not claim cure, treatment, or disease prevention.

        Keep total runtime near {target_seconds} seconds while respecting each clip's fixed role and duration band.
            The clips should flow as one cohesive problem-to-solution UGC narrative.
        Prompts must be precise and deterministic enough to reduce bad generations.
        Return ONLY valid JSON matching the provided schema.
        """
    ).strip()

    user_prompt = textwrap.dedent(
        f"""
        BRAND: {brand.get('name', '')} ({brand.get('website', '')})
        PRODUCT NAME (may be blank — infer from blog): {product.get('name', '')}
        PRODUCT CATEGORY: {product.get('category', '')}
        STATED KEY BENEFITS (may be empty): {', '.join(product.get('key_benefits', []) or [])}
        PLATFORM: {ad.get('platform', '')}  ASPECT: {ad.get('aspect_ratio', '9:16')}
        STYLE: {ad.get('style', '')}
        CALL TO ACTION: {ad.get('call_to_action', '')}
        NARRATION TONE: {vo.get('tone', 'calm, trustworthy expert')}
        TARGET LENGTH: ~{target_seconds}s
        VIDEO QUALITY PRIORITY: authentic person-led UGC realism, low artifact risk, strict continuity
        AUDIO QUALITY PRIORITY: natural trustworthy delivery, clear articulation, platform-ready pacing
        STYLE HINT (soft, override if conflicting): {ad.get('style', '')}

        SOURCE BLOG TITLE: {blog.title}
        SOURCE BLOG URL: {blog.url}

        SOURCE BLOG CONTENT (use this for accurate product facts and benefits):
        \"\"\"
        {blog_text}
        \"\"\"
        """
    ).strip()

    schema = build_storyboard_schema(clip_specs)

    try:
        resp = _chat_create(
            client,
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "ugc_storyboard", "strict": True, "schema": schema},
            },
        )
    except Exception as exc:
        # Fallback for models/SDKs that don't support json_schema — use json_object.
        print(f"WARN: structured json_schema call failed ({exc}); retrying with json_object mode.", file=sys.stderr)
        resp = _chat_create(
            client,
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            messages=[
                {"role": "system", "content": system_prompt + "\n\nReturn ONLY a JSON object."},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )

    raw = resp.choices[0].message.content or "{}"
    try:
        creative = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(
            f"WARN: invalid JSON from OpenAI ({exc}); retrying once with compact-json instructions.",
            file=sys.stderr,
        )
        retry_resp = _chat_create(
            client,
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            messages=[
                {
                    "role": "system",
                    "content": (
                        system_prompt
                        + "\n\nCRITICAL: Return valid JSON only (no markdown), "
                        "and keep values concise while preserving technical precision."
                    ),
                },
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        raw_retry = retry_resp.choices[0].message.content or "{}"
        try:
            creative = json.loads(raw_retry)
        except json.JSONDecodeError as retry_exc:
            fail(f"OpenAI did not return valid JSON after retry: {retry_exc}")

    return creative


def count_words(text: str) -> int:
    return len([w for w in (text or "").split() if w.strip()])


def extract_narration_text(audio_prompt: str) -> str:
    text = (audio_prompt or "").strip()
    if not text:
        return ""

    narration_match = re.search(r"narration\s*:\s*", text, flags=re.IGNORECASE)
    if narration_match:
        tail = text[narration_match.end() :]
        stop_match = re.search(r"\b(music|sfx|ambient|on-camera|dialogue)\s*:", tail, flags=re.IGNORECASE)
        text = tail[: stop_match.start()] if stop_match else tail

    quoted = re.findall(r"[\"\u201c\u201d]([^\"\u201c\u201d]{8,})[\"\u201c\u201d]", text)
    if quoted:
        text = " ".join(quoted)

    return re.sub(r"\s+", " ", text).strip()


def narration_word_tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9+']+", text)


def quoted_narration_segments(audio_prompt: str) -> list[dict[str, Any]]:
    pattern = re.compile(r'"([^"]+)"|“([^”]+)”')
    segments: list[dict[str, Any]] = []
    for match in pattern.finditer(audio_prompt or ""):
        text = (match.group(1) or match.group(2) or "").strip()
        if not text:
            continue
        style = "straight" if match.group(1) is not None else "curly"
        tokens = narration_word_tokens(text)
        segments.append(
            {
                "start": match.start(),
                "end": match.end(),
                "style": style,
                "text": text,
                "tokens": tokens,
                "word_count": len(tokens),
            }
        )
    return segments


def fit_audio_prompt_to_duration(
    audio_prompt: str,
    duration_seconds: int,
    max_narration_wps: float,
) -> tuple[str, int, int]:
    def trim_segment_text(text: str, max_words: int) -> str:
        normalized = re.sub(r"\s+", " ", (text or "").strip())
        if max_words <= 0 or not normalized:
            return ""

        if len(narration_word_tokens(normalized)) <= max_words:
            return normalized

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", normalized) if s.strip()]
        if not sentences:
            # No clear sentence boundaries: keep original to avoid chopped phrasing.
            return normalized

        kept_sentences: list[str] = []
        remaining = max_words

        for sentence in sentences:
            wc = len(narration_word_tokens(sentence))
            if wc <= remaining:
                kept_sentences.append(sentence)
                remaining -= wc
            else:
                break

        if kept_sentences:
            return " ".join(kept_sentences).strip()

        # If the first sentence alone exceeds budget, keep original to avoid incomplete wording.
        return normalized

    segments = quoted_narration_segments(audio_prompt)
    total_words = sum(int(seg["word_count"]) for seg in segments)
    if total_words <= 0:
        return audio_prompt, total_words, total_words

    safe_wps = max(1.0, max_narration_wps)
    allowed_words = max(6, int(math.floor(duration_seconds * safe_wps)))
    if total_words <= allowed_words:
        return audio_prompt, total_words, total_words

    if allowed_words < len(segments):
        allowed_words = len(segments)

    updated = audio_prompt
    new_total_words = 0
    remaining_budget = allowed_words
    replacements: list[dict[str, Any]] = []

    for idx, seg in enumerate(segments):
        remaining_segments = len(segments) - idx - 1
        min_reserve_for_rest = remaining_segments
        max_for_this = max(1, remaining_budget - min_reserve_for_rest)
        keep_words = min(int(seg["word_count"]), max_for_this)

        trimmed_text = trim_segment_text(str(seg["text"]), keep_words)
        trimmed_tokens = narration_word_tokens(trimmed_text)
        if not trimmed_tokens:
            trimmed_text = str(seg["text"])
            trimmed_tokens = list(seg["tokens"])
        actual_keep = len(trimmed_tokens)

        remaining_budget = max(remaining_segments, remaining_budget - actual_keep)
        new_total_words += actual_keep

        quoted = f'"{trimmed_text}"' if seg["style"] == "straight" else f"“{trimmed_text}”"
        replacements.append({"start": int(seg["start"]), "end": int(seg["end"]), "value": quoted})

    for rep in reversed(replacements):
        updated = updated[: rep["start"]] + rep["value"] + updated[rep["end"] :]

    if new_total_words >= total_words:
        # No useful reduction achieved without harming phrasing.
        return audio_prompt, total_words, total_words

    return updated, total_words, new_total_words


def estimate_hook_duration_seconds(
    audio_prompt: str,
    words_per_second: float,
    min_seconds: int,
    max_seconds: int,
) -> int:
    narration = extract_narration_text(audio_prompt)
    word_count = len(narration_word_tokens(narration))
    if word_count <= 0:
        return min_seconds

    safe_wps = words_per_second if words_per_second > 0 else 2.5
    estimated = int(math.ceil(word_count / safe_wps)) + 1
    return max(min_seconds, min(max_seconds, estimated))


def build_master_prompt_from_clip(clip: dict[str, Any], min_words: int) -> str:
    """Deterministically expand a clip into a detailed master prompt."""
    shot = (clip.get("shot_description") or "Authentic UGC scene with one person and product context.").strip()
    objective = (clip.get("objective") or "Show a relatable customer problem-to-solution moment.").strip()

    camera = clip.get("camera_plan", {}) or {}
    motion = clip.get("motion_plan", {}) or {}
    lighting = clip.get("lighting_plan", {}) or {}
    composition = clip.get("composition_plan", {}) or {}
    style = clip.get("style_plan", {}) or {}

    beats = motion.get("beat_timeline", []) or []
    beats_text = "; ".join(str(b).strip() for b in beats if str(b).strip())
    if not beats_text:
        beats_text = "establish product, refine framing, hold premium hero finish"

    quality_targets = ", ".join(str(x).strip() for x in (clip.get("quality_targets", []) or []) if str(x).strip())
    if not quality_targets:
        quality_targets = "stable label readability, smooth motion, premium lighting consistency"

    failure_avoidance = ", ".join(
        str(x).strip() for x in (clip.get("failure_avoidance", []) or []) if str(x).strip()
    )
    if not failure_avoidance:
        failure_avoidance = "preserve product geometry and label proportions; avoid jitter or abrupt reframing"

    prompt = (
        f"{shot} "
        f"Objective: {objective}. "
        f"Camera direction: {camera.get('shot_size', 'close-up')} framing with {camera.get('lens_feel', 'cinematic lens')}, "
        f"{camera.get('movement', 'slow controlled move')} at {camera.get('movement_speed', 'steady pace')}, "
        f"focus behavior: {camera.get('focus_behavior', 'keep face and product moments sharp and stable')}. "
        f"Motion design: primary action {motion.get('primary_action', 'natural human action tied to the scene intent')}; "
        f"micro motion {motion.get('secondary_micro_motion', 'gentle highlight drift')}. "
        f"Timeline beats: {beats_text}. "
        f"Lighting: key {lighting.get('key_light', 'soft directional key')}, "
        f"fill {lighting.get('fill_light', 'clean diffused fill')}, "
        f"specular strategy {lighting.get('specular_strategy', 'controlled premium edge highlights')}, "
        f"mood {lighting.get('mood', 'calm and premium')}, color tone {lighting.get('color_tone', 'clean neutral')}. "
        f"Composition: subject placement {composition.get('subject_placement', 'person-led framing with product context')}, "
        f"background {composition.get('background_treatment', 'realistic lifestyle environment')}, "
        f"depth strategy {composition.get('depth_strategy', 'strong foreground-background separation')}, "
        f"negative space {composition.get('negative_space_guidance', 'keep clean breathing room')}. "
        f"Style: {style.get('look', 'premium commercial')}, texture {style.get('texture', 'crisp natural materials')}, "
        f"grade {style.get('grade', 'clean contrast and controlled highlights')}, "
        f"realism {style.get('realism_level', 'photoreal')}, finish {style.get('commercial_finish', 'high-end ad polish')}. "
        f"Quality priorities: {quality_targets}. "
        f"Consistency checks: {failure_avoidance}. "
        "Keep one subject, one primary action, one camera move, physically plausible motion, and no on-screen text. "
        "Favor authentic human behavior over static product-only framing."
    )

    # Ensure floor is always met, even if some fields are sparse.
    while count_words(prompt) < min_words:
        prompt += (
            " Preserve person continuity and believable behavior across the clip, maintain stable framing, "
            "and finish with a clear product-context payoff suitable for conversion-focused UGC advertising."
        )

    return prompt


def normalize_ideal_customer_profile(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        fail(
            "OpenAI response field ideal_customer_profile must be an object with "
            "person_name, age, career, hobbies, life_context, purchase_motivation."
        )

    person_name = str(raw.get("person_name", "")).strip()
    age = raw.get("age")
    career = str(raw.get("career", "")).strip()
    life_context = str(raw.get("life_context", "")).strip()
    purchase_motivation = str(raw.get("purchase_motivation", "")).strip()

    hobbies_raw = raw.get("hobbies", [])
    hobbies: list[str] = []
    if isinstance(hobbies_raw, list):
        hobbies = [str(h).strip() for h in hobbies_raw if str(h).strip()]

    if not person_name or len(person_name.split()) > 4:
        fail("ideal_customer_profile.person_name must be a specific concise name for one person")
    if not isinstance(age, int) or age < 18 or age > 85:
        fail("ideal_customer_profile.age must be an integer between 18 and 85")
    if not career:
        fail("ideal_customer_profile.career is required")
    if len(hobbies) < 2:
        fail("ideal_customer_profile.hobbies must contain at least 2 specific hobbies")
    if not life_context:
        fail("ideal_customer_profile.life_context is required")
    if not purchase_motivation:
        fail("ideal_customer_profile.purchase_motivation is required")

    return {
        "person_name": person_name,
        "age": age,
        "career": career,
        "hobbies": hobbies,
        "life_context": life_context,
        "purchase_motivation": purchase_motivation,
    }


def validate_storyboard_payload(storyboard: dict[str, Any]) -> dict[str, Any]:
    try:
        validated = StoryboardModel.model_validate(storyboard)
    except ValidationError as exc:
        fail(f"Assembled storyboard failed schema validation: {exc}")
    return validated.model_dump(mode="python")


# --------------------------------------------------------------------------- #
# Step 5: Assemble + validate + write outputs
# --------------------------------------------------------------------------- #


def assemble_storyboard(
    creative: dict[str, Any],
    cfg: dict[str, Any],
    blog: BlogContent,
    image: SelectedImage,
    clip_specs: list[ClipSpec],
) -> dict[str, Any]:
    ad = cfg.get("ad", {})
    quality = cfg.get("quality", {})
    aspect = ad.get("aspect_ratio", "9:16")
    resolution = ad.get("resolution", "480p")
    min_master_prompt_words = int(quality.get("min_master_prompt_words", 60))
    auto_expand_short_master_prompt = bool(quality.get("auto_expand_short_master_prompt", True))
    voice_cfg = cfg.get("voiceover", {})
    words_per_second = float(voice_cfg.get("words_per_second", 2.5) or 2.5)
    max_narration_wps = float(voice_cfg.get("max_narration_words_per_second", words_per_second) or words_per_second)

    clip_specs_by_role = {s.role: s for s in clip_specs}
    raw_role_map: dict[str, dict[str, Any]] = {}
    for raw in creative.get("clips", []):
        role = canonical_role(str(raw.get("role", "")).strip())
        if role in clip_specs_by_role and role not in raw_role_map:
            raw_role_map[role] = raw

    missing_roles = [s.role for s in clip_specs if s.role not in raw_role_map]
    if missing_roles:
        fail(
            "OpenAI response is missing required clip roles from structure.md: "
            + ", ".join(missing_roles)
        )

    clips_out: list[dict[str, Any]] = []
    target_seconds = int(ad.get("target_seconds", 30))

    for spec in clip_specs:
        raw_clip = raw_role_map[spec.role]
        duration = int(raw_clip.get("duration_seconds", spec.min_seconds))
        duration = max(spec.min_seconds, min(spec.max_seconds, duration))

        use_image_reference = bool(raw_clip.get("use_image_reference", True))

        master_prompt = (raw_clip.get("video_prompt") or "").strip()
        master_prompt_word_count = count_words(master_prompt)
        if master_prompt_word_count < min_master_prompt_words and auto_expand_short_master_prompt:
            role_objective = {
                "hook": "Show the ideal customer in a relatable opening moment.",
                "problem": "Show the customer experiencing the core problem clearly.",
                "solution": "Show the same customer using the product and reaching an early believable improvement.",
                "education": "Explain the support mechanism through the person context.",
                "product": "Show practical product use in the customer's routine.",
                "soft_cta": "Show improved state and natural recommendation tone.",
                "story_arc": "In one continuous clip, show hook, problem, product-use action, and early resolution.",
                "product_hero": "Deliver a clean product-focused final payoff with soft CTA tone.",
            }
            shot_prefix = "Product-only" if spec.role in {"product_hero", "soft_cta"} else "Person-led"
            seed_clip = {
                "role": spec.role,
                "shot_description": f"{shot_prefix} {spec.role_label} clip for {blog.title}",
                "objective": role_objective.get(spec.role, f"Deliver the {spec.role_label} stage clearly and conversion-oriented."),
                "narration_segment": raw_clip.get("narration_segment", ""),
                "quality_targets": ["clear subject", "stable motion", "premium lighting"],
                "failure_avoidance": ["avoid jitter", "avoid identity drift", "no text overlays"],
            }
            master_prompt = build_master_prompt_from_clip(seed_clip, min_master_prompt_words)
            master_prompt_word_count = count_words(master_prompt)
            print(
                f"WARN: role={spec.role} master prompt too short; auto-expanded to {master_prompt_word_count} words.",
                file=sys.stderr,
            )

        audio_prompt = str(raw_clip.get("audio_prompt", "")).strip()
        if not audio_prompt:
            audio_prompt = "calm trustworthy narration, soft uplifting ambient bed, subtle room tone, clean product handling sounds"

        if spec.role == "hook":
            duration = estimate_hook_duration_seconds(
                audio_prompt=audio_prompt,
                words_per_second=words_per_second,
                min_seconds=spec.min_seconds,
                max_seconds=spec.max_seconds,
            )

        fitted_audio_prompt, before_words, after_words = fit_audio_prompt_to_duration(
            audio_prompt=audio_prompt,
            duration_seconds=duration,
            max_narration_wps=max_narration_wps,
        )
        if fitted_audio_prompt != audio_prompt:
            print(
                f"WARN: role={spec.role} narration trimmed from {before_words} to {after_words} words "
                f"to fit {duration}s clip pacing.",
                file=sys.stderr,
            )
            audio_prompt = fitted_audio_prompt

        if spec.role in {"hook", "problem", "solution", "education", "product", "story_arc"}:
            continuity_tail = "Keep the same named on-camera persona continuity across person-led clips."
            if continuity_tail.lower() not in master_prompt.lower():
                master_prompt = f"{master_prompt.rstrip()} {continuity_tail}".strip()

            person_only_opening_tail = "Start with person-only framing for the first 1.8 to 2.0 seconds."
            if person_only_opening_tail.lower() not in master_prompt.lower():
                master_prompt = f"{master_prompt.rstrip()} {person_only_opening_tail}".strip()

        if spec.role in {"product_hero", "soft_cta"}:
            product_only_tail = "Product-only shot. No people, faces, or body parts visible."
            if "no people" not in master_prompt.lower():
                master_prompt = f"{master_prompt.rstrip()} {product_only_tail}".strip()

        clips_out.append(
            {
                "clip_index": spec.index,
                "role": spec.role,
                "duration_seconds": duration,
                "use_image_reference": use_image_reference,
                "video_prompt": master_prompt,
                "audio_prompt": audio_prompt,
            }
        )

    def _set_clip_duration(clip_obj: dict[str, Any], new_duration: int) -> None:
        clip_obj["duration_seconds"] = new_duration

    min_total = sum(s.min_seconds for s in clip_specs)
    max_total = sum(s.max_seconds for s in clip_specs)
    effective_target_seconds = max(min_total, min(max_total, target_seconds))
    locked_duration_roles = {"hook"}

    total_seconds = sum(c["duration_seconds"] for c in clips_out)
    if clips_out and total_seconds != effective_target_seconds:
        if total_seconds < effective_target_seconds:
            deficit = effective_target_seconds - total_seconds
            for clip_obj in clips_out:
                if deficit <= 0:
                    break
                if str(clip_obj.get("role", "")) in locked_duration_roles:
                    continue
                spec = clip_specs_by_role.get(str(clip_obj["role"]))
                max_s = spec.max_seconds if spec else int(clip_obj["duration_seconds"])
                room = max_s - int(clip_obj["duration_seconds"])
                if room <= 0:
                    continue
                add = min(room, deficit)
                _set_clip_duration(clip_obj, int(clip_obj["duration_seconds"]) + add)
                deficit -= add
        else:
            excess = total_seconds - effective_target_seconds
            for clip_obj in reversed(clips_out):
                if excess <= 0:
                    break
                if str(clip_obj.get("role", "")) in locked_duration_roles:
                    continue
                spec = clip_specs_by_role.get(str(clip_obj["role"]))
                min_s = spec.min_seconds if spec else int(clip_obj["duration_seconds"])
                room = int(clip_obj["duration_seconds"]) - min_s
                if room <= 0:
                    continue
                sub = min(room, excess)
                _set_clip_duration(clip_obj, int(clip_obj["duration_seconds"]) - sub)
                excess -= sub

    total_seconds = sum(c["duration_seconds"] for c in clips_out)
    ideal_customer_profile = normalize_ideal_customer_profile(creative.get("ideal_customer_profile", {}))

    storyboard: dict[str, Any] = {
        "schema_version": "4.2",
        "generated_by": "generate_ugc_script.py",
        "source": {"blog_url": blog.url, "blog_title": blog.title},
        "structure_source": "structure.md",
        "ad": {
            "platform": ad.get("platform", ""),
            "aspect_ratio": aspect,
            "resolution": resolution,
            "target_seconds": ad.get("target_seconds", 30),
            "total_clip_seconds": total_seconds,
            "style": ad.get("style", ""),
        },
        "brand": cfg.get("brand", {}),
        "story": {
            "ideal_customer_profile": ideal_customer_profile,
            "primary_problem": creative.get("primary_problem", ""),
            "resolution_path": creative.get("resolution_path", ""),
            "concept": creative.get("concept", ""),
            "call_to_action": creative.get("call_to_action", ""),
        },
        "video_generation": {
            "model": "xai/grok-imagine-video",
            "image_ref": image.url,
            "input_defaults": {
                "aspect_ratio": aspect,
                "resolution": resolution,
            },
            "prompt_assembly": "final_prompt = video_prompt + '\\nAUDIO: ' + audio_prompt",
            "clips": clips_out,
        },
    }
    return validate_storyboard_payload(storyboard)


def write_outputs(storyboard: dict[str, Any], cfg: dict[str, Any], out_dir: str, source_url: str) -> str:
    _ = cfg  # reserved for future output settings
    os.makedirs(out_dir, exist_ok=True)

    base_slug = slug_from_url(source_url)
    json_path = next_available_json_path(out_dir, base_slug)

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(storyboard, fh, indent=2, ensure_ascii=False)

    return json_path


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a UGC ad script file from a blog URL + config.")
    parser.add_argument("--url", required=True, help="Product blog URL.")
    parser.add_argument("--config", required=True, help="Path to config JSON.")
    parser.add_argument("--out", default=None, help="Output directory (overrides config.output.dir).")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        fail("OPENAI_API_KEY not set. Add it to your .env file.")

    cfg = load_config(args.config)
    out_dir = args.out or cfg.get("output", {}).get("dir", "output")
    workspace_dir = os.path.dirname(os.path.abspath(__file__))
    structure_path = os.path.join(workspace_dir, "structure.md")

    client = OpenAI(api_key=api_key)

    print(f"[0/5] Loading clip structure: {structure_path}")
    clip_specs = load_clip_structure(structure_path)

    print(f"[1/5] Fetching blog: {args.url}")
    blog = fetch_blog(args.url)
    if not blog.text:
        fail("Could not extract any text content from the blog page.")
    print(f"      Title: {blog.title!r} · {len(blog.image_candidates)} image candidate(s)")

    print("[2/5] Selecting best product image…")
    image = select_image(blog, cfg, client)
    print(f"      Chosen: {image.url} ({image.width}x{image.height}) — {image.reason}")

    print("[3/5] Generating structured prompt plan with OpenAI…")
    creative = generate_storyboard(client, cfg, blog, clip_specs)
    storyboard = assemble_storyboard(creative, cfg, blog, image, clip_specs)
    print(
        f"      {len(storyboard['video_generation']['clips'])} clips · {storyboard['ad']['total_clip_seconds']}s total"
    )

    print("[4/5] Writing URL-named JSON file…")
    json_path = write_outputs(storyboard, cfg, out_dir, args.url)
    print(f"      JSON prompt script: {json_path}")

    print("[5/5] Complete")
    print("Done. No clips or audio were generated (handled by the downstream script).")


if __name__ == "__main__":
    main()
