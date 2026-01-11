"""
Batch photo restoration using Google Vertex AI (Gemini 3 Pro Image Preview).

Features
- CLI flags for input/output folders and per-run image limit.
- Best-effort billing account check (prints billing accounts; Google does not expose remaining free-trial credit via API).
- Resume-safe processing (skips images already written).
- Optional PyQt6 UI with guided setup steps and live log output.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import io
import traceback
import subprocess
import sys
import time
import sqlite3
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple

try:
    import requests  # type: ignore
except ImportError:  # pragma: no cover
    print("requests not installed. Install with `pip install requests`.", file=sys.stderr)
    sys.exit(1)

try:
    from PIL import Image, ImageOps  # type: ignore
except ImportError:  # pragma: no cover
    print("Pillow not installed. Install with `pip install Pillow`.", file=sys.stderr)
    sys.exit(1)

# PyQt6 is optional; the CLI works without it.
try:  # pragma: no cover
    from PyQt6 import QtCore, QtGui, QtWidgets

    PYQT_AVAILABLE = True
except Exception:
    PYQT_AVAILABLE = False


# --- CONFIG DEFAULTS (override with CLI flags or env vars) ---
MODEL_ID = os.environ.get("GEMINI_MODEL_ID", "gemini-3-pro-image-preview")
REGION = os.environ.get("GEMINI_REGION", "global")
PROMPT = """
ROLE: Senior Digital Imaging Technician (DIT) & Restoration Artist.

INPUT: [Image]

TASK:
1. FRAMING: Detect/Remove borders. Do NOT crop image content.
2. COLORIZATION: If B&W -> Colorize (Kodachrome 64). If Color -> Correct fading.
3. LOOK: Canon R5, 50mm, f/1.8.
4. RESTORATION: Treat grain as noise (Chroma Denoise). Keep texture (Subsurface Scattering).
5. TONAL RECOVERY: Lift crushed blacks.
6. TEXTURE: Clean skin texture. No artifacts.

OUTPUT: A 4K, full-color, borderless digital photo.
""".strip()

COST_PER_IMAGE_USD = 0.25  # update if pricing changes
DB_PATH = Path(__file__).with_name("runs.db")
PREFS_PATH = Path(__file__).with_name(".restorer_prefs.json")


# --- DATA STRUCTURES ---
@dataclass
class BatchConfig:
    project_id: str
    input_dir: Path
    output_dir: Path
    max_images: Optional[int] = None  # None = all
    start_index: int = 0
    refresh_every: int = 10
    timeout: int = 120
    prompt: str = PROMPT
    dry_run: bool = False  # Skip API calls; copy input to output for flow testing.
    check_orientation: bool = False  # Warn if EXIF rotation flags are present.
    recursive: bool = False  # Include subfolders when scanning input.


# --- UTILITIES ---
def run_cmd(cmd: List[str]) -> Tuple[int, str, str]:
    """Run command and return (returncode, stdout, stderr)."""
    env = os.environ.copy()
    # Prevent gcloud from blocking on interactive prompts (component install/update).
    env.setdefault("CLOUDSDK_CORE_DISABLE_PROMPTS", "1")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=20)
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except subprocess.TimeoutExpired:
        return 124, "", "command timed out (possibly waiting for input)"
    except FileNotFoundError:
        return 127, "", "command not found"


def get_project_id() -> str:
    code, out, err = run_cmd(["gcloud", "config", "get-value", "project"])
    if code != 0 or not out:
        raise RuntimeError(
            "Could not read current gcloud project. Run `gcloud auth login` and set a project."
        )
    # Verify the configured project actually exists; stale configs can break billing checks.
    desc_code, _, desc_err = run_cmd(["gcloud", "projects", "describe", out])
    if desc_code != 0:
        raise RuntimeError(
            f"Configured project '{out}' is not accessible. Run `gcloud projects list` and then `gcloud config set project <id>`."
        )
    return out


def get_active_account() -> Optional[str]:
    # Fast path
    code, out, _ = run_cmd(["gcloud", "config", "get-value", "account"])
    if code == 0 and out and out != "(unset)":
        return out
    # Fallback
    code, out, _ = run_cmd(["gcloud", "auth", "list", "--format=value(account)", "--filter=status:ACTIVE"])
    if code == 0 and out:
        return out.splitlines()[0]
    return None


def list_projects() -> List[str]:
    code, out, err = run_cmd(
        ["gcloud", "projects", "list", "--format=value(projectId)"]
    )
    if code != 0 or not out:
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


# --- PERSISTENCE / DB ---
def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                input_path TEXT,
                output_path TEXT,
                status TEXT,
                message TEXT,
                input_hash TEXT,
                output_hash TEXT,
                started_at REAL,
                ended_at REAL,
                duration_sec REAL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def log_image_start(input_path: Path, output_path: Path) -> int:
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.execute(
            "INSERT INTO runs (input_path, output_path, status, started_at) VALUES (?, ?, ?, ?)",
            (str(input_path), str(output_path), "running", time.time()),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def log_image_end(row_id: int, status: str, message: str, input_hash: str, output_hash: str, duration: float) -> None:
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            """
            UPDATE runs
            SET status=?, message=?, input_hash=?, output_hash=?, ended_at=?, duration_sec=?
            WHERE id=?
            """,
            (status, message, input_hash, output_hash, time.time(), duration, row_id),
        )
        conn.commit()
    finally:
        conn.close()


def load_prefs() -> dict:
    if PREFS_PATH.exists():
        try:
            return json.loads(PREFS_PATH.read_text())
        except Exception:
            return {}
    return {}


def save_prefs(data: dict) -> None:
    try:
        PREFS_PATH.write_text(json.dumps(data, indent=2))
    except Exception:
        pass


def get_access_token() -> str:
    code, out, _ = run_cmd(["gcloud", "auth", "print-access-token"])
    if code == 0 and out:
        return out
    # One retry after short sleep.
    time.sleep(2)
    code, out, err = run_cmd(["gcloud", "auth", "print-access-token"])
    if code != 0:
        raise RuntimeError(f"Auth token fetch failed: {err or 'unknown error'}")
    return out


def best_effort_billing_accounts() -> List[dict]:
    """
    Vertex AI does not expose remaining free-trial credits via API.
    This helper lists billing accounts to confirm you have one attached.
    """
    code, out, err = run_cmd(["gcloud", "beta", "billing", "accounts", "list", "--format=json"])
    if code != 0:
        return []
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return []


def run_setup_check() -> None:
    """
    CLI preflight to get from 'have Google account' to 'ready to run'.
    Non-destructive: only reads config and reports missing pieces with next-step commands.
    """

    def status(label: str, ok: bool, detail: str = "") -> None:
        prefix = "✅" if ok else "⚠️"
        print(f"{prefix} {label}" + (f": {detail}" if detail else ""))

    # 0) gcloud present?
    code, out, err = run_cmd(["gcloud", "--version"])
    if code != 0:
        status("gcloud SDK", False, "Install: brew install --cask google-cloud-sdk")
        return
    status("gcloud SDK", True, out.splitlines()[0] if out else "found")

    # 1) Auth
    code, out, err = run_cmd(["gcloud", "auth", "list", "--format=json"])
    if code != 0 or not out.strip():
        status("Auth", False, "Run: gcloud auth login")
        return
    try:
        accounts = json.loads(out)
    except json.JSONDecodeError:
        accounts = []
    active = [a for a in accounts if a.get("status") == "ACTIVE"]
    status("Auth", bool(active), active[0]["account"] if active else "Run: gcloud auth login")

    # 2) Project set
    code, proj, _ = run_cmd(["gcloud", "config", "get-value", "project"])
    if code != 0 or not proj or proj == "(unset)":
        status("Project", False, "Set/create: gcloud projects create <id>; gcloud config set project <id>")
        return
    status("Project", True, proj)

    # 3) Project exists
    code, _, err = run_cmd(["gcloud", "projects", "describe", proj])
    if code != 0:
        status("Project exists", False, f"Create or pick a valid project. Error: {err[:120]}")
        return
    status("Project exists", True)

    # 4) Billing account linked
    code, out, err = run_cmd(
        ["gcloud", "beta", "billing", "projects", "describe", proj, "--format=json"]
    )
    linked = False
    if code == 0 and out:
        try:
            info = json.loads(out)
            linked = bool(info.get("billingEnabled"))
        except json.JSONDecodeError:
            linked = False
    if linked:
        status("Billing link", True)
    else:
        status(
            "Billing link",
            False,
            "Link a billing account: gcloud beta billing projects link "
            f"{proj} --billing-account <ACCOUNT_ID>",
        )

    # 5) Billing accounts (for user awareness)
    accounts = best_effort_billing_accounts()
    if accounts:
        acct_lines = "; ".join(f"{a.get('displayName','?')} ({a.get('name','')})" for a in accounts)
        status("Available billing accounts", True, acct_lines)
    else:
        status("Available billing accounts", False, "Run: gcloud beta billing accounts list")

    # 6) APIs enabled
    required_apis = {
        "aiplatform.googleapis.com": "Vertex AI API",
        "storage.googleapis.com": "Cloud Storage (for staging, sometimes required)",
    }
    code, out, err = run_cmd(["gcloud", "services", "list", "--enabled", "--format=value(config.name)"])
    enabled = set(out.splitlines()) if code == 0 else set()
    for api, label in required_apis.items():
        if api in enabled:
            status(f"{label}", True)
        else:
            status(
                f"{label}",
                False,
                f"Enable: gcloud services enable {api} --project {proj}",
            )

    print("\nIf everything above is green, you're ready to run the batch script.")


def list_images(input_dir: Path, recursive: bool = False) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".heic"}
    
    if recursive:
        results = []
        for root, dirs, files in os.walk(input_dir):
            # Prune 'backs' folders from traversal (case-insensitive)
            dirs[:] = [d for d in dirs if d.lower() != "backs"]
            
            for f in files:
                if f.startswith("._"):
                    continue
                p = Path(root) / f
                if p.suffix.lower() in exts:
                    results.append(p)
        return sorted(results)
    
    # Non-recursive
    return sorted([p for p in input_dir.iterdir() if p.suffix.lower() in exts and p.is_file() and not p.name.startswith("._")])


def find_orientation_flags(files: List[Path]) -> List[Tuple[Path, int]]:
    """
    Returns images whose EXIF Orientation tag is present and not 1 (normal).
    Google ignores these flags; users should rotate pixels manually.
    """
    flagged: List[Tuple[Path, int]] = []
    for p in files:
        try:
            with Image.open(p) as img:
                exif = img.getexif()
                if not exif:
                    continue
                orientation = exif.get(274)  # EXIF Orientation tag
                if orientation and orientation != 1:
                    flagged.append((p, int(orientation)))
        except Exception:
            # Ignore unreadable images here; processing step will surface errors.
            continue
    return flagged


def ensure_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def build_payload(prompt: str, encoded_image: str, mime: str, image_size: str = "4K") -> dict:
    return {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt},
                    {"inlineData": {"mimeType": mime, "data": encoded_image}},
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "candidateCount": 1,
            "responseModalities": ["IMAGE"],
            "imageConfig": {"imageSize": image_size},
        },
    }


def read_image_b64(path: Path) -> Tuple[str, str, str]:
    """
    Reads image. 
    Always returns the SHA256 hash of the ORIGINAL file on disk.
    If conversion is needed (non-JPG or rotation), returns base64 of the converted JPEG.
    Otherwise returns base64 of the original bytes.
    """
    # 1. Read original bytes and compute hash
    with path.open("rb") as f:
        original_data = f.read()
    original_hash = hashlib.sha256(original_data).hexdigest()

    needs_conversion = False
    
    # 2. Check extension
    if path.suffix.lower() not in {".jpg", ".jpeg"}:
        needs_conversion = True
    
    # 3. Check EXIF rotation (if it looks like a JPG)
    if not needs_conversion:
        try:
            # Use BytesIO to avoid re-opening file from disk
            with Image.open(io.BytesIO(original_data)) as img:
                exif = img.getexif()
                if exif and exif.get(274, 1) != 1:
                    needs_conversion = True
        except Exception:
            pass

    if needs_conversion:
        # Convert to high-quality upright JPEG
        try:
            with Image.open(io.BytesIO(original_data)) as img:
                img = ImageOps.exif_transpose(img)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                
                bio = io.BytesIO()
                img.save(bio, format="JPEG", quality=100, subsampling=0)
                converted_data = bio.getvalue()
                encoded = base64.b64encode(converted_data).decode("utf-8")
                # Return ORIGINAL hash, but CONVERTED data for API
                return encoded, "image/jpeg", original_hash
        except Exception as e:
             # Fallback for HEIC on macOS using 'sips' if PIL fails
            if path.suffix.lower() == ".heic" and sys.platform == "darwin":
                try:
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                        tmp_path = tmp.name
                    
                    cmd = ["sips", "-s", "format", "jpeg", "-s", "formatOptions", "100", str(path), "--out", tmp_path]
                    subprocess.run(cmd, check=True, capture_output=True)
                    
                    # Open the converted JPEG with PIL to handle rotation (exif_transpose)
                    with Image.open(tmp_path) as img:
                        img = ImageOps.exif_transpose(img)
                        if img.mode != "RGB":
                            img = img.convert("RGB")
                        
                        bio = io.BytesIO()
                        img.save(bio, format="JPEG", quality=100, subsampling=0)
                        converted_data = bio.getvalue()
                    
                    # Clean up temp file
                    os.unlink(tmp_path)
                    
                    encoded = base64.b64encode(converted_data).decode("utf-8")
                    return encoded, "image/jpeg", original_hash
                except Exception as sips_error:
                    print(f"sips conversion failed: {sips_error}", file=sys.stderr)
            
            raise e

    # No conversion needed
    encoded = base64.b64encode(original_data).decode("utf-8")
    return encoded, "image/jpeg", original_hash


# --- CORE PROCESSING ---
def process_one_image(
    index: int,
    total: int,
    image_path: Path,
    output_path: Path,
    endpoint: str,
    token: str,
    prompt: str,
    timeout: int,
    log: Callable[[str], None],
    dry_run: bool = False,
    progress_callback: Optional[Callable[[dict], None]] = None,
    db_row_id: Optional[int] = None,
    image_size: str = "4K",
) -> Tuple[bool, bool]:
    """Returns (success, blocked). blocked=True when the API refuses content."""
    max_retries = 10
    encoded, mime, input_hash = read_image_b64(image_path)
    payload = build_payload(prompt, encoded, mime, image_size)
    start_time = time.time()

    if dry_run:
        # For testing pipeline without hitting the API, just copy the input bytes.
        output_bytes = base64.b64decode(encoded)
        output_path.write_bytes(output_bytes)
        output_hash = hashlib.sha256(output_bytes).hexdigest()
        duration = time.time() - start_time
        if db_row_id:
            log_image_end(db_row_id, "success", "dry-run", input_hash, output_hash, duration)
        log("      ✨ DRY-RUN (copied input to output)")
        return True, False

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json; charset=utf-8",
    }

    for attempt in range(max_retries):
        if attempt:
            log(f"      ➤ Retrying (attempt {attempt + 1}/{max_retries})...")
            time.sleep(5 if attempt < 3 else 15)

        try:
            resp = requests.post(
                endpoint, headers=headers, json=payload, timeout=timeout, stream=False
            )
        except requests.RequestException as e:
            log(f"      ❌ Network error: {e}")
            continue

        if resp.status_code == 401:
            log("      🔐 Auth expired.")
            return False, False  # signal caller to refresh token

        if resp.status_code == 429:
            log("      ⏳ Rate limited. Sleeping 60s...")
            time.sleep(60)
            continue

        if resp.status_code >= 400:
            msg = f"API error {resp.status_code}: {resp.text[:200]}"
            log(f"      ❌ {msg}")
            if db_row_id:
                duration = time.time() - start_time
                log_image_end(db_row_id, "error", msg, input_hash, "", duration)
            return False, False

        try:
            payload_json = resp.json()
        except json.JSONDecodeError:
            log("      ❌ Response not JSON.")
            continue

        candidate = payload_json.get("candidates", [{}])[0]
        parts = candidate.get("content", {}).get("parts", [])
        for part in parts:
            inline = part.get("inlineData")
            if inline and inline.get("data"):
                output_bytes = base64.b64decode(inline["data"])
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(output_bytes)
                
                output_hash = hashlib.sha256(output_bytes).hexdigest()
                duration = time.time() - start_time
                if db_row_id:
                    log_image_end(db_row_id, "success", "", input_hash, output_hash, duration)
                log("      ✨ SUCCESS")
                return True, False

        finish = candidate.get("finishReason") or candidate.get("finish_reason")
        safety = candidate.get("safetyRatings") or candidate.get("safety_ratings")
        prompt_fb = payload_json.get("promptFeedback") or payload_json.get("prompt_feedback")
        log(
            "      ⚠️ Empty content."
            + (f" finishReason={finish}" if finish else "")
            + (f" safety={safety}" if safety else "")
            + (f" promptFeedback={prompt_fb}" if prompt_fb else "")
        )
        # Avoid dumping huge payloads, but print a short snippet for debugging.
        log(f"      ℹ️ Raw response (truncated): {str(payload_json)[:300]}")
        blocked = str(finish).upper() in {
            "SAFETY",
            "SAFETY_FILTER",
            "CONTENT_FILTER",
            "IMAGE_PROHIBITED_CONTENT",
            "BLOCKED",
        }
        if blocked:
            duration = time.time() - start_time
            if db_row_id:
                log_image_end(
                    db_row_id,
                    "blocked",
                    f"finishReason={finish}",
                    input_hash,
                    "",
                    duration,
                )
            log("      ⛔ Content blocked by model; skipping further retries for this image.")
            return False, True
        # loop will retry

    return False, False


def run_normalization(config: BatchConfig, log: Callable[[str], None]) -> None:
    """
    Reads all images in input_dir, applies EXIF rotation, converts to RGB,
    and saves as high-quality JPG in output_dir.
    """
    ensure_output_dir(config.output_dir)
    files = list_images(config.input_dir, recursive=config.recursive)
    if not files:
        log("❌ No images found.")
        return

    log(f"🔄 Normalizing {len(files)} images to {config.output_dir}...")
    
    count = 0
    for p in files:
        try:
            # Maintain subfolder structure in output
            rel_path = p.relative_to(config.input_dir)
            out_path = config.output_dir / rel_path.with_suffix(".jpg")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            
            with Image.open(p) as img:
                # 1. Apply EXIF rotation
                img = ImageOps.exif_transpose(img)
                
                # 2. Convert to RGB (handles RGBA, P, etc.)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                
                # 3. Save as highest quality JPG
                # subsampling=0 ensures 4:4:4 chroma subsampling (best quality)
                # quality=100 is max standard JPEG quality
                img.save(out_path, "JPEG", quality=100, subsampling=0)
                
            count += 1
            if count % 10 == 0:
                log(f"   Processed {count}/{len(files)}...")
                
        except Exception as e:
            log(f"   ❌ Failed to normalize {p.name}: {e}")

    log(f"✅ Normalization complete. {count} images saved to {config.output_dir}")


def run_batch(
    config: BatchConfig,
    log: Callable[[str], None],
    progress_callback: Optional[Callable[[dict], None]] = None,
) -> None:
    ensure_output_dir(config.output_dir)
    files = list_images(config.input_dir, recursive=config.recursive)
    if not files:
        log("❌ No images found (jpg/jpeg/png).")
        return

    if config.check_orientation:
        flagged = find_orientation_flags(files)
        if flagged:
            log(f"⚠️ {len(flagged)} images have EXIF rotation flags; rotate pixels before sending:")
            for p, orient in flagged[:10]:
                log(f"   - {p.name} (orientation={orient})")
            if len(flagged) > 10:
                log(f"   ...and {len(flagged) - 10} more")
        else:
            log("✅ Orientation check: no rotation flags found.")

    # Stats
    already_done = 0
    pending_files = []
    
    # We must read files to get the hash for the filename.
    # To avoid reading ALL files just to check existence, we can check if a file matching the pattern exists?
    # Pattern: PRO_4K_[Hash]_[Stem].png. We don't know [Hash] without reading.
    # Compromise: We will use a glob pattern to check for existing files for that stem?
    # No, to be 100% correct we need the hash.
    # Let's iterate and compute hash on the fly for the "planned" files, 
    # but for the initial "already_done" stats, maybe just check if ANY file exists for that stem?
    # This might be slightly inaccurate if you have collisions, but much faster.
    
    # Actually, the user wants the hash in the filename.
    # Let's do this: Iterate files. For each file, compute hash ONLY if we are about to process it?
    # But we need to know if it's done.
    # Strategy: Check for `PRO_4K_*_{p.stem}.png` in the output dir.
    
    log(f"🔍 Scanning {len(files)} files...")
    
    for p in files:
        rel_path = p.relative_to(config.input_dir)
        output_parent = config.output_dir / rel_path.parent
        
        # Check if we have an existing output for this file stem
        # We assume the format is PRO_4K_{ShortHash}_{Stem}.png
        # We search for any file ending in _{Stem}.png in the corresponding output folder
        
        is_done = False
        if output_parent.exists():
            # Optimization: List dir once per parent? 
            # For now, just globbing per file.
            candidates = list(output_parent.glob(f"PRO_4K_*_{p.stem}.png"))
            if candidates:
                # Assume if any exists, it's done.
                # If you have duplicates with same name but different content, this heuristic might fail 
                # without checking the full hash, but it's a reasonable trade-off for speed.
                if any(c.stat().st_size > 0 for c in candidates):
                    is_done = True

        if is_done:
            already_done += 1
            continue
            
        pending_files.append(p)

    pending = len(pending_files)
    if pending == 0:
        log(f"✅ All {already_done} images already processed. Nothing to do.")
        return

    start_idx = min(config.start_index, pending)
    planned_files = pending_files[start_idx:]
    if config.max_images is not None:
        planned_files = planned_files[: config.max_images]

    planned = len(planned_files)
    total_considered = len(files)
    est_cost = planned * COST_PER_IMAGE_USD

    log(f"📸 Found {total_considered} images. Already done: {already_done}. Pending: {pending}. Planned this run: {planned}.")
    log(f"💰 Est. cost @ ${COST_PER_IMAGE_USD:.2f}/image: ${est_cost:.2f}")
    log(f"📂 Saving to: {config.output_dir}")
    endpoint = f"https://aiplatform.googleapis.com/v1/projects/{config.project_id}/locations/{REGION}/publishers/google/models/{MODEL_ID}:generateContent"

    token = "DRY-RUN" if config.dry_run else get_access_token()
    processed = 0
    success_count = 0
    fail_count = 0
    total_duration = 0.0
    failed_items: List[Tuple[Path, Path]] = [] # Note: We won't know output path until loop

    for idx, image_path in enumerate(planned_files, start=1):
        log(f"   [Processing {idx}/{planned}] {image_path.name}...")
        
        # 1. Read and Hash (to determine output name)
        try:
            # We use read_image_b64 to get the hash and the encoded data
            # But wait, process_one_image calls read_image_b64 again.
            # Efficiency fix: We should probably let process_one_image determine the output path?
            # Or calculate hash here.
            with image_path.open("rb") as f:
                header_bytes = f.read(8192) # Read chunk for partial hash? No, user wants full identity.
                f.seek(0)
                file_hash = hashlib.sha256(f.read()).hexdigest()[:8] # Short hash
            
            rel_path = image_path.relative_to(config.input_dir)
            out_name = f"PRO_4K_{file_hash}_{image_path.stem}.png"
            output_path = config.output_dir / rel_path.parent / out_name
            
            # Double check existence with exact hash now
            if output_path.exists() and output_path.stat().st_size > 0:
                log(f"      ⏭️ Skipping (already exists with same hash).")
                processed += 1
                continue

        except Exception as e:
            log(f"      ❌ Failed to read/hash input: {e}")
            fail_count += 1
            continue

        db_row_id = log_image_start(image_path, output_path) if not config.dry_run else None

        if processed % config.refresh_every == 0 and processed != 0:
            token = get_access_token()

        success, blocked = process_one_image(
            idx,
            total_considered,
            image_path,
            output_path,
            endpoint,
            token,
            config.prompt,
            config.timeout,
            log,
            config.dry_run,
            progress_callback,
            db_row_id,
        )
        if not success and not config.dry_run and not blocked:
            # Attempt a token refresh once then retry once.
            token = get_access_token()
            success, blocked = process_one_image(
                idx,
                planned,
                image_path,
                output_path,
                endpoint,
                token,
                config.prompt,
                config.timeout,
                log,
                config.dry_run,
                progress_callback,
                db_row_id,
            )
            if not success:
                log("      ❌ Giving up on this image.")
                fail_count += 1
                if db_row_id:
                    log_image_end(db_row_id, "error", "unresolved error", "", "", 0.0)
            elif success:
                success_count += 1
        else:
            if success:
                success_count += 1
            else:
                fail_count += 1

        if db_row_id:
            conn = sqlite3.connect(DB_PATH)
            try:
                cur = conn.execute("SELECT duration_sec FROM runs WHERE id=?", (db_row_id,))
                row = cur.fetchone()
                if row and row[0]:
                    total_duration += row[0]
            finally:
                conn.close()

        processed += 1
        avg_time = (total_duration / success_count) if success_count else 0.0
        remaining = planned - success_count
        eta = remaining * avg_time

        if progress_callback:
            progress_callback(
                {
                    "processed": processed,
                    "success": success_count,
                    "fail": fail_count,
                    "avg_time_sec": avg_time,
                    "eta_sec": eta,
                    "planned": planned,
                    "pending": pending,
                    "total": total_considered,
                }
            )

    log("✅ Batch complete.")


    log("✅ Batch complete.")


# --- CLI INTERFACE ---
def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Restore photos with Gemini 3 Pro Image Preview.")
    parser.add_argument("--input", required=False, default="input_images", help="Input folder of images.")
    parser.add_argument("--output", required=False, default="output_images", help="Output folder.")
    parser.add_argument("--max-images", type=int, default=None, help="Limit how many images to process this run.")
    parser.add_argument("--start-index", type=int, default=0, help="Skip the first N images.")
    parser.add_argument("--prompt-file", type=str, help="Path to custom prompt text file.")
    parser.add_argument("--timeout", type=int, default=120, help="Per-request timeout seconds.")
    parser.add_argument("--dry-run", action="store_true", help="Do not call the API; copy input -> output to test flow.")
    parser.add_argument("--check-orientation", action="store_true", help="Warn if EXIF rotation flags are present.")
    parser.add_argument("--recursive", action="store_true", help="Scan input folder recursively for images.")
    parser.add_argument(
        "--setup-check",
        action="store_true",
        help="Run GCP readiness checks (auth, project, billing link, APIs) and exit.",
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Launch PyQt6 UI (requires PyQt6 installed). CLI flags ignored except --prompt-file.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def load_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file:
        return Path(args.prompt_file).read_text()
    return PROMPT


def cli_entry(argv: Optional[Iterable[str]] = None) -> None:
    init_db()
    args = parse_args(argv)
    prompt = load_prompt(args)

    if args.setup_check:
        run_setup_check()
        return

    if args.ui:
        if not PYQT_AVAILABLE:
            print("PyQt6 not installed. Install with `pip install PyQt6`.", file=sys.stderr)
            sys.exit(1)
        launch_ui(prompt)
        return

    if args.dry_run:
        project_id = "dry-run-project"
        active_account = None
    else:
        try:
            project_id = get_project_id()
        except RuntimeError as e:
            print(f"❌ {e}")
            sys.exit(1)
        active_account = get_active_account()

    config = BatchConfig(
        project_id=project_id,
        input_dir=Path(args.input).expanduser(),
        output_dir=Path(args.output).expanduser(),
        max_images=args.max_images,
        start_index=args.start_index,
        timeout=args.timeout,
        prompt=prompt,
        dry_run=args.dry_run,
        check_orientation=args.check_orientation,
        recursive=args.recursive,
    )

    if not args.dry_run:
        acct = active_account or "(unknown account — run `gcloud auth login`)"
        print(f"👤 Account: {acct}")
        print(f"🗂️ Project: {project_id}")

    if not args.dry_run:
        accounts = best_effort_billing_accounts()
        if accounts:
            print("💳 Billing accounts detected (Google does not expose remaining free-trial balance via API):")
            for acct in accounts:
                print(f"   - {acct.get('displayName','(no name)')} [{acct.get('name','')}] state={acct.get('open','')}")
        else:
            print("⚠️ Could not list billing accounts (ensure gcloud beta components are installed).")
    if not args.dry_run:
        print("🧪 DRY-RUN mode: no API calls will be made; inputs are copied to outputs.")

    run_batch(config, log=print)


# --- PYQT UI ---
if PYQT_AVAILABLE:  # pragma: no cover

    class Worker(QtCore.QThread):
        log_signal = QtCore.pyqtSignal(str)
        progress_signal = QtCore.pyqtSignal(dict)
        done_signal = QtCore.pyqtSignal()

        def __init__(self, config: BatchConfig):
            super().__init__()
            self.config = config

        def run(self) -> None:
            run_batch(self.config, log=self.log_signal.emit, progress_callback=self.progress_signal.emit)
            self.done_signal.emit()


    class SingleImageWorker(QtCore.QThread):
        log_signal = QtCore.pyqtSignal(str)
        result_signal = QtCore.pyqtSignal(str, str)
        done_signal = QtCore.pyqtSignal()

        def __init__(self, image_path: Path, output_path: Path, prompt: str, image_size: str, project_id: str):
            super().__init__()
            self.image_path = image_path
            self.output_path = output_path
            self.prompt = prompt
            self.image_size = image_size
            self.project_id = project_id

        def run(self) -> None:
            try:
                endpoint = f"https://aiplatform.googleapis.com/v1/projects/{self.project_id}/locations/{REGION}/publishers/google/models/{MODEL_ID}:generateContent"
                token = get_access_token()
                timeout = 120

                self.log_signal.emit(f"Processing {self.image_path.name}...")

                db_row_id = log_image_start(self.image_path, self.output_path)

                success, blocked = process_one_image(
                    index=0,
                    total=1,
                    image_path=self.image_path,
                    output_path=self.output_path,
                    endpoint=endpoint,
                    token=token,
                    prompt=self.prompt,
                    timeout=timeout,
                    log=self.log_signal.emit,
                    dry_run=False,
                    progress_callback=None,
                    db_row_id=db_row_id,
                    image_size=self.image_size
                )

                if success:
                    self.result_signal.emit(str(self.output_path), "")
                    self.log_signal.emit(f"Successfully processed: {self.output_path}")
                elif blocked:
                    error_msg = "Content blocked by model safety filters"
                    self.result_signal.emit("", error_msg)
                    self.log_signal.emit(f"Error: {error_msg}")
                else:
                    error_msg = "Processing failed (see logs for details)"
                    self.result_signal.emit("", error_msg)
                    self.log_signal.emit(f"Error: {error_msg}")

            except Exception as e:
                error_msg = str(e)
                self.result_signal.emit("", error_msg)
                self.log_signal.emit(f"Exception: {error_msg}")
                self.log_signal.emit(traceback.format_exc())

            self.done_signal.emit()


    class DragDropLabel(QtWidgets.QLabel):
        file_dropped = QtCore.pyqtSignal(str)

        def __init__(self, parent=None):
            super().__init__(parent)
            self.setAcceptDrops(True)

        def dragEnterEvent(self, event):
            if event.mimeData().hasUrls():
                urls = event.mimeData().urls()
                if urls and urls[0].isLocalFile():
                    file_path = urls[0].toLocalFile()
                    # Check if it's an image file
                    valid_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.heic', '.heif']
                    if any(file_path.lower().endswith(ext) for ext in valid_extensions):
                        event.acceptProposedAction()
                    else:
                        event.ignore()
                else:
                    event.ignore()
            else:
                event.ignore()

        def dropEvent(self, event):
            urls = event.mimeData().urls()
            if urls and urls[0].isLocalFile():
                file_path = urls[0].toLocalFile()
                self.file_dropped.emit(file_path)
                event.acceptProposedAction()


    class MainWindow(QtWidgets.QWidget):
        def __init__(self, prompt: str):
            super().__init__()
            self.prompt = prompt
            self.worker: Optional[Worker] = None
            self.project_id = None
            self.flagged_count = 0
            self.recursive = False
            self.prefs = load_prefs()
            self.init_ui()

        def init_ui(self) -> None:
            self.setWindowTitle("Gemini 4K Restorer")
            main_layout = QtWidgets.QVBoxLayout()

            # Create tab widget
            self.tabs = QtWidgets.QTabWidget()

            # Build batch processing tab
            batch_tab = self._build_batch_tab()
            single_tab = self._build_single_image_tab()

            self.tabs.addTab(batch_tab, "Batch Processing")
            self.tabs.addTab(single_tab, "Single Image")

            main_layout.addWidget(self.tabs)
            self.setLayout(main_layout)

            # Prefill paths after widgets exist
            if self.prefs.get("input_path"):
                self.input_edit.setText(self.prefs["input_path"])
            if self.prefs.get("output_path"):
                self.output_edit.setText(self.prefs["output_path"])
            self.populate_projects(log=False)

            # Keep stats reactive
            self.max_spin.valueChanged.connect(self.update_stats)
            self.start_spin.valueChanged.connect(self.update_stats)

            self.update_stats()
            # Auto-check account/billing shortly after launch (non-blocking enough for UI).
            QtCore.QTimer.singleShot(250, self.check_billing)

        def _build_batch_tab(self) -> QtWidgets.QWidget:
            batch_widget = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout()

            self.input_edit = QtWidgets.QLineEdit()
            self.output_edit = QtWidgets.QLineEdit()
            self.project_combo = QtWidgets.QComboBox()
            self.project_refresh_btn = QtWidgets.QPushButton("Refresh")
            self.project_set_btn = QtWidgets.QPushButton("Use Project")
            check_orientation_btn = QtWidgets.QPushButton("Check Orientation")
            self.max_spin = QtWidgets.QSpinBox()
            self.max_spin.setRange(0, 10000)
            self.max_spin.setValue(int(self.prefs.get("max_images", 50)))
            self.start_spin = QtWidgets.QSpinBox()
            self.start_spin.setRange(0, 100000)
            self.start_spin.setValue(int(self.prefs.get("start_index", 0)))

            self.recursive_check = QtWidgets.QCheckBox("Include subfolders")
            self.recursive_check.stateChanged.connect(self.update_stats)
            self.recursive_check.setChecked(bool(self.prefs.get("recursive", False)))

            browse_in = QtWidgets.QPushButton("Browse Input")
            browse_out = QtWidgets.QPushButton("Browse Output")
            browse_in.clicked.connect(self.pick_input)
            browse_out.clicked.connect(self.pick_output)
            check_orientation_btn.clicked.connect(self.check_orientation_now)
            self.project_refresh_btn.clicked.connect(lambda: self.populate_projects(log=True))
            self.project_set_btn.clicked.connect(self.switch_project)

            form = QtWidgets.QFormLayout()
            in_row = QtWidgets.QHBoxLayout()
            in_row.addWidget(self.input_edit)
            in_row.addWidget(browse_in)
            in_row.addWidget(check_orientation_btn)
            in_row_widget = QtWidgets.QWidget()
            in_row_widget.setLayout(in_row)
            form.addRow("Input folder", in_row_widget)
            form.addRow("Output folder", self._with_row(self.output_edit, browse_out))
            form.addRow("Max images (0 = all)", self.max_spin)
            form.addRow("Start index", self.start_spin)
            form.addRow("", self.recursive_check)
            project_row = QtWidgets.QHBoxLayout()
            project_row.addWidget(self.project_combo)
            project_row.addWidget(self.project_refresh_btn)
            project_row.addWidget(self.project_set_btn)
            project_row_widget = QtWidgets.QWidget()
            project_row_widget.setLayout(project_row)
            form.addRow("Project", project_row_widget)
            layout.addLayout(form)

            self.billing_label = QtWidgets.QLabel("Billing: unknown")
            check_billing_btn = QtWidgets.QPushButton("Check Billing")
            check_billing_btn.clicked.connect(self.check_billing)
            layout.addWidget(self.billing_label)
            layout.addWidget(check_billing_btn)

            self.account_label = QtWidgets.QLabel("Account/Project: unknown")
            layout.addWidget(self.account_label)

            self.orientation_label = QtWidgets.QLabel("Orientation: not checked")
            layout.addWidget(self.orientation_label)

            self.stats_label = QtWidgets.QLabel("Stats: pending N/A")
            self.cost_label = QtWidgets.QLabel("Cost: N/A")
            self.eta_label = QtWidgets.QLabel("ETA: N/A")
            self.avg_label = QtWidgets.QLabel("Avg/image: N/A")
            layout.addWidget(self.stats_label)
            layout.addWidget(self.cost_label)
            layout.addWidget(self.avg_label)
            layout.addWidget(self.eta_label)

            self.start_btn = QtWidgets.QPushButton("Start Batch")
            self.start_btn.clicked.connect(self.start_batch)

            layout.addWidget(self.start_btn)

            self.log_box = QtWidgets.QTextEdit()
            self.log_box.setReadOnly(True)
            layout.addWidget(self.log_box)

            batch_widget.setLayout(layout)
            return batch_widget

        def _build_single_image_tab(self) -> QtWidgets.QWidget:
            single_widget = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout()

            # Input Section
            input_group = QtWidgets.QGroupBox("Input Image")
            input_layout = QtWidgets.QVBoxLayout()

            select_row = QtWidgets.QHBoxLayout()
            self.single_select_btn = QtWidgets.QPushButton("Select Image")
            self.single_select_btn.clicked.connect(self.select_single_image)
            select_row.addWidget(self.single_select_btn)

            self.single_image_path_label = QtWidgets.QLabel("No image selected")
            select_row.addWidget(self.single_image_path_label)
            select_row.addStretch()
            input_layout.addLayout(select_row)

            self.single_image_preview = DragDropLabel()
            self.single_image_preview.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.single_image_preview.setMinimumHeight(300)
            self.single_image_preview.setStyleSheet("QLabel { background-color: #f0f0f0; border: 2px dashed #999; color: #666; } QLabel:hover { border-color: #4a90e2; }")
            self.single_image_preview.setText("Drop image here or click 'Select Image'")
            self.single_image_preview.file_dropped.connect(self.on_image_dropped)
            input_layout.addWidget(self.single_image_preview)
            input_group.setLayout(input_layout)
            layout.addWidget(input_group)

            # Prompt Section
            prompt_group = QtWidgets.QGroupBox("Custom Prompt")
            prompt_layout = QtWidgets.QVBoxLayout()
            self.single_prompt_edit = QtWidgets.QTextEdit()
            self.single_prompt_edit.setPlainText(PROMPT)
            self.single_prompt_edit.setMaximumHeight(150)
            self.single_prompt_edit.setStyleSheet("QTextEdit { background-color: #ffffff; color: #000000; }")
            prompt_layout.addWidget(self.single_prompt_edit)
            prompt_group.setLayout(prompt_layout)
            layout.addWidget(prompt_group)

            # Size and Output Section
            config_group = QtWidgets.QGroupBox("Output Configuration")
            config_layout = QtWidgets.QVBoxLayout()

            size_row = QtWidgets.QHBoxLayout()
            size_row.addWidget(QtWidgets.QLabel("Output Size:"))
            self.single_size_combo = QtWidgets.QComboBox()
            self.single_size_combo.addItems(["4K", "2K", "1K"])
            self.single_size_combo.setCurrentText("2K")
            size_row.addWidget(self.single_size_combo)
            size_row.addStretch()
            config_layout.addLayout(size_row)

            self.single_output_alongside = QtWidgets.QRadioButton("Save alongside input")
            self.single_output_alongside.setChecked(True)
            self.single_output_alongside.toggled.connect(self.toggle_output_location)
            config_layout.addWidget(self.single_output_alongside)

            self.single_output_custom = QtWidgets.QRadioButton("Custom output location:")
            config_layout.addWidget(self.single_output_custom)

            custom_row = QtWidgets.QHBoxLayout()
            custom_row.addSpacing(20)
            self.single_output_path_edit = QtWidgets.QLineEdit()
            self.single_output_path_edit.setEnabled(False)
            self.single_output_path_edit.setStyleSheet("QLineEdit { background-color: #ffffff; color: #000000; }")
            custom_row.addWidget(self.single_output_path_edit)
            self.single_output_browse_btn = QtWidgets.QPushButton("Browse")
            self.single_output_browse_btn.setEnabled(False)
            self.single_output_browse_btn.clicked.connect(self.pick_single_output)
            custom_row.addWidget(self.single_output_browse_btn)
            config_layout.addLayout(custom_row)

            suffix_row = QtWidgets.QHBoxLayout()
            suffix_row.addWidget(QtWidgets.QLabel("Filename suffix:"))
            self.single_filename_suffix = QtWidgets.QLineEdit()
            self.single_filename_suffix.setPlaceholderText("e.g., _restored")
            self.single_filename_suffix.setStyleSheet("QLineEdit { background-color: #ffffff; color: #000000; }")
            suffix_row.addWidget(self.single_filename_suffix)
            config_layout.addLayout(suffix_row)

            config_group.setLayout(config_layout)
            layout.addWidget(config_group)

            # Process Button and Progress
            self.single_process_btn = QtWidgets.QPushButton("Process Image")
            self.single_process_btn.setEnabled(False)
            self.single_process_btn.clicked.connect(self.process_single_image)
            layout.addWidget(self.single_process_btn)

            self.single_progress = QtWidgets.QProgressBar()
            self.single_progress.setRange(0, 0)
            self.single_progress.setVisible(False)
            layout.addWidget(self.single_progress)

            # Log Section
            log_group = QtWidgets.QGroupBox("Processing Log")
            log_layout = QtWidgets.QVBoxLayout()
            self.single_log_box = QtWidgets.QTextEdit()
            self.single_log_box.setReadOnly(True)
            self.single_log_box.setMaximumHeight(150)
            self.single_log_box.setStyleSheet("QTextEdit { background-color: #ffffff; color: #000000; font-family: monospace; }")
            log_layout.addWidget(self.single_log_box)
            log_group.setLayout(log_layout)
            layout.addWidget(log_group)

            # Result Section
            result_group = QtWidgets.QGroupBox("Result")
            result_layout = QtWidgets.QVBoxLayout()

            self.single_result_label = QtWidgets.QLabel("No result yet")
            result_layout.addWidget(self.single_result_label)

            self.single_result_preview = QtWidgets.QLabel()
            self.single_result_preview.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.single_result_preview.setMinimumHeight(300)
            self.single_result_preview.setStyleSheet("QLabel { background-color: #f0f0f0; border: 1px solid #ccc; }")
            result_layout.addWidget(self.single_result_preview)

            result_group.setLayout(result_layout)
            layout.addWidget(result_group)

            layout.addStretch()
            single_widget.setLayout(layout)

            # Initialize instance variables for single image processing
            self.single_image_path = None
            self.single_worker = None

            return single_widget

        def _with_row(self, widget: QtWidgets.QWidget, button: QtWidgets.QWidget) -> QtWidgets.QWidget:
            row = QtWidgets.QHBoxLayout()
            row.addWidget(widget)
            row.addWidget(button)
            container = QtWidgets.QWidget()
            container.setLayout(row)
            return container

        def append_log(self, text: str) -> None:
            self.log_box.append(text)

        def populate_projects(self, log: bool = False) -> None:
            projects = list_projects()
            self.project_combo.clear()
            if not projects:
                self.project_combo.addItem("(no projects found)")
                if log:
                    self.append_log("No projects found; run `gcloud projects list` or create one in the console.")
                return
            self.project_combo.addItems(projects)
            preferred = self.prefs.get("project_id")
            if preferred and preferred in projects:
                self.project_combo.setCurrentText(preferred)
            elif log:
                self.append_log(f"Projects available: {', '.join(projects)}")

        def switch_project(self) -> None:
            proj = self.project_combo.currentText().strip()
            if not proj or "(no projects found)" in proj:
                self.append_log("No project selected. Use the dropdown first.")
                return
            code, out, err = run_cmd(["gcloud", "config", "set", "project", proj])
            if code == 0:
                self.append_log(f"Set project to: {proj}")
                self.prefs["project_id"] = proj
                self.save_prefs()
                # Refresh account/project labels after switch
                self.check_billing()
            else:
                self.append_log(f"Failed to set project: {err or out or '(no output)'}")

        def pick_input(self) -> None:
            path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select input folder")
            if path:
                self.input_edit.setText(path)
                self.run_orientation_check(Path(path))
                self.update_stats()
                self.save_prefs()

        def pick_output(self) -> None:
            path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select output folder")
            if path:
                self.output_edit.setText(path)
                self.update_stats()
                self.save_prefs()

        def run_orientation_check(self, folder: Path) -> None:
            files = list_images(folder, recursive=self.recursive_check.isChecked())
            flagged = find_orientation_flags(files)
            self.flagged_count = len(flagged)
            if flagged:
                preview = "; ".join(f"{p.name}(o={o})" for p, o in flagged[:5])
                more = f" ...+{len(flagged)-5}" if len(flagged) > 5 else ""
                self.orientation_label.setText(f"Orientation: {len(flagged)} need rotation: {preview}{more}")
                self.append_log(f"⚠️ {len(flagged)} images have EXIF rotation flags; rotate pixels before processing.")
            else:
                self.orientation_label.setText("Orientation: OK (no rotation flags)")
                self.append_log("Orientation check passed (no flags).")
            self.save_prefs()

        def check_orientation_now(self) -> None:
            folder = Path(self.input_edit.text()).expanduser()
            if not folder.exists():
                self.append_log("Input folder not found; set it first.")
                return
            self.run_orientation_check(folder)

        def update_stats(self) -> None:
            input_dir = Path(self.input_edit.text()).expanduser()
            output_dir = Path(self.output_edit.text()).expanduser()
            if not input_dir.exists() or not output_dir.exists():
                self.stats_label.setText("Stats: pending N/A")
                self.cost_label.setText("Cost: N/A")
                return
            files = list_images(input_dir, recursive=self.recursive_check.isChecked())
            total = len(files)
            existing_count = 0
            for p in files:
                rel_path = p.relative_to(input_dir)
                check_path = output_dir / rel_path.with_name(f"PRO_4K_{p.stem}.png")
                if check_path.exists() and check_path.stat().st_size > 0:
                    existing_count += 1
            
            already = existing_count
            pending = max(0, total - already)
            start_idx = min(self.start_spin.value(), pending)
            planned = pending - start_idx
            max_images = self.max_spin.value()
            if max_images > 0:
                planned = min(planned, max_images)
            est_cost = planned * COST_PER_IMAGE_USD
            self.stats_label.setText(
                f"Stats: total {total}, done {already}, pending {pending}, planned this run {planned}"
            )
            self.cost_label.setText(f"Cost: ~${est_cost:.2f} @ ${COST_PER_IMAGE_USD:.2f}/image")
            self.avg_label.setText("Avg/image: N/A")
            self.eta_label.setText("ETA: N/A")
            self.save_prefs()

        def update_progress_labels(self, stats: dict) -> None:
            avg = stats.get("avg_time_sec", 0.0)
            eta = stats.get("eta_sec", 0.0)
            self.avg_label.setText(f"Avg/image: {avg:.1f}s")
            if eta > 3600:
                eta_str = f"{eta/3600:.1f} h"
            elif eta > 60:
                eta_str = f"{eta/60:.1f} min"
            else:
                eta_str = f"{eta:.0f} s"
            self.eta_label.setText(f"ETA: ~{eta_str}")
            self.stats_label.setText(
                f"Stats: total {stats.get('total')}, success {stats.get('success')}, fail {stats.get('fail')}, planned {stats.get('planned')}"
            )

        def check_billing(self) -> None:
            # Keep this resilient: missing gcloud/project used to crash the app.
            try:
                accounts = best_effort_billing_accounts()
            except Exception as e:
                accounts = []
                self.append_log(f"Billing lookup failed: {e}")

            if accounts:
                text = "; ".join(
                    f"{a.get('displayName','(no name)')} ({a.get('name','')}) state={a.get('open','')}"
                    for a in accounts
                )
                self.billing_label.setText(f"Billing: {text}")
            else:
                self.billing_label.setText(
                    "Billing: unavailable (install gcloud or open console.cloud.google.com/billing)"
                )

            # Refresh project list whenever billing is checked (gcloud auth may have changed).
            self.populate_projects(log=False)

            acct = get_active_account()
            try:
                proj = get_project_id()
            except Exception as e:
                proj = "(project not set)"
                self.append_log(f"Project lookup failed: {e}")
                available = list_projects()
                if available:
                    self.append_log(
                        "Projects available under this account: "
                        + ", ".join(available)
                        + ". Run `gcloud config set project <id>` to pick one."
                    )
                else:
                    self.append_log("No projects listed; run `gcloud projects list` or create one in the console.")

            self.account_label.setText(f"Account/Project: {acct or 'unknown'} / {proj}")
            if acct:
                self.append_log(f"Active account: {acct}")
            if "(project not set)" in proj:
                self.append_log("Tip: run `gcloud config set project <id>` to pick the project you want.")
            if not acct:
                self.append_log("Tip: run `gcloud auth login` to choose the account you want.")

        def start_batch(self) -> None:
            dry = False
            try:
                project_id = get_project_id()
            except RuntimeError as e:
                candidate = self.project_combo.currentText().strip()
                if candidate and "(no projects found)" not in candidate:
                    self.append_log(f"Project lookup failed ({e}); trying to set '{candidate}'...")
                    code, out, err = run_cmd(["gcloud", "config", "set", "project", candidate])
                    if code == 0:
                        project_id = candidate
                        dry = False
                        self.append_log(f"Project set to {candidate}")
                    else:
                        project_id = "dry-run-project"
                        dry = True
                        self.append_log(f"Failed to set project: {err or out or '(no output)'}")
                else:
                    project_id = "dry-run-project"
                    dry = True
                    self.append_log(f"Using DRY-RUN because project lookup failed: {e}")
                available = list_projects()
                if available:
                    self.append_log(
                        "Projects available under this account: "
                        + ", ".join(available)
                        + ". Run `gcloud config set project <id>` to pick one."
                    )

            input_dir = Path(self.input_edit.text()).expanduser()
            output_dir = Path(self.output_edit.text()).expanduser()
            max_images = self.max_spin.value() or None
            start_index = self.start_spin.value()

            cfg = BatchConfig(
                project_id=project_id,
                input_dir=input_dir,
                output_dir=output_dir,
                max_images=max_images,
                start_index=start_index,
                prompt=self.prompt,
                dry_run=dry,
                check_orientation=True,
                recursive=self.recursive_check.isChecked(),
            )

            self.start_btn.setEnabled(False)
            self.worker = Worker(cfg)
            self.worker.log_signal.connect(self.append_log)
            self.worker.done_signal.connect(self.batch_done)
            self.worker.progress_signal.connect(self.update_progress_labels)
            acct = get_active_account() or "(unknown)"
            self.append_log(f"Using account: {acct} | project: {project_id} | dry-run={dry}")
            self.worker.start()

        def batch_done(self) -> None:
            self.start_btn.setEnabled(True)
            self.append_log("Done.")

        def select_single_image(self) -> None:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                "Select Image",
                "",
                "Images (*.png *.jpg *.jpeg *.bmp *.gif *.heic *.heif);;All Files (*)"
            )
            if path:
                self.single_image_path = Path(path)
                self.single_image_path_label.setText(path)

                # Load and display preview
                try:
                    pixmap = QtGui.QPixmap(path)
                    if pixmap.isNull():
                        self.single_image_path_label.setText(f"{path} (preview not available)")
                    else:
                        scaled = pixmap.scaledToWidth(400, QtCore.Qt.TransformationMode.SmoothTransformation)
                        self.single_image_preview.setPixmap(scaled)
                except Exception as e:
                    self.single_image_path_label.setText(f"{path} (error loading preview: {e})")

                # Enable process button
                self.single_process_btn.setEnabled(True)

        def on_image_dropped(self, path: str) -> None:
            self.single_image_path = Path(path)
            self.single_image_path_label.setText(path)

            # Load and display preview
            try:
                pixmap = QtGui.QPixmap(path)
                if pixmap.isNull():
                    self.single_image_path_label.setText(f"{path} (preview not available)")
                else:
                    scaled = pixmap.scaledToWidth(400, QtCore.Qt.TransformationMode.SmoothTransformation)
                    self.single_image_preview.setPixmap(scaled)
            except Exception as e:
                self.single_image_path_label.setText(f"{path} (error loading preview: {e})")

            # Enable process button
            self.single_process_btn.setEnabled(True)

        def toggle_output_location(self, checked: bool) -> None:
            if checked:
                self.single_output_path_edit.setEnabled(False)
                self.single_output_browse_btn.setEnabled(False)
            else:
                self.single_output_path_edit.setEnabled(True)
                self.single_output_browse_btn.setEnabled(True)

        def pick_single_output(self) -> None:
            path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select output folder")
            if path:
                self.single_output_path_edit.setText(path)

        def process_single_image(self) -> None:
            if not self.single_image_path:
                QtWidgets.QMessageBox.warning(self, "No Image", "Please select an image first.")
                return

            # Determine output path
            if self.single_output_alongside.isChecked():
                output_dir = self.single_image_path.parent
            else:
                custom_path = self.single_output_path_edit.text().strip()
                if not custom_path:
                    QtWidgets.QMessageBox.warning(self, "No Output", "Please select an output location.")
                    return
                output_dir = Path(custom_path)
                if not output_dir.exists():
                    output_dir.mkdir(parents=True, exist_ok=True)

            # Determine output filename
            suffix = self.single_filename_suffix.text().strip()
            size_str = self.single_size_combo.currentText()

            if suffix:
                output_filename = f"{self.single_image_path.stem}{suffix}.png"
            else:
                output_filename = f"PRO_{size_str}_{self.single_image_path.stem}.png"

            output_path = output_dir / output_filename

            # Get custom prompt and size
            custom_prompt = self.single_prompt_edit.toPlainText().strip()
            if not custom_prompt:
                custom_prompt = PROMPT

            # Start processing
            self.single_process_btn.setEnabled(False)
            self.single_progress.setVisible(True)
            self.single_result_label.setText("Processing...")
            self.single_log_box.clear()

            try:
                project_id = get_project_id()
            except RuntimeError as e:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Project Error",
                    f"Failed to get project ID: {e}\nPlease set a project in the Batch Processing tab."
                )
                self.single_process_btn.setEnabled(True)
                self.single_progress.setVisible(False)
                return

            self.single_worker = SingleImageWorker(
                image_path=self.single_image_path,
                output_path=output_path,
                prompt=custom_prompt,
                image_size=size_str,
                project_id=project_id
            )
            self.single_worker.log_signal.connect(self.append_single_log)
            self.single_worker.result_signal.connect(self.on_single_image_result)
            self.single_worker.done_signal.connect(self.on_single_image_done)
            self.single_worker.start()

        def append_single_log(self, text: str) -> None:
            self.single_log_box.append(text)

        def on_single_image_result(self, output_path: str, error: str) -> None:
            if error:
                self.single_result_label.setText(f"Error: {error}")
                QtWidgets.QMessageBox.critical(self, "Processing Error", f"Failed to process image:\n{error}")
            else:
                self.single_result_label.setText(f"Success! Saved to: {output_path}")

                # Load and display result preview
                try:
                    pixmap = QtGui.QPixmap(output_path)
                    if not pixmap.isNull():
                        scaled = pixmap.scaledToWidth(400, QtCore.Qt.TransformationMode.SmoothTransformation)
                        self.single_result_preview.setPixmap(scaled)
                except Exception as e:
                    self.single_result_label.setText(f"{output_path} (error loading preview: {e})")

        def on_single_image_done(self) -> None:
            self.single_process_btn.setEnabled(True)
            self.single_progress.setVisible(False)

        def save_prefs(self) -> None:
            data = {
                "input_path": self.input_edit.text(),
                "output_path": self.output_edit.text(),
                "max_images": self.max_spin.value(),
                "start_index": self.start_spin.value(),
                "recursive": self.recursive_check.isChecked(),
                "project_id": self.project_combo.currentText().strip(),
            }
            save_prefs(data)


def launch_ui(prompt: str) -> None:  # pragma: no cover
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow(prompt)
    window.resize(720, 520)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    cli_entry()
