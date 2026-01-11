# Gemini Photo Restorer

Restore and upscale vintage photos to 4K using Google Vertex AI **Gemini 3 Pro Image Preview**. Works as a CLI or a small PyQt6 GUI with live logs and resume-safe output.

## 1) One-time setup (Google Cloud)
- Create/choose a Google account and visit `https://console.cloud.google.com`.
- Click **Activate Free Trial** (Google currently grants \$300 credit; price is ~\$0.25 per 4K image).
- Create a project (e.g., `family-restore-project`).
- Enable APIs:
  - Search and enable **Vertex AI API**.
  - Open **Model Garden**, search **"Gemini 3 Pro Image"**, open the card, click **ENABLE**.
- Install gcloud SDK (macOS): `brew install --cask google-cloud-sdk` (or download installer).
- Authenticate and set the project:
  ```bash
  gcloud auth login
  gcloud config set project family-restore-project
  ```

## 2) Local prerequisites
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # PyQt6 is only needed for the GUI
```

## 3) CLI usage
```bash
python restore_batch.py --input /path/to/in --output /path/to/out --max-images 50
```
Flags:
- `--input` / `--output`: folders you want to use (defaults: `input_images`, `output_images`).
- `--max-images`: limit how many images to run this session (omit for all).
- `--start-index`: skip N images (0-based).
- `--prompt-file`: point to a custom prompt text file.
- `--timeout`: per-request timeout in seconds (default 120).
- `--dry-run`: skip API calls; copy each input file to output to test the pipeline safely.
- `--check-orientation`: scan EXIF orientation flags and warn; Google ignores these flags, so rotate pixels first.
- `--recursive`: scan input folders recursively.
- `--setup-check`: run GCP readiness checks (gcloud installed, auth, project, billing link, required APIs) and exit.
- `--ui`: launch the PyQt UI instead of the CLI.

Resume logic: files named `PRO_4K_<original>.png` that already exist with non-zero size are skipped.

Billing note: Google does not expose free-trial remaining balance via API. The script prints any billing accounts it can see; check actual credit at `console.cloud.google.com/billing`.

## 4) PyQt6 UI
```bash
python restore_batch.py --ui
```
- Select input/output folders, choose a max image count, and click **Start Batch**.
- **Check Billing** shows detected billing accounts (balance still requires the console).
- Log window streams the same messages as the CLI.
- When you pick an input folder, the UI scans for EXIF rotation flags (orientation ≠ 1) and warns which files need manual rotation; Google ignores these flags.
- UI includes “Include subfolders” plus live stats (total/done/pending/planned) and estimated cost at $0.25/image based on Max Images and Start Index.
- Launch scripts respect `RESTORER_USE_SYSTEM_PY=1` to skip creating/using the bundled venv (useful on shared machines).

## 5) Best practices
- Manually rotate all images upright before running; the model will not fix orientation.
- Keep input in a flat folder (no subfolders).
- Run a small batch first (e.g., `--max-images 20`) and review results/costs the next day.
- Cost is roughly \$0.25 per 4K image with Gemini 3 Pro Image Preview; your \$300 trial covers ~1200 images.

## 6) Troubleshooting
- `Auth token fetch failed`: run `gcloud auth login` again and ensure the project is selected.
- `404 Not Found`: likely the model is not enabled in Model Garden.
- Rate limits (`429`): the script backs off automatically; re-run to resume if interrupted.
- Setup readiness: `python restore_batch.py --setup-check` for a quick pass/fail list with fix commands.
