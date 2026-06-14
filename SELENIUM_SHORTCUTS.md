# Selenium MCP Shortcuts — fluxflow-ui

Living cookbook of validated Selenium MCP flows for the FluxFlow UI (Flask, port
7860). Re-skim this file **before** any UI test session; **append** any reliable
new flow afterwards. Do not store credentials or sensitive data here. Always
**close the Selenium session** when the scenario is done.

## Prereqs

```bash
# Terminal A — launch the UI (canonical command from the README)
fluxflow-ui
# UI must be reachable at http://localhost:7860
```

The `./launch_ui.sh` / `./launch.sh` shell scripts in the repo root are
convenience wrappers for source checkouts (`launch_ui.sh` sources `.venv` and
runs the configured UI app; `launch.sh` activates the same console script
with host/port flags); use them only when you are working from a clone
without an installed package.

Default checkpoint location: any `*.safetensors` (legacy) or versioned directory
under a path the file browser is allowed to traverse. The allowlist is the CWD
unless overridden by `FLUXFLOW_BROWSE_ROOTS` (colon-separated absolute paths),
plus `_cache/` for cached models.

## SH-01 — Smoke: page loads, no console errors

Use when: verifying a build / config / dependency change hasn't broken the page.

1. `selenium.start_browser` (Chrome, headless OK).
2. `selenium.navigate` → `http://localhost:7860`.
3. `selenium.get_console_logs` → expect no `SEVERE` entries.
4. `selenium.find_element` by css `#training` (the default-active tab panel in
   `templates/index.html`; the `#generation` panel is hidden until the user
   clicks the generation tab, so do not assert against it for smoke).
5. `selenium.close_browser`.

## SH-02 — Generation happy path (M7 regression)

_Added 2026-06-14 for v0.10.0 (`m7-ui` milestone)._

Use when: verifying the **text → image** internal wiring (per-token text → flow
processor) after editing `generation_worker.py` or bumping `fluxflow-core`.

1. `selenium.start_browser`.
2. `selenium.navigate` → `http://localhost:7860`.
3. Click the **Generation Studio** tab (button id `tab-generation`).
4. In the checkpoint field, paste a path to a known-good v0.10.0
   `.safetensors` checkpoint file (directory paths are **not** currently
   supported — `/api/generation/inspect` calls `safetensors.torch.load_file`
   directly and will raise on a directory; fixing that requires a code change
   to `app_flask.py:inspect_model`). Click **Load Model** — the handler runs
   `/api/generation/inspect` (dimension auto-detect) then `/api/generation/load`
   in sequence. Wait for the `loadStatus` element to show
   `Model loaded (VAE=…, Feature=…)`.
5. Fill the prompt field, e.g. `a small red house at sunset`.
6. (Optional) tick **Enable CFG** (checkbox id `use_cfg`), leave the negative
   prompt empty — this exercises the new `build_cfg_null_pair` null path.
7. Click **Generate Image**.
8. Wait for the result image to render (PNG `data:image/png;base64,...` in
   `<img>`). Timeout 5 min on CPU/MPS.
9. `selenium.get_console_logs` → no `SEVERE` errors; HTTP `/api/generation/generate`
   returned status `success`.
10. `selenium.close_browser`.

**Pass criteria**: image visible AND status text is `Image generated successfully!`
AND no console SEVERE.

**Common failure modes after a flow-signature change**:
- `TypeError: forward() takes 4 positional arguments but 5 were given` →
  flow_processor still on a pre-v0.10.0 signature.
- `nan` in output / black image with CFG on → null path likely zeros-like; should
  use `build_cfg_null_pair`.
- `ValueError: too many values to unpack (expected 2)` → text_encoder returns
  a single tensor (pre-M3 core checkout).

## SH-03 — API-only smoke (no browser)

Use when: Selenium MCP is unavailable or you just need to prove the HTTP
contract still works.

```bash
# 1) inspect — detects vae_dim / feature_maps_dim from the checkpoint
curl -s -X POST http://localhost:7860/api/generation/inspect \
  -H 'Content-Type: application/json' \
  -d '{"checkpoint_path": "<absolute/path/to/checkpoint>"}'

# 2) load — pass the dims returned by /inspect (defaults are 64/64, so omit
#    them only if your checkpoint truly is 64/64).
curl -s -X POST http://localhost:7860/api/generation/load \
  -H 'Content-Type: application/json' \
  -d '{"checkpoint_path": "<absolute/path/to/checkpoint>", "vae_dim": 64, "feature_maps_dim": 64}'

# 3) generate
curl -s -X POST http://localhost:7860/api/generation/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "a small red house at sunset", "img_width": 256, "img_height": 256, "ddim_steps": 8}' \
  | python -c 'import json,sys; r=json.load(sys.stdin); print(r["status"], r["message"]); print("img_bytes:", len(r.get("image","")))'
```

Pass: `status == success` and a non-empty base64 image string.

## Maintenance

- Add a new SH-NN block when you discover a reliable, reusable flow.
- Mark obsolete entries `(DEPRECATED)` in the heading; remove on the next pass.
- Keep entries short — selectors, click order, one sentence on intent.
