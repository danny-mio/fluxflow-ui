# FluxFlow UI (package)

Python package for the FluxFlow web interface. This file documents the
**package layout** — for installation, user-facing usage, and CFG / generation
walkthroughs, see the repository root:

- Top-level **[README.md](../../README.md)** — install, quick start, CFG overview.
- **[docs/USER_GUIDE.md](../../docs/USER_GUIDE.md)** — full user guide.
- **[docs/TROUBLESHOOTING.md](../../docs/TROUBLESHOOTING.md)** — fixes for
  common issues.
- **[../../CHANGELOG.md](../../CHANGELOG.md)** — release history.

## Entry points (summary)

This is a pointer summary; the canonical install / launch instructions live
in the top-level [README.md](../../README.md). Do not duplicate command
details here.

- **Flask (primary)** — `fluxflow-ui` (installed console script) or
  `python -m fluxflow_ui.app_flask`.
- **Gradio (alternative)** — `python -m fluxflow_ui.app`.

## Package layout

```
src/fluxflow_ui/
├── app.py            # Gradio entry point (alternative)
├── app_flask.py      # Flask entry point (primary, REST API)
├── tabs/             # UI tab implementations
│   ├── generation.py
│   └── training.py
├── utils/            # Shared workers and config
│   ├── config_manager.py
│   ├── generation_worker.py
│   └── training_runner.py
├── templates/        # Flask HTML templates
└── static/           # CSS / JS assets
```

## v0.10.0 note

Since v0.10.0 the generation worker unpacks per-token text embeddings
(`text_seq`, `text_mask`) from `BertTextEncoder` and uses
`fluxflow.utils.build_cfg_null_pair` for the CFG null branch. The change is
internal — the REST API and request payloads are unchanged. See the top-level
[README.md](../../README.md) and
[fluxflow-core MIGRATION-v0.10.0-redesign.md](https://github.com/danny-mio/fluxflow-core/blob/develop/docs/MIGRATION-v0.10.0-redesign.md)
for context.

## Configuration

User configs are persisted as JSON under `.ui_configs/` (gitignored) and managed
by `utils/config_manager.py`. See the user guide for field-by-field details.

## License

MIT — see [LICENSE](../../LICENSE) at the repository root.
