# PostgreSQL GPT-2 Visualization Assets

This folder contains documentation and generator scripts that explain how the
GPT-2 model is represented inside PostgreSQL by the `pg_llm` extension.

Generated assets are produced by `scripts/generate_visualizations.py` and
include:

- `training_metrics.png` &mdash; a chart that plots loss and gradient norms across
  epochs using synthetic data to mirror the metrics logged in `llm_tape`.
- `gradient_flow.gif` &mdash; an animation that highlights how activations and
  gradients travel through the core tables (`llm_param`, `llm_tape`,
  `llm_clip`).
- `gpt2_pg_overview.md` &mdash; a narrative walkthrough of the database entities
  and how training proceeds.

The generated images are intentionally not checked into version control to keep
the repository lightweight. Run the script below whenever you need fresh
copies.

To regenerate the images, run:

```bash
python scripts/generate_visualizations.py
```

The script emits assets into this directory by default. You can change the
output location with `--output-dir`.
