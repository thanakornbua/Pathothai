# Publication package

This folder contains curated, publication-ready materials for the ROI-supervised HER2 classification pipeline.

Contents
- figures/: final plots and overlays referenced in the manuscript or appendix.
- examples/: lightweight samples (e.g., 2–3 tiles, XML snippet) for documentation.
- manifest.json: machine-readable list of included assets and their provenance.
- export_notebook.py: script to export a clean notebook to HTML/PDF and copy selected figures.

How to update
1. Run the notebook to generate results; ensure wandb has captured artifacts.
2. Export figures (confusion matrix, ROC, Grad-CAM overlays) to publication/figures.
3. Update manifest.json with any new assets (path, checksum, description).
4. Use export_notebook.py to produce a clean HTML/PDF snapshot.

Notes
- Do not place sensitive data here. Use anonymized or synthetic examples.
- Keep assets minimal and reproducible; prefer references to wandb artifacts when large.
