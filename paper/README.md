# Technical report

The report source is `technical-report.tex`. Its diagrams are deterministic
TikZ or data-derived plots; no generated raster illustration is required.

Build the evaluation figure from the saved result artifacts:

```powershell
uv run python scripts/plot_technical_report_results.py `
  --results-root 'C:\Projects\GNN-Traffic-Light-Optimization-Results' `
  --output 'paper\figures\evaluation-summary.pdf'
```

Build the appendix graph figure from the data embedded in the interactive HTML:

```powershell
uv run python scripts\plot_movement_graph_examples.py `
  --grid-report docs\assets\movement-graph-3x3.html `
  --output-dir docs\assets `
  --paper-output paper\figures\movement-graph-3x3.pdf
```

Compile from WSL with a TeX Live installation:

```powershell
wsl.exe -e bash -lc "cd /mnt/c/Projects/GNN-Traffic-Light-Optimization/paper && latexmk -pdf -interaction=nonstopmode -halt-on-error technical-report.tex"
```
