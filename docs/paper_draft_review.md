# Actionable review of `docs/paper_draft.md`

This audit compared the integrated draft with `docs/paper_plan.md`, maintained documentation, current serialization and evaluation code, `docs/results/city_structure_statistics.csv`, the iteration-85 evaluation export and console log, and the current figures. The draft is technically coherent overall. The numerical result table, structural table, 29/4 feature dimensions, and maximal-clique explanation agree with their repository sources. The following items remain before the draft can be called source-checked.

## High priority

1. **The training/evaluation graphic contradicts the endpoint protocol** (`§7.2`, Figure 4, around lines 369–373). The raster itself ends in a clipboard labelled **“Best checkpoint.”** A caption cannot reliably override text embedded in a figure, and the image can therefore make iteration 85 look selected from periodic Freiburg evaluation. Crop/remove that final clipboard or regenerate the graphic with “final reported checkpoint (iteration 85).” Also state that its city maps and intersection icons are schematic, not renderings of the saved OSM networks.

2. **“Manually stopped” is not established by the archived run files** (`§7.2`, line 369; Appendix C, line 600). The YAML and metadata prove a generic target of 1,000; the console proves completion of iteration 85, its evaluation, and no later recorded iteration. The manual-stop cause comes from the author’s clarification, not from those files. Preserve the fact but make the provenance explicit, for example: “The author stopped the generic 1,000-iteration launch after iteration 85; the archive contains the completed iteration-85 update/evaluation and no later update.” Do not say the archive itself proves the stop mechanism.

3. **Related work is still intentionally unverified** (`§2`, Table 1, References). All bracketed placeholders remain publication blockers: Devailly year, TransferLight status/method details, several Table 1 cells, SUMO pages, and incomplete author/page/DOI records. The opening banner’s “complete first draft” is acceptable as drafting status only, but “integrated and checked” should exclude §2 explicitly until primary-source verification is complete.

4. **Figure B1’s caption omits the lowest-completion city** (Appendix B, line 588). It says “Freiburg and Mannheim remain relatively low-completion scenarios,” but Stuttgart is lower at roughly 48% for all policies. Replace with “Stuttgart, Mannheim, and Freiburg…” or describe all sub-60% cases. The plot also has overlapping `St` labels and an apparent stray/occluded Karlsruhe label near the right edge; regenerate with collision-aware labels or remove repeated city labels in favor of a compact legend/direct annotation scheme.

## Medium priority

5. **Figure 4 should not be framed as necessary evidence for checkpoint selection.** Its current caption devotes most of its text to undoing the figure’s misleading endpoint. Once corrected, caption it simply as the rollout/evaluation protocol: four rollout cities, 40 jobs/update, and Freiburg periodic evaluation with zero PPO jobs.

6. **The GUI greedy-inference aside is unnecessary and distracts from the frozen protocol** (`§4.6`, after the worked mask example). The user explicitly does not want a sampled-versus-greedy comparison in this paper. Remove “The interactive GUI runner…” unless the paper later discusses deployment modes; the evaluation section already states sampled inference precisely.

7. **Clarify the structural edge columns** (`§6`, Table 2). “Typed message edges” counts only the four Movement/LaneGroup relations (`4 × Movements`); “unsignalized connectors” is a separate edge count. The caption technically says this, but the prose “Stuttgart has the most … typed Movement-message edges” and table layout could still lead readers to treat the former as total GNN edges. Rename the column to “Movement–LaneGroup edges” or add a “total graph edges” column equal to typed edges plus unsignalized connectors.

8. **Use consistent split terminology across tables and prose.** Table 2 uses “No PPO rollout,” Table 3 uses lowercase “rollout” / “no PPO rollout,” the CSV uses “held out from PPO rollouts,” and prose alternates among held-out evaluation and validation. Recommended canonical labels: “PPO rollout” and “held out from PPO rollouts,” with one sentence explaining that Freiburg was periodically monitored and is therefore a validation/evaluation city, not an untouched test city.

9. **The phase-synthesis figure title slightly overstates safety** (`docs/assets/phase-synthesis-pipeline.svg`; Figure 3). Its embedded title says “conflict-safe selectable phases,” while the manuscript carefully limits this to compatibility under SUMO foes plus the added merge rule. Prefer “conflict-derived selectable phases” inside the graphic to match the section and caption.

10. **The overview’s mask terminology can be more precise** (`docs/assets/movement-scoring-generalist-policy.svg`; Figure 1). Synthesized incidence rows are already conflict-compatible selectable phases; the runtime mask enforces temporary availability (not all aspects of legality). Rename “runtime legal mask” to “runtime availability mask,” matching §4.6 and avoiding the suggestion that masking creates conflict safety.

## Verified items (no correction needed)

- **Structural table:** all five rows in Table 2 match `docs/results/city_structure_statistics.csv`, including controller counts, LaneGroups, Movements, connectors, phase totals/ranges, four typed edges per Movement, and lane-kilometres. Mannheim is largest by total junctions and graph-node count; Stuttgart has the most Movements and Movement/LaneGroup typed edges.
- **Feature dimensions:** `src/movement/dataset.py` serializes 29 LaneGroup entries (7 static + 22 dynamic, excluding `halting_count_detector`) and 4 Movement entries (controlled-link count, oracle demand, normalized demand, previous-green indicator). The draft correctly distinguishes retained dataclass fields from checkpoint inputs.
- **Phase synthesis:** the draft consistently says **all maximal compatible sets/cliques**, correctly distinguishes maximal from maximum, and correctly states why smaller protected phases are retained.
- **Iteration-85 means:** every throughput, completion, wait-density, and completed-trip waiting value in Table 3 agrees with the mean rows in the archived `summary.csv` after displayed rounding. The percentage comparisons in §9.1 are also correct.
- **Dispersion figure:** Figure 5 correctly shows six seed observations and mean ± one sample standard deviation; its caption correctly says these are not confidence intervals.
- **Freiburg protocol:** the draft correctly states zero PPO rollout jobs, periodic evaluation, and therefore validation/held-out-rollout evidence rather than an untouched final test.
- **Worked incidence example:** the matrix multiplication is internally correct (`[1.6, 0.6, 2.1]ᵀ`), and the mask example correctly removes the highest logit before categorical selection.

## Remaining direct-work checklist

- Correct/regenerate the training/evaluation figure and its caption.
- Correct Figure B1 labelling/caption; inspect all final rendered figures at publication size.
- Apply the terminology fixes above to Figure 1, Figure 3, and split labels.
- Complete the primary-source related-work and bibliography audit, including exact official SUMO citations.
- After edits, perform one final equation/code audit and table/evidence audit, then render the archival PDF and inspect cross-references and asset legibility.
- Keep the additional experiments in Appendix C explicitly optional: fresh seeds, independent training runs, untouched city, longer horizons/demand regimes, reward and representation ablations, and a controlled modern-generalist comparison are future evidence, not prerequisites for this representation-focused draft.
