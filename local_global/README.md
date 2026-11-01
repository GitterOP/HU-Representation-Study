# Local, global, and endmember preservation

This folder answers whether low global reconstruction error also implies good
preservation of critical spatial regions, material abundances, and endmember
signatures. It analyzes existing outputs; it does not train or tune models.

Run from any directory with the project's scientific Python environment:

```bash
python local_global/analyze_local_global.py
```

The script resolves the project root from its own location. Optional arguments
`--project-root PATH` and `--output-dir PATH` allow analysis of another checkout
without modifying its inputs. Dependencies: NumPy, SciPy, pandas, Matplotlib,
PyYAML. All output files are regenerated in this folder by default.

Read `analysis_report.md` for the Spanish findings, `discussion_draft.md` for an
English discussion draft, and `spec.md` for the method-independent region rules
and interpretation limits. No existing LaTeX file is edited.

Main figures:
- `global_local_endmember.png/.pdf`: global reconstruction error versus boundary
  reconstruction, boundary abundance error, and mean endmember SAD.
- `preservation_rankings.png/.pdf`: rank agreement across complementary criteria.
- `material_preservation_rankings.png/.pdf`: global rankings versus each
  dominant-material region and each individual endmember signature.
- `regions_DATASET.png`: reference material layout, boundary band, mixture
  structure, and the common critical mask.
- `spatial_errors_DATASET.png`: all six systems with fixed reconstruction and
  abundance color scales across methods and datasets.
- `endmember_signatures_DATASET.png`: raw and peak-normalized matched signatures.

Tables:
- `global_metrics.csv`: conventional/relative reconstruction RMSE, SAM, mean and
  worst SAD, abundance aggregation variants, error tails, macro material RMSE.
- `region_metrics.csv`: identical masks across methods; local metrics and squared
  error shares. Overlapping mask shares must not be added.
- `endmember_metrics.csv`, `material_connections.csv`,
  `material_region_comparisons.csv`: material-wise links among
  signatures, local reconstruction, and abundance accuracy.
- `rankings.csv`, `rank_correlations.csv`, `winner_agreement.csv`: descriptive
  comparisons within each scene, using six systems rather than pooled pixels.
- `sensitivity.csv`: boundary widths and mixture cutoffs.
- `metric_audit.csv`: recomputed results against the source summary.
- `reference_consistency.csv`: reconstruction from reference signatures and
  abundances under the repository's existing scales, without calibration.
- `winner_error_localization.csv`: mask area versus squared-error contribution
  for the best global reconstruction system in each dataset.
- `region_definitions.csv`, `reference_masks_DATASET.npz`: exact mask coverage.
- `provenance.json`: input hashes, dependency versions, shared color limits.

Important definitions: the source CSV calls conventional reconstruction RMSE
`rRMSE`. The actual norm-relative error is exported separately as `relative_rmse`.
The source abundance `aRMSE` is the mean of pixelwise RMSE, which differs from
pooled abundance RMSE. This analysis preserves both definitions explicitly.

The reference mixing model is also checked directly. Reference signatures and
abundances do not exactly reproduce the observed cube under the loaded scales,
especially in Samson. Interpret disagreement between evaluation targets subject
to this benchmark limitation; do not attribute it solely to learned features.

All displayed method names follow PCA, AE, AE-Patch, ViT, MAE, DINO. Figure text
is in English. These are descriptive results for the saved systems; they do not
identify architectural causes or establish significance across repeated runs.

Run the synthetic sanity checks with:

```bash
python -m unittest discover -s local_global -p 'test_*.py' -v
```

These checks cover endmember permutation, the possibility of exact reconstruction
with incorrect signatures/abundances, and disjoint spatial error decomposition.
