# Local-to-global preservation analysis

## Question and scope
Do systems with lower global reconstruction RMSE also preserve critical spatial
regions, endmember signatures, and material abundances? Analyze the 18 existing
method/dataset results; do not train, tune, rerun unmixing, or modify the manuscript.

## Analysis plan
1. Reconstruct each cube from its saved abundances and endmembers. Audit RMSE,
   SAM, mean SAD, and the repository's mean-pixel abundance RMSE against the CSV.
   Also diagnose reconstruction from reference signatures and abundances under
   the loaded scales, without fitting or changing any calibration.
2. Match endmembers one-to-one to reference signatures by minimum total SAD.
   Apply the same permutation to abundances throughout the analysis.
3. Define masks independently of method performance: four-neighbor dominant-GT
   boundaries, one-pixel 3x3 dilation of those boundaries, mixed pixels with
   maximum GT abundance below 0.8, their union, pure interior pixels (maximum
   abundance at least 0.9 outside the boundary band), dominant material masks,
   and bottom-decile brightness as an exploratory low-signal proxy.
4. Quantify reconstruction RMSE, pixel-error tails, abundance errors, and
   contributions to scene reconstruction squared error for each mask. Evaluate
   each material's SAD alongside its abundance error and local reconstruction.
5. Compare rankings and descriptive Spearman correlations within each dataset
   (six systems). Repeat boundary widths 0/1/2 and mixture cutoffs 0.7/0.8/0.9.
6. Generate English figures with shared error scales, CSVs, provenance, a Spanish
   evidence-based report, and an English discussion draft in Markdown.

## Interpretation rules
- Reconstruction RMSE is sqrt(mean squared spectral residual); relative RMSE is
  that value divided by the RMS signal. Keep both explicit. Do not alter sources.
- The saved aRMSE is mean(sqrt(mean material squared error per pixel)); also
  export pooled abundance RMSE to make the aggregation difference explicit.
- GT masks may overlap; SSE shares must not be summed across overlapping masks.
- Compare systems within datasets. Correlations with n=6 are descriptive; no
  pixel-level p-values, causal architectural claims, or independent-replicate
  claims. Threshold sensitivity is not an independent experiment.
- Low signal does not establish shadow. Dominant-label boundaries do not capture
  every continuous abundance transition. Matched endmembers are scene-level
  signatures; their SAD is not a local per-pixel error.
- Model names are PCA, AE, AE-Patch, ViT, MAE, DINO. Keep internal keys unchanged.

## Deliverables and verification
All new work lives in local_global. Verify shape/finite-value constraints,
unique matching, source metric agreement, equal region masks across methods,
RMSE/SSE decomposition, and consistent figure normalization. Record exact input
hashes. Keep the new folder on the isolated branch while the other AI works in
the main checkout; provide direct links and a portable archive of the results.
