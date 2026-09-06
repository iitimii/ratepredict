# Mandatory pre-training checklist

Requested by the project owner on 2026-09-06. **Go over this checklist before starting any training run.** Record the experiment scope and review outcome below. Unchecked applicable items block training; items outside the explicitly selected scope must be marked not applicable with a reason. A numerical-only experiment does not require finished text ingestion, but must not be described as a full multimodal experiment.

Status: **NOT REVIEWED — NOT CLEARED FOR TRAINING**.

## 1. Experiment scope and immutable inputs

- [ ] Specify original production 2-hour forecasting or isolated transformer-based daily forecasting. They have different targets and must not be conflated.
- [ ] Specify numerical-only, metadata/headline-only exploratory multimodal, or full-document multimodal scope.
- [ ] Freeze dataset hash, source registry, extraction/version metadata, code revision, feature list, and experiment configuration.
- [ ] Review all requested sources, including missing sources from both Data Sources CSVs and the original system. Record included and excluded sources and reasons.
- [ ] Review [dataset completeness](../data/COMPLETENESS.md). Distinguish acquisition, date alignment, validation, and embedding progress. Do not describe the 20-year calendar as 20 years of complete USDT/NGN targets.

## 2. Targets and samples

- [ ] Define the target market/venue, timezone, daily cutoff or 2-hour bucket, forecast origin, lookback, and horizon.
- [ ] For the daily experiment, explicitly define `avg_rate`: the source has OHLCV, not a ready-made daily average. Do not silently use close or claim a true trade VWAP from aggregated OHLCV.
- [ ] Build and validate `avg_rate`, `high_rate`, and `low_rate`, or explicitly configure another documented target contract. Confirm units and high/low relationships.
- [ ] Validate daily aggregation: first open, maximum high, minimum low, last close, summed genuine activity; handle incomplete days and malformed bars explicitly.
- [ ] Verify the seven-day label formulas and inverse transformation in `transformer-based/labels.py` and `normalization.py` against manually checked samples.
- [ ] Recompute valid sample counts after final label construction and filtering. Missing future targets exclude samples; never fill future labels.
- [ ] Decide whether fully observed lookback history is required or masked history is allowed, and record the resulting sample-count difference.

Snapshot evidence: 2,254 complete USDT/NGN OHLC days from 2020-01-08 through 2026-03-10. There are 2,247 candidate seven-day origins using close only as a provisional availability proxy, and 1,883 also have 365 consecutive observed price-history days. These are overlapping windows, not independent examples or finalized label counts. The Parquet view lacks the default target column names. The dataset implementation allows masked input history.

## 3. Numerical quality and historical availability

- [ ] Review anomalies, duplicates/corrections, missing periods, currencies, units, and historical definition changes for every included feature.
- [ ] Confirm placeholder all-zero trade fields are masked where unavailable; genuine zero activity must remain distinguishable.
- [ ] Preserve missingness masks and define feature staleness rules. Do not fill from future observations.
- [ ] Establish publication/availability timestamps for monthly/quarterly/annual macro series. Exclude unresolved series from the first causal experiment or document a defensible conservative availability convention.
- [ ] Check daily market observations against the forecast cutoff and timezone too; global-market same-day closes may occur after the chosen origin.
- [ ] Document revisions and vintage limitations. Period dates and current revised history are not proof of what was known at a historical origin.
- [ ] Fit imputation, clipping thresholds, dimensionality reduction, feature selection, and normalization only on training data where learned; reuse fitted state thereafter.

## 4. Unstructured data (required when text is used)

- [ ] Review [the embedding workflow](../transformer-based/docs/unstructured_embedding_plan.md), identify which source IDs really represent text sources, and freeze their ordering.
- [ ] Download and extract permitted document bodies, or explicitly label a title/description-only experiment. Sitemap links, file indexes, and generic calendar definitions are not document bodies.
- [ ] Check PDF/HTML extraction and OCR on representative old/new files, including negation, decimals, units, tables, dates, and page ordering.
- [ ] Validate source relevance and deduplicate navigation links, repeated documents, alternate HTML/PDF versions, and syndicated articles. Keep provenance and document versions.
- [ ] Attach verified publication/availability dates. Keep year-only records out of daily training until resolved; do not use sitemap modification time as article publication time.
- [ ] Verify manual holiday dates and actual event occurrences separately. Do not expand approximate legacy schedules into claimed historical events or repeat retrospective causal commentary across past dates.
- [ ] Freeze embedding model identity, dimensions, preprocessing/chunking/pooling versions, input hashes, and stored vectors. Record limitations from using a modern pretrained encoder in historical backtests.
- [ ] Validate all vectors are finite and dimensionally consistent; cache exact vectors and map every vector to source documents.
- [ ] Aggregate chunks into documents, then documents into source/day vectors using only text available at that origin. Retain document counts, coverage status, and missingness masks.
- [ ] Distinguish verified no-publication days from failed or incomplete collection. Never embed empty-source names or fill missing historical text with generated prose.

Current `daily_text_embeddings.parquet`: zero rows. Full-document multimodal readiness is blocked. Mark this section not applicable only for an explicitly numerical-only run.

## 5. Evaluation and leakage controls

- [ ] Define chronological train, validation, and untouched test intervals. Do not randomly split overlapping windows.
- [ ] Purge boundary origins so target dates cannot overlap across partitions; check actual timestamp sets for the chosen horizon. Historical input-context overlap is not itself prohibited.
- [ ] Fit normalizers on training data only and reuse them for validation/test/inference.
- [ ] Define primary errors by horizon and target channel and establish a persistence/no-change baseline; compare numerical-only against text-enabled models on identical origins.
- [ ] Select model size and tuning budget appropriate to the limited effective sample size. Use validation for selection and keep the test period untouched.
- [ ] Check performance by historical period/regime and missingness, not only a pooled score. Text must demonstrate incremental held-out forecasting value.

## 6. Executable training and reproducibility

- [ ] Implement the actual model, loss, optimizer, batching, training loop, validation, early stopping, and checkpoint/restart behavior for the chosen dataset.
- [ ] Replace/adapt the unrelated electricity example in `transformer-based/run_chronos.py` if using that route. It is not a RatePredict training entrypoint.
- [ ] Verify Parquet-to-dataset adaptation, UTC-midnight index, configured target columns, feature ordering, and empty-text handling for numerical-only runs.
- [ ] Run the relevant existing dataset/label/normalizer tests and an end-to-end batch/forward/backward smoke check; verify finite loss and expected shapes.
- [ ] Record seeds, dependency versions, hardware, data hash, model configuration, fitted preprocessing, metrics, and output paths in the run manifest.

## Review record — complete before training

- Review date:
- Reviewed with:
- Experiment scope:
- Dataset/code identity:
- Target definition and time cutoff:
- Included/excluded features and reasons:
- Train/validation/test intervals and eligible sample counts:
- Text section applicability:
- Remaining blockers or documented limitations:
- Outcome: NOT REVIEWED / BLOCKED / READY FOR SPECIFIED EXPERIMENT

This checklist is a documented workflow gate, not yet an automatically enforced training guard.
