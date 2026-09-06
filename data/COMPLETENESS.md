# Dataset completeness assessment

Assessment date: 2026-09-06. Canonical snapshot window: 2006-01-01 through 2026-09-02, or 7,550 calendar days. Newer curated NBS resources are included in the unstructured inventory but not yet in the canonical snapshot.

## What can currently be measured

| Measure | Value | Interpretation |
|---|---:|---|
| Numerical daily-view populated cells | 131,061 / 475,650 = 27.5541% | 63 existing numerical columns × 7,550 calendar days; raw observation density, not cadence-adjusted collection completeness |
| Complete USDT/NGN OHLC days | 2,254 / 7,550 = 29.8543% | Target-price history covers 2020-01-08 through 2026-03-10 |
| Daily text embeddings | 0 rows | No completed training embeddings; does not imply no acquisition work |
| Canonical document metadata records | 3,755 | Titles/descriptions/URLs, not verified extracted bodies |
| Canonical publisher sitemap records | 3,117 | Archive-discovery material, not articles |
| Newer NBS resource metadata | 28 | 8 PDFs and 20 spreadsheet/archive packages; not 28 extracted text documents |
| Manual holiday ranges / undated legacy definitions | 84 / 69 | Planning/context records, not verified historical text occurrences |
| Inventory source/event columns | 132 | Includes query variants, quantitative references, metadata partitions, and event definitions; not 132 independent text publishers |

## A single percentage is not yet an audited measure

An accurate percentage of **all desired numerical and unstructured data for two decades** is currently unknown. We have not frozen the complete expected series/cadence inventory or enumerated each publisher's historical document archive.

For a provisional dashboard only, assigning equal 50% weight to numerical daily observation density and completed text-embedding coverage produces:

```text
0.5 × 27.5541% + 0.5 × 0% = 13.7770% ≈ 14%
```

Call this a **provisional daily-input coverage proxy**, never “14% of the full corpus collected.” The 50/50 weighting is an explicit reporting assumption, not a measured property or agreed model weighting. Do not use it as a readiness gate or trend metric without freezing its denominator.

It mixes numerical observation density and text processing readiness. Monthly/annual observations and market holidays make numerical daily density too strict, while planned features absent from the 63-column view make it too lenient. It does not account for text acquisition work, reliable no-publication days, data quality, revision/vintage safety, source overlap, or verified historical availability. It is neither an upper nor lower bound on actual full-corpus completeness.

Source-count ratios and canonical long-form record counts are not substitutes: one time-series row can create many feature records, one article can create many chunks, and one sitemap can point to many uncollected articles.

## Definition required for a defensible 100%

1. Freeze the union of required numerical series, document publishers/collections, and event families. Resolve aliases/query duplication while retaining original requirements and missing sources.
2. Define an expected observation calendar for every numerical series: market days, monthly releases, quarterly releases, annual indicators, or verified event dates. Report valid acquired observations / expected observations per feature and year, then aggregate with explicit fixed weights.
3. Enumerate historical article/document URLs and validate archive coverage by source/year. Report indexed, downloaded, extracted, date-validated, and embedded fractions separately against the same expected corpus. Unknown archive denominators remain unknown.
4. Report source-years whose archives were comprehensively checked, including verified zero-document periods. A day without publication is not necessarily missing data.
5. Report raw acquisition separately from training-ready availability, quality, and embeddings. Prefer numerical and text scores side by side; a combined score requires an explicit weighting rule.
6. Preserve a strict 20-year coverage view and a separate feasible-history view. Some crypto markets and publishers did not exist in 2006; document “not applicable” and any source substitution without fabricating history or silently declaring the original full-window requirement satisfied.

Review [the mandatory pre-training checklist](../docs/pre_training_checklist.md) before any training run. A scoped numerical experiment can be valid while the full dataset remains incomplete.
