# Unstructured data to forecasting embeddings

Proposed on 2026-09-06. This is a workflow recommendation, not an implemented pipeline or a claim that one encoder is universally best. Review the [mandatory pre-training checklist](../../docs/pre_training_checklist.md) before training.

## Initial design

Use a frozen pretrained text encoder, precompute embeddings offline, and train the forecasting model on aligned numerical features plus one vector per actual text source per day. Keep chunk and document vectors so pooling can change without re-embedding everything.

Start with `text-embedding-3-small`, requesting 512 dimensions as a compact baseline. The OpenAI embedding API supports shortened vectors through `dimensions`; the model's default is 1,536 dimensions. The 512-dimension choice is an engineering starting point to validate, not an empirically established optimum for NGN forecasting. Compare default dimensions and/or `text-embedding-3-large` only on the validation period if the text-enabled model warrants further work. [Official embedding guide](https://developers.openai.com/api/docs/guides/embeddings).

Use offline Batch requests for the historical corpus after a representative extraction/embedding pilot. The Batch API supports embeddings and offers a 50% discount versus synchronous requests, with a 24-hour processing window. Cache by content and configuration hashes and retry only missing/failed records. [Official Batch guide](https://developers.openai.com/api/docs/guides/batch).

No API requests or paid processing have been started by writing this plan.

## 1. Prepare real documents

Start with institutional CBN/DMO/Fed/BoE documents whose indexes are already collected, and available NBS report PDFs. Expand publisher sitemaps into actual article URLs before trying to embed them. Filter irrelevant/navigation links and duplicate HTML/PDF alternatives; the current indexes are not a validated document corpus.

Extract clean HTML body text and PDF text. Use OCR for scanned pages when needed. Preserve section headings, paragraph order, tables with units, policy changes, negations, and dates. Remove repeated navigation, headers, footers, and boilerplate. Inspect samples from every source and decade; record extraction failures explicitly. Keep raw source hashes and immutable document versions.

Treat numerical spreadsheets as quantitative data and verified holiday/meeting occurrences as event features. Do not fill missing text columns with source names, sitemap XML, recurring event definitions, or model-generated historical narratives. A title/description-only experiment may be useful, but must be labeled separately from a full-document experiment.

## 2. Establish historical availability

Each document needs `document_id`, `source_id`, source URL, raw content hash, title, body, language, publication timestamp, timezone, timestamp precision, availability timestamp/evidence, extraction status/version, and acquisition timestamp. Publication and acquisition times are different fields.

Use text only after it was actually available at the forecast origin. Documents with year-only dates remain outside daily training until dates are resolved. Do not substitute sitemap modification time for article publication time. For date-only publications, choose and document a conservative next-day availability convention if no intraday evidence exists; it does not resolve uncertain publication dates or revised content. Modern archive revisions may not represent the historical text.

Document that a modern pretrained encoder can have seen later material during pretraining. A fixed encoder and causal input filtering do not prove an encoder trained before every backtest date; describe the experiment's historical-validity limits.

## 3. Chunk and embed

Split by paragraph/section boundaries, initially targeting roughly 500–800 tokens per chunk with a small overlap (about 50–100 tokens). Keep short releases whole. Preserve table headers with relevant rows and title/section context. These sizes are tunable starting points; validate them on actual extraction samples.

Send cleaned document content to the encoder. Keep source identity and dates as explicit metadata rather than relying on the vector to encode provenance. Do not summarize away numerical detail or direction before embedding. One shared encoder/configuration across sources keeps the dimensions and representation space consistent.

Persist float32 chunk vectors and chunk offsets in Parquet with document ID, content hash, model ID, dimensions, available model revision, extraction/chunking version, and embedding request identity. Cache exact returned vectors when the provider does not expose an immutable model revision. Check vector length and finiteness, duplicate work, and request completeness.

## 4. Pool without letting document length dominate

For the first baseline:

1. Mean-pool chunks into a document vector, accounting for overlap if token weighting is used; normalize the resulting nonzero vector.
2. Mean-pool the unique document vectors available for each source/day; normalize the resulting nonzero vector.
3. Preserve document/chunk counts and pooling version. Retain originals to evaluate learned pooling later.

Pooling documents before pooling the day stops long reports from receiving one vote per chunk against a short release's single vote. This is a simple baseline and may blur opposing stories; evaluate it against alternatives using forecasting validation metrics.

The intended tensor is `text[day, source, embedding_dimension]`. Do not concatenate all publishers into one undifferentiated daily text string. Do not forward-fill old text as if it were new; the lookback window already contains earlier documents.

## 5. Align and store

Build source/day views compatible with `daily_text_embeddings.parquet`: date, source ID, model identity/revision, embedding, document count, and `has_text`. Use the forecast-availability day, not the economic reference period. Keep stable source ordering and the same dimensions for every source.

Missing source/day vectors are masked in the model. Keep a separate coverage status such as `observed_with_documents`, `verified_no_documents`, `not_collected`, or `extraction_failed`; the current boolean `has_text` alone cannot express these differences. Extend the schema and/or add a coverage companion table before relying on that distinction. Do not turn the 132-column inventory automatically into 132 text channels: it contains quantitative-only and provenance entries as well as duplicate source/query concepts.

Parquet fits the existing offline training workflow. A vector database is optional for semantic search and is not required to load these tensors.

## 6. Evaluate whether text actually helps

Use the exact same forecast origins/splits to compare numerical-only, title-only (if attempted), and full-document embeddings. Compare forecast errors by target and horizon, by time period, and by source availability; retrieval similarity benchmarks do not establish currency forecasting value.

Fit learned projections, dimensionality reduction, pooling, normalizers, and forecasting weights on training data only. Encoder/model/dimension selection uses validation data, not the held-out test. Precomputation with a fixed encoder is separate from learning preprocessing across all dates.

Before scaling, run a representative pilot across institutions, publishers, older scans, and newer digital documents. Review extraction and date evidence, estimate actual token volume, verify source/day alignment, and record the budget. The highest-priority work is reliable bodies and dates, followed by embeddings.
