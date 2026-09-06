# USD/NGN Decision Intelligence

This repository implements the production specification for a world-class USD/NGN 2-hour decision intelligence system for an OTC desk.

It contains two layers:

- Executable domain contracts and policy logic in [`ratepredict`](./ratepredict).
- Production-grade methodology, governance, data, and runbook documents in [`docs`](./docs).

## Repository Structure

- [`docs/world_class_production_methodology_v3.md`](./docs/world_class_production_methodology_v3.md): End-to-end methodology and operating model.
- [`docs/training_blueprint.md`](./docs/training_blueprint.md): Training objectives, required datapoints, dataset design, and environment guidance.
- [`docs/pre_training_checklist.md`](./docs/pre_training_checklist.md): Mandatory review before any training run; record scope, target definitions, leakage checks, and readiness.
- [`docs/runtime_data_pipeline.md`](./docs/runtime_data_pipeline.md): Scheduled runtime bar refresh, cached external data flow, and remaining source decisions.
- [`docs/data_contracts.md`](./docs/data_contracts.md): Point-in-time data contracts and interface semantics.
- [`docs/acceptance_gatebook.md`](./docs/acceptance_gatebook.md): Formal pass/fail gates for promotion, continuation, and stop-review.
- [`docs/model_governance_standard.md`](./docs/model_governance_standard.md): Promotion, continuation, challenger, and audit controls.
- [`docs/operating_modes_and_runbooks.md`](./docs/operating_modes_and_runbooks.md): Degradation hierarchy, verification workflows, and incident actions.
- [`app/main.py`](./app/main.py): FastAPI desk app for live inference and signal review.
- [`artifacts/`](./artifacts): Trained model files, metadata, feature importances, and the latest BigQuery export.
- [`ratepredict/types.py`](./ratepredict/types.py): Public types that operationalize the specification.
- [`ratepredict/policy.py`](./ratepredict/policy.py): Deterministic policy engine defaults for modes, penalties, gates, and recommendations.
- [`tests/test_policy.py`](./tests/test_policy.py): Acceptance tests for degradation logic, circularity controls, and economics gates.

## Locked Production Defaults

- Launch posture: human in the loop.
- Risk posture: capital preservation.
- Primary objective: predict the next 2-hour move of the USDT/NGN close, because that is the OTC execution price.
- Cross-venue and macro series remain exogenous features, not target components.
- Economic continuation gate: `+15 bps` monthly incremental net alpha versus passive spread-capture baseline.
- Included feature families must justify themselves in ablations; any family that does not improve signal quality or economics should be removed.
- Live `usdtngn` is fetched from the protected qbot rates-export endpoint (provider currently Bybit), while live `btcngn` still comes from the Quidax public ticker endpoint.

## Live Quote Setup

Live `usdtngn` is fetched directly from the protected qbot rates-export endpoint. Set these environment variables on the server that runs the Streamlit app:

```bash
QBOT_USDTNGN_RATE_URL=https://qbot-test.qdx.global/api/v1/rates-export/current/USDTNGN
QBOT_SERVICE_TOKEN=...
QBOT_CF_ACCESS_CLIENT_ID=...
QBOT_CF_ACCESS_CLIENT_SECRET=...
```

The browser never calls qbot directly. Streamlit makes the HTTP request on the server side with the service token and Cloudflare Access credentials, so other users can open the app normally without seeing or supplying those secrets.

### Important: Streamlit Community Cloud

Cloudflare Access service tokens authenticate the caller, but they do not bypass
separate Cloudflare IP/ASN, country, WAF, Browser Integrity Check, or bot rules.
If the same credentials return `200` locally and Cloudflare HTML with `403` on
Community Cloud, secret loading is not the problem: a Cloudflare security rule is
rejecting the Community Cloud source network.

The error shown by the app includes a safe `cf_ray` identifier. Give that Ray ID
and the request timestamp to the Cloudflare zone owner, then:

1. Find the request in **Security > Events** and identify the exact product/rule
   that blocked it.
2. Keep the Access application policy set to **Service Auth** for the issued
   service token.
3. Add the narrowest exception for the exact hostname/API path and the approved
   deployment source. Do not make the endpoint public.

Cloudflare documents the required [service-token headers](https://developers.cloudflare.com/cloudflare-one/access-controls/service-credentials/service-tokens/)
and the [WAF skip options](https://developers.cloudflare.com/waf/custom-rules/skip/options/).
Streamlit publishes its current [Community Cloud egress IP list](https://docs.streamlit.io/deploy/streamlit-community-cloud/status/#ip-addresses),
but warns that those addresses may change without notice.

If the security policy cannot accommodate Community Cloud, deploy on company
infrastructure with an approved/static egress IP. A different public cloud host
is only a solution if its outbound network is explicitly allowed by Cloudflare.

### Deploy on Render (alternative host; verify its egress is allowed)

This repo includes [`render.yaml`](./render.yaml).

1. Create a Web Service on Render from this GitHub repo
2. Fill in secret env vars: `QBOT_SERVICE_TOKEN`, `QBOT_CF_ACCESS_CLIENT_ID`, `QBOT_CF_ACCESS_CLIENT_SECRET`, `GEMINI_API_KEY`
3. Deploy and open the Render URL

Or run locally (guaranteed if your laptop curl works):

```bash
streamlit run app/main.py
```

## Verification

Run:

```bash
. .venv/bin/activate
python -m unittest discover -s tests
streamlit run app/main.py
```
