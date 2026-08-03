# ProofPrint

**Chain of custody for AI-generated media.** Generate through a multi-provider Genblaze
pipeline, seal a SHA-256 provenance manifest *inside* the file itself, store it immutably
on Backblaze B2 — then prove, byte for byte, that it hasn't been touched.

Built for the **Backblaze Generative AI Media Hackathon** on [Genblaze](https://github.com/backblaze-labs/genblaze) + [Backblaze B2](https://www.backblaze.com/cloud-storage).

---

## The problem

Since **2 August 2026**, [EU AI Act Article 50](https://artificialintelligenceact.eu/article/50/)
requires providers of generative AI systems to mark synthetic audio, image, video and text
**in a machine-readable format, detectable as artificially generated**.

Almost every generative media stack in production today fails this. A model returns a URL,
someone downloads a PNG, it moves through Slack, a CMS and three re-exports, and within a
day nobody can answer the two questions that matter:

1. **What produced this?** Which model, which prompt, which parameters, when?
2. **Is this still what was produced?** Or has it been edited since?

Newsrooms, agencies, stock marketplaces and compliance teams are being asked those questions
now, and "check our internal spreadsheet" is not an answer that survives a dispute.

## The approach

ProofPrint makes the proof **part of the asset** and the reference copy **immutable**.

```
brief ──► Gemini chat ──► NVIDIA NIM (FLUX.1) ──► manifest ──► embed in file ──► B2 Object Lock
          expand prompt    ├ fallback: SD 3.5 Turbo   SHA-256    PNG iTXt        GOVERNANCE
                           ├ fallback: SDXL           canonical  JPEG XMP        + ledger record
                           └ failover: Google Gemini  hash       MP4 uuid box    + content-addressed
```

Verification then answers three **independent** questions, because a single boolean can't
distinguish the ways provenance breaks:

| Layer | Question | Mechanism |
|---|---|---|
| 1 | Is a manifest present? | `handler.extract()` on the file's own bytes |
| 2 | Is the record internally consistent? | `Manifest.verify_hash()` — editing the embedded prompt/model/timestamp breaks the canonical hash |
| 3 | Do the bytes still match what we sealed? | Re-hash the upload, compare against the **Object-Locked** reference digest in B2 |

Layer 3 is what makes this more than self-attestation. The reference hash is written to an
immutable B2 object *before the file ever leaves the pipeline*, so forging a pass requires
altering a record that cannot be altered.

Verdicts: `AUTHENTIC` · `MODIFIED` (genuine record, edited bytes) · `TAMPERED` (record itself
was altered) · `UNSIGNED` · `UNKNOWN_ORIGIN`.

---

## How this uses Backblaze B2

B2 is not a dumping ground here — it is the evidence store, and four distinct B2 capabilities
carry real weight:

**1. Object Lock as the trust anchor.** Sealed assets, manifests and ledger records are written
with `ObjectLockConfig(mode="GOVERNANCE", retain_until=…)` via Genblaze's `ObjectStorageSink(manifest_lock=…)`
and `backend.put(object_lock=…)`. Ordinary deletes and overwrites fail. This is the property
that makes layer-3 verification meaningful — the ledger record holding the reference digest is
precisely what an attacker would need to rewrite, so it is locked too. GOVERNANCE (not
COMPLIANCE) keeps a demo bucket recoverable by a privileged key; retention is configurable via
`B2_OBJECT_LOCK_DAYS`.

Object Lock is settled with a single throwaway probe object at startup rather than assumed: a
bucket created without it, or an app key lacking `writeFileRetentions`, degrades to standard
writes with a loud log line instead of killing a mint half-way through. `/api/health` and every
ledger record report what was *actually* applied, never what was merely requested.

**2. Content-addressable keying.** `KeyStrategy.CONTENT_ADDRESSABLE` means the storage key *is*
the SHA-256 — the path itself is an integrity claim, and identical outputs across thousands of
generated variants collapse onto one object.

**3. An append-only ledger, one immutable object per mint.** Records live at
`proofprint/ledger/{timestamp}_{run_id}.json` rather than in one mutable index file, so
concurrent mints can never clobber each other and a prefix listing is a cheap chronological
scan. This is the app's only database — there is no Postgres.

**4. B2 serves the bytes.** `/api/asset/{sha}` issues a short-lived presigned URL and redirects.
Media is served by B2 directly, never proxied through the app.

Bucket layout:

```
proofprint/
  runs/{tenant}/{date}/{run_id}/manifest.json      Genblaze sink · Object Lock
  assets/{sha[:2]}/{sha[2:4]}/{sha}.png            content-addressed raw output
  sealed/{sha256}.png                              manifest embedded in-file · Object Lock
  ledger/{timestamp}_{run_id}.json                 append-only index
```

## How this uses Genblaze

Genblaze is the orchestration and provenance engine, not a thin wrapper:

- **`Pipeline`** builds each mint, with `tenant_id` / `project_id` for multi-tenancy.
- **`fallback_models`** gives in-provider model fallback: FLUX.1-schnell → SD 3.5 Turbo → SDXL,
  handled inside Genblaze without the caller knowing.
- **Cross-provider failover** on top: if the whole NVIDIA NIM leg is down, the mint re-runs
  against Google. Genblaze's uniform Pipeline API is exactly what makes that a provider swap
  instead of a rewrite. Every attempt — including failures and their latencies — is surfaced
  in the UI, because "it fell back twice and still delivered" is the real production story.
- **`ObjectStorageSink` + `S3StorageBackend.for_backblaze`** with `CONTENT_ADDRESSABLE` keying
  and `manifest_lock` for immutable manifests.
- **Provenance manifests** — the core of the product. `manifest.canonical_hash`,
  `verify_hash()`, `verify()` and `verification_report()` drive the verification verdicts.
- **`genblaze_core.media` handlers** (`PngHandler`, `JpegHandler`, `WebpHandler`, `Mp4Handler`,
  `get_handler`, `sniff_mime`) embed and extract the manifest in-file. This is the machine-readable
  marking Article 50 asks for.
- **`parent_run_id`** (set the way `Pipeline.from_result()` sets it) gives iteration lineage —
  "Iterate on this" links a new run to its ancestor, and certificates render the full v1 → v2 → v3
  chain plus direct children.
- **`genblaze_google.chat`** expands a short human brief into a production prompt, so the
  certificate shows *both* what the human asked for and what the model was actually given.

## Providers and models

| Role | Provider | Models |
|---|---|---|
| Prompt expansion | Google (`genblaze-google`) | `gemini-2.5-flash` |
| Image generation (primary) | NVIDIA NIM (`genblaze-nvidia`) | `black-forest-labs/flux.1-schnell` |
| Image fallback (in-provider) | NVIDIA NIM | `stabilityai/stable-diffusion-3-5-large-turbo`, `stabilityai/stable-diffusion-xl` |
| Image failover (cross-provider) | Google (`genblaze-google`) | `gemini-2.5-flash-image`, `gemini-3.1-flash-image` |
| Storage | Backblaze B2 (`genblaze-s3`) | S3-compatible, Object Lock |

Every image model in the primary path is **open-weight** (FLUX.1, Stable Diffusion 3.5, SDXL).

---

## Run it locally

Requires **Python 3.11+**.

```bash
git clone <this-repo> && cd proofprint
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env    # then fill in the values below
set -a && source .env && set +a

uvicorn app.main:app --reload --port 8000
```

Open <http://localhost:8000>. `GET /api/health` reports exactly what is and isn't configured.

### Getting the credentials

| Variable | Where |
|---|---|
| `B2_BUCKET`, `B2_KEY_ID`, `B2_APP_KEY` | [Backblaze B2](https://www.backblaze.com/cloud-storage) → create bucket **with Object Lock enabled**, then [App Keys](https://secure.backblaze.com/app_keys.htm) |
| `NVIDIA_API_KEY` | [build.nvidia.com](https://build.nvidia.com) → any model → *Get API Key* (free credits, no card) |
| `GEMINI_API_KEY` | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) (free tier) |

> **Object Lock must be enabled at bucket creation** — B2 cannot turn it on afterwards. If your
> bucket doesn't have it, set `B2_OBJECT_LOCK_DAYS=0`; everything works except the immutability
> guarantee.

## Deploy

The repo ships a `Dockerfile` and `render.yaml`.

1. Push to GitHub.
2. Render → **New → Blueprint** → point at the repo.
3. Set the secret env vars (`B2_*`, `NVIDIA_API_KEY`, `GEMINI_API_KEY`). Everything else has defaults.

Health check is wired to `/api/health`.

## API

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | Config transparency — providers, bucket, Object Lock state |
| `POST /api/mint` | `{brief, expand, parent_run_id, project_id}` → generate, seal, store, record |
| `POST /api/verify` | multipart file upload → three-layer verdict |
| `GET /api/ledger` | Newest-first archive, read from B2 |
| `GET /api/record/{sha256}` | One ledger record |
| `GET /api/lineage/{run_id}` | Ancestors + direct children |
| `GET /api/asset/{sha256}` | 302 → short-lived presigned B2 URL |

Interactive docs at `/docs`.

## Try the tamper demo

1. **Studio** → generate anything → **Download sealed file**.
2. **Verify** → drop that file → `AUTHENTIC`, with the full recovered manifest.
3. Open it in any editor, change one pixel, re-export, drop it again → `MODIFIED`.

## Notes on the SDK

Built against `genblaze==0.4.5` (core 0.3.8 / s3 0.3.6 / nvidia 0.3.3 / google 0.3.4). Two
things worth flagging for other builders:

- **GitHub release tags don't map to PyPI versions.** Tags `v0.5.x`–`v0.7.x` exist on GitHub
  while `0.4.5` is the latest umbrella on PyPI, so `pip install genblaze==0.7.0` fails. Pin
  from PyPI, not from the release page.
- **`Pipeline.from_result()` requires a live `PipelineResult`.** For lineage across HTTP
  requests you only have the parent's `run_id`, and the public API has no way to pass it —
  `from_result()` internally does nothing but set `_parent_run_id`. A public
  `parent_run_id=` argument on `Pipeline(...)` would make cross-process lineage a first-class
  operation.

## License

MIT
