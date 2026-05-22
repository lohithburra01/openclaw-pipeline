# Poster Setup — Resume Here

**Status (2026-05-22):** the n8n poster workflow is **built but not yet imported
or tested**. For this race weekend the videos are posted manually. Pick the
poster setup back up from the steps below.

## The workflow

`formulytics_poster.json` — an importable n8n workflow (15 nodes). It polls the
season-manifest Google Sheet, finds sessions marked `rendered`, captions them
from the real OpenF1 results via Gemini, uploads the video to Zernio, and posts
to Instagram + YouTube. Regenerate it any time with
`python build_poster_workflow.py`.

## Resume steps (in n8n Cloud)

1. Import `formulytics_poster.json` — new workflow → three-dot menu →
   Import from File. Rename it "Formulytics Poster", Save, leave it inactive.
2. Create 4 credentials and attach each to its nodes:

   | Credential (n8n type) | Value | Nodes |
   |---|---|---|
   | Google Sheets OAuth2 | the Google account that owns the Sheet | Read Manifest, Claim Row, Mark Posted |
   | Google Drive OAuth2 | same Google account | Download Video |
   | Header Auth "Gemini" | header `x-goog-api-key` = Gemini API key | Generate Caption |
   | Header Auth "Zernio" | header `Authorization` = `Bearer <zernio key>` | Zernio Presign, Zernio Create Post |

   The "Zernio Upload" node needs no credential — its URL is pre-signed.
3. Edit the **Config** node — paste the manifest Sheet id, and verify the two
   Zernio account ids against the Zernio dashboard.
4. Test without waiting for a race: add one row to the Sheet by hand with
   `render_status` = `rendered` and a real `drive_file_id`, then run the
   workflow once and confirm it posts.

## Not done / known gaps

- The poster has never run in n8n — expect to test-run and fix a node or two.
- Instagram posting via API needs a **Business or Creator** Instagram account.
- For the *automatic* end-to-end (producer feeds the Sheet → poster posts), the
  manifest setup and merging the `season-manifest` branch still need to happen —
  see `docs/superpowers/specs/2026-05-22-pipeline-manifest-design.md` and
  `docs/superpowers/plans/2026-05-22-pipeline-manifest-producer.md`.
