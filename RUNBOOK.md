# Formulytics Pipeline — Runbook

A plain-language operating guide for the F1 content pipeline. For the full
design, see `docs/superpowers/specs/2026-05-22-pipeline-manifest-design.md`.

## What the pipeline is

Two halves, coordinated by one Google Sheet (the "manifest"):

- **Producer** — renders a video after each F1 race/sprint and uploads it to a
  Google Drive folder. Runs automatically on GitHub Actions.
- **Poster** — captions the video and posts it to Instagram and YouTube. Runs
  on n8n Cloud.

The **manifest sheet** has one row per session. Open it any time to see what has
been rendered (`render_status`) and posted (`post_status`).

Both halves email `info@formula-neon.com` if a run fails, so a problem is never
silent.

## Statuses

- `render_status`: `pending`, `rendered`, `skipped`, `error`.
- `post_status`: `pending`, `posting`, `posted`, `skipped`, `error`.

`skipped` means the pipeline deliberately ignores the row. You can set a row's
`post_status` to `skipped` yourself to stop the poster retrying something.

## Yearly maintenance checklist

1. **Google login token.** If Drive or Sheets access fails, regenerate it: run
   `python get_gdrive_token.py`, then update the `GDRIVE_REFRESH_TOKEN` GitHub
   secret. The Google Cloud consent screen must stay in **Production** mode, or
   the token expires every 7 days.
2. **Team colours.** `get_constructor_color` in `race_replay.py` lists 2026
   teams. When F1 teams change, update it, or new teams render grey.
3. **Gemini model name.** AI models are retired every year or two. When captions
   fail with a model error, update the model name.
4. **Zernio connections.** The poster (running on n8n Cloud) uses Zernio as the
   bridge to Instagram and YouTube. Those platforms drop their links to Zernio
   every few months — reconnect them in Zernio when posts start failing.
5. **GitHub token (`WORKFLOW_PAT`).** Must have **no expiry**, or the producer's
   scheduling silently freezes the day it expires. To (re)create it: at
   github.com/settings/tokens generate a classic token with the `repo` and
   `workflow` scopes, set expiry to "No expiration", then update the
   `WORKFLOW_PAT` secret under the repo's Settings -> Secrets and variables ->
   Actions.

## What to do if...

- **A race did not post.** Open the manifest, find the row.
  - `render_status` is `error`: open the producer's GitHub Actions logs. To
    retry, set `render_status` back to `pending`.
  - `render_status` is `rendered` but `post_status` is `error`: the poster
    failed. To retry, set `post_status` back to `pending`.
- **You got a "producer run failed" email.** Open the linked GitHub Actions run;
  the log names the failed step. Most often it is OpenF1 data not ready (this
  auto-recovers on the next run) or an expired Google token (regenerate it).
- **Drive or Sheets access fails.** Regenerate the Google token (item 1 above).
- **Posting stopped entirely.** Check Zernio's Instagram/YouTube connections and
  reconnect them.
- **The manifest sheet looks damaged.** Google Sheets keeps automatic version
  history: File -> Version history -> restore an earlier version.

## Storage

Each video is roughly 60-70 MB and a season has about 30 races and sprints, so
videos accumulate in Google Drive at roughly 2 GB per year. When the account
nears its storage limit, delete old posted videos or add storage.
