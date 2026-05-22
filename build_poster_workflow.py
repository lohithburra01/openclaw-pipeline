"""
build_poster_workflow.py
------------------------
Generates `formulytics_poster.json` — the n8n workflow for the Formulytics
posting pipeline (the "poster").

The poster reads the season-manifest Google Sheet (NOT the Drive folder),
finds sessions whose video is rendered but not yet posted, writes an AI
caption from the real OpenF1 results, uploads the video to Zernio, posts it
to Instagram + YouTube, and marks the manifest row posted.

Run:  python build_poster_workflow.py
It writes formulytics_poster.json and validates that it is well-formed JSON.

Generating the workflow with a script (rather than hand-writing JSON) means
all the embedded JavaScript is escaped correctly and the output is guaranteed
to be valid JSON that n8n can import.
"""

import json

# ----------------------------------------------------------------------
# JavaScript for the Code nodes (kept as plain strings; json.dump escapes them)
# ----------------------------------------------------------------------

CONFIG_JS = """// EDIT THESE FOUR VALUES, then this workflow needs no other editing.
return [{ json: {
  // The Google Sheet id from its URL: docs.google.com/spreadsheets/d/<THIS>/edit
  manifestSheetId: "PASTE_YOUR_MANIFEST_SHEET_ID_HERE",
  // Your Zernio account ids (Zernio dashboard -> connected accounts).
  zernioInstagramAccountId: "6a103559520992756d9bdd85",
  zernioYoutubeAccountId: "6a1035ed520992756d9be08e",
  // Gemini model id. Update if Google retires this model.
  geminiModel: "gemini-3.5-flash"
} }];
"""

PICK_ROW_JS = """// Turn the raw sheet values into rows, then pick ONE session to post.
const resp = $input.first().json;
const values = resp.values || [];
if (values.length < 2) return [];          // empty sheet or header only

const headers = values[0];
const rows = values.slice(1).map((raw, i) => {
  const r = {};
  headers.forEach((h, j) => { r[h] = (raw[j] !== undefined ? raw[j] : ""); });
  r._row = i + 2;                          // 1-based sheet row (row 1 = header)
  return r;
});

const STALE_MS = 30 * 60 * 1000;           // a 'posting' row older than 30 min crashed
const now = Date.now();

const eligible = rows.filter(r => {
  if (r.render_status !== "rendered") return false;   // video must exist
  if (!r.drive_file_id) return false;
  const ps = r.post_status;
  if (ps === "posted" || ps === "skipped") return false;
  if (ps === "posting") {
    const started = Date.parse(r.post_started_at || "");
    if (started && (now - started) < STALE_MS) return false;  // claimed, still running
  }
  if (ps === "error" && Number(r.attempts || 0) >= 3) return false;  // give up after 3
  return true;
});

if (eligible.length === 0) return [];      // nothing to do; workflow stops here
eligible.sort((a, b) => String(a.session_date).localeCompare(String(b.session_date)));
return [{ json: eligible[0] }];            // oldest unposted session
"""

BUILD_CAPTION_REQUEST_JS = """// Build the OpenF1 results summary and the full Gemini request body.
const row = $('Pick Row To Post').first().json;
const results = $('OpenF1 Result').first().json;
const drivers = $('OpenF1 Drivers').first().json;

const dmap = {};
if (Array.isArray(drivers)) {
  for (const d of drivers) {
    dmap[d.driver_number] = {
      name: d.full_name || d.broadcast_name || d.last_name || ('#' + d.driver_number),
      team: d.team_name || ''
    };
  }
}

let summary = '';
if (Array.isArray(results) && results.length) {
  const classified = results
    .filter(r => r.position != null)
    .sort((a, b) => a.position - b.position);
  summary = classified.slice(0, 10).map(r => {
    const d = dmap[r.driver_number] || { name: '#' + r.driver_number, team: '' };
    const flags = [r.dnf && 'DNF', r.dns && 'DNS', r.dsq && 'DSQ'].filter(Boolean).join('/');
    const gap = (r.position > 1 && r.gap_to_leader != null) ? (' +' + r.gap_to_leader) : '';
    return 'P' + r.position + ' ' + d.name + ' (' + d.team + ')' + gap +
           (flags ? ' [' + flags + ']' : '');
  }).join('\\n');
}

const prompt =
  'You write social captions for Formulytics (@formulytics), an F1 data-' +
  'visualization account. Tone: fan-to-fan, raw, opinionated, all lowercase.\\n\\n' +
  'CONFIRMED OFFICIAL RESULT (from F1 timing - never contradict this):\\n' +
  (summary || '(results not in yet - use Google Search to find them)') + '\\n\\n' +
  'Session: ' + row.gp_name + ' ' + row.session + '\\n\\n' +
  'Use Google Search for context: key battles, crashes, penalties, ' +
  'championship implications, records.\\n\\n' +
  'Return ONLY a JSON object - no markdown, no code fences - with exactly two ' +
  'string keys:\\n' +
  '- "instagram_caption": a punchy hook line with one emoji, then 2-3 short ' +
  'factual lines, then a debate question, then exactly 5 hashtags on one line ' +
  'ending with #f1reels. all lowercase. no em dashes.\\n' +
  '- "youtube_title": format "' + row.gp_name + ' 2026 ' + row.session +
  ' [flag emoji] [Main Story In Title Case] | F1 2026", max 100 characters.';

return [{ json: {
  geminiBody: { tools: [{ google_search: {} }], contents: [{ parts: [{ text: prompt }] }] },
  sessionId: row.session_id,
  driveFileId: row.drive_file_id,
  rowNumber: row._row
}}];
"""

PARSE_CAPTION_JS = """// Pull the JSON caption out of the Gemini response.
const resp = $input.first().json;
let txt = '';
try {
  txt = (resp.candidates[0].content.parts || [])
    .filter(p => p.text).map(p => p.text).join('');
} catch (e) {
  throw new Error('Gemini returned no usable text');
}
txt = txt.replace(/```json/gi, '').replace(/```/g, '').trim();

let parsed;
try { parsed = JSON.parse(txt); }
catch (e) { throw new Error('Gemini did not return valid JSON: ' + txt.slice(0, 200)); }

const clean = s => String(s || '').replace(/[\\u2014\\u2013]/g, '-').trim();
let yt = clean(parsed.youtube_title);
if (yt.length > 100) yt = yt.slice(0, 97).trim() + '...';

const prev = $('Build Caption Request').first().json;
return [{ json: {
  instagramCaption: clean(parsed.instagram_caption),
  youtubeTitle: yt,
  sessionId: prev.sessionId,
  driveFileId: prev.driveFileId,
  rowNumber: prev.rowNumber
}}];
"""

# ----------------------------------------------------------------------
# Node + connection assembly
# ----------------------------------------------------------------------

nodes = []
order = []  # node names in execution order, for linear connections


def add(name, ntype, type_version, params):
    nodes.append({
        "parameters": params,
        "id": name.lower().replace(" ", "-").replace("?", ""),
        "name": name,
        "type": ntype,
        "typeVersion": type_version,
        "position": [260 + len(nodes) * 260, 320],
    })
    order.append(name)


HTTP = "n8n-nodes-base.httpRequest"
CODE = "n8n-nodes-base.code"

GOOGLE_AUTH = {"authentication": "predefinedCredentialType",
               "nodeCredentialType": "googleSheetsOAuth2Api"}
DRIVE_AUTH = {"authentication": "predefinedCredentialType",
              "nodeCredentialType": "googleDriveOAuth2Api"}
HEADER_AUTH = {"authentication": "genericCredentialType",
               "genericAuthType": "httpHeaderAuth"}


def code_node(name, js):
    add(name, CODE, 2, {"jsCode": js, "mode": "runOnceForAllItems"})


# 1 - run every 15 minutes
add("Every 15 Minutes", "n8n-nodes-base.scheduleTrigger", 1.2,
    {"rule": {"interval": [{"field": "minutes", "minutesInterval": 15}]}})

# 2 - the only node you edit: ids and model
code_node("Config", CONFIG_JS)

# 3 - read the whole manifest tab
add("Read Manifest", HTTP, 4.2, dict(GOOGLE_AUTH, **{
    "method": "GET",
    "url": "=https://sheets.googleapis.com/v4/spreadsheets/"
           "{{ $json.manifestSheetId }}/values/sessions",
    "options": {},
}))

# 4 - choose one session to post (or stop)
code_node("Pick Row To Post", PICK_ROW_JS)

# 5 - claim the row: post_status=posting, post_started_at=now  (columns J:K)
add("Claim Row", HTTP, 4.2, dict(GOOGLE_AUTH, **{
    "method": "PUT",
    "url": "=https://sheets.googleapis.com/v4/spreadsheets/"
           "{{ $('Config').first().json.manifestSheetId }}"
           "/values/sessions!J{{ $json._row }}:K{{ $json._row }}",
    "sendQuery": True,
    "queryParameters": {"parameters": [
        {"name": "valueInputOption", "value": "RAW"}]},
    "sendBody": True,
    "specifyBody": "json",
    "jsonBody": "={{ JSON.stringify({ values: [[ 'posting', $now.toISO() ]] }) }}",
    "options": {},
}))

# 6 - official classification from OpenF1
add("OpenF1 Result", HTTP, 4.2, {
    "method": "GET",
    "url": "=https://api.openf1.org/v1/session_result?session_key="
           "{{ $('Pick Row To Post').first().json.openf1_session_key }}",
    "options": {},
})

# 7 - driver names/teams from OpenF1
add("OpenF1 Drivers", HTTP, 4.2, {
    "method": "GET",
    "url": "=https://api.openf1.org/v1/drivers?session_key="
           "{{ $('Pick Row To Post').first().json.openf1_session_key }}",
    "options": {},
})

# 8 - assemble the results summary + the Gemini request body
code_node("Build Caption Request", BUILD_CAPTION_REQUEST_JS)

# 9 - Gemini writes the caption (Google Search grounding on)
add("Generate Caption", HTTP, 4.2, dict(HEADER_AUTH, **{
    "method": "POST",
    "url": "=https://generativelanguage.googleapis.com/v1beta/models/"
           "{{ $('Config').first().json.geminiModel }}:generateContent",
    "sendBody": True,
    "specifyBody": "json",
    "jsonBody": "={{ JSON.stringify($json.geminiBody) }}",
    "options": {},
}))

# 10 - parse Gemini's JSON output
code_node("Parse Caption", PARSE_CAPTION_JS)

# 11 - ask Zernio for an upload URL
add("Zernio Presign", HTTP, 4.2, dict(HEADER_AUTH, **{
    "method": "POST",
    "url": "https://zernio.com/api/v1/media/presign",
    "sendBody": True,
    "specifyBody": "json",
    "jsonBody": "={{ JSON.stringify({ fileName: $json.sessionId + '.mp4', "
                "fileType: 'video/mp4' }) }}",
    "options": {},
}))

# 12 - download the rendered video from Google Drive (binary)
add("Download Video", HTTP, 4.2, dict(DRIVE_AUTH, **{
    "method": "GET",
    "url": "=https://www.googleapis.com/drive/v3/files/"
           "{{ $('Parse Caption').first().json.driveFileId }}?alt=media",
    "options": {"response": {"response": {"responseFormat": "file"}}},
}))

# 13 - upload the video bytes to Zernio's presigned URL (no auth; url is signed)
add("Zernio Upload", HTTP, 4.2, {
    "method": "PUT",
    "url": "={{ $('Zernio Presign').first().json.uploadUrl }}",
    "sendHeaders": True,
    "headerParameters": {"parameters": [
        {"name": "Content-Type", "value": "video/mp4"}]},
    "sendBody": True,
    "contentType": "binaryData",
    "inputDataFieldName": "data",
    "options": {},
})

# 14 - create the post on Instagram + YouTube
add("Zernio Create Post", HTTP, 4.2, dict(HEADER_AUTH, **{
    "method": "POST",
    "url": "https://zernio.com/api/v1/posts",
    "sendBody": True,
    "specifyBody": "json",
    "jsonBody": "={{ JSON.stringify({"
                " content: $('Parse Caption').first().json.instagramCaption,"
                " mediaItems: [{ url: $('Zernio Presign').first().json.publicUrl,"
                " type: 'video' }],"
                " platforms: ["
                "  { platform: 'instagram',"
                "    accountId: $('Config').first().json.zernioInstagramAccountId,"
                "    platformSpecificData: { contentType: 'reels', shareToFeed: true } },"
                "  { platform: 'youtube',"
                "    accountId: $('Config').first().json.zernioYoutubeAccountId,"
                "    platformSpecificData: {"
                "      title: $('Parse Caption').first().json.youtubeTitle,"
                "      visibility: 'public' } }"
                " ],"
                " publishNow: true"
                "}) }}",
    "options": {},
}))

# 15 - mark the manifest row posted (columns J:O)
add("Mark Posted", HTTP, 4.2, dict(GOOGLE_AUTH, **{
    "method": "PUT",
    "url": "=https://sheets.googleapis.com/v4/spreadsheets/"
           "{{ $('Config').first().json.manifestSheetId }}"
           "/values/sessions!J{{ $('Parse Caption').first().json.rowNumber }}"
           ":O{{ $('Parse Caption').first().json.rowNumber }}",
    "sendQuery": True,
    "queryParameters": {"parameters": [
        {"name": "valueInputOption", "value": "RAW"}]},
    "sendBody": True,
    "specifyBody": "json",
    "jsonBody": "={{ JSON.stringify({ values: [[ 'posted', $now.toISO(),"
                " $now.toISO(),"
                " ($('Zernio Create Post').first().json.id ||"
                " $('Zernio Create Post').first().json._id || ''),"
                " '0', '' ]] }) }}",
    "options": {},
}))

# linear connections: each node feeds the next
connections = {}
for i in range(len(order) - 1):
    connections[order[i]] = {
        "main": [[{"node": order[i + 1], "type": "main", "index": 0}]]
    }

workflow = {
    "name": "Formulytics Poster",
    "nodes": nodes,
    "connections": connections,
    "active": False,
    "settings": {"executionOrder": "v1"},
    "pinData": {},
}

with open("formulytics_poster.json", "w", encoding="utf-8", newline="\n") as f:
    json.dump(workflow, f, indent=2)

# validate the file we just wrote
with open("formulytics_poster.json", encoding="utf-8") as f:
    reloaded = json.load(f)
print(f"OK: formulytics_poster.json written - valid JSON, "
      f"{len(reloaded['nodes'])} nodes, {len(reloaded['connections'])} connections.")
