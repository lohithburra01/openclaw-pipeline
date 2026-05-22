# N8N Workflow Build V2 — Formulytics Posting Pipeline

## What You Are Building

A fully automated n8n workflow that:
1. Triggers when a new MP4 file appears in Google Drive (event-driven, not polling)
2. Skips files starting with TEST_ (test renders must never post publicly)
3. Parses the filename to extract GP location, year, session type
4. Calls OpenF1 API to get real race results (winner, podium, gaps, fastest lap)
5. Calls Gemini 3.5 Flash with Google Search grounding to write a caption using that real data
6. Posts the video to Instagram and YouTube via Zernio API
7. Never double-posts — uses Google Drive custom properties to mark files as posted

## Output File

Produce a single file: `formulytics_n8n_workflow.json`
Valid n8n workflow JSON, importable via n8n UI → top-left menu → Import from file.
Workflow name: `Formulytics Posting Pipeline`

---

## Credentials (Added Manually in n8n UI — Never in the JSON)

Before importing the workflow, create these in n8n:
n8n → top-left menu → Credentials → New Credential

### Credential 1 — Google Drive OAuth2
- Type: Google Drive OAuth2 API
- Name it exactly: `Formulytics Google Drive`
- Sign in with the Google account that owns the formulytics-output folder
- Grant Drive access when prompted

### Credential 2 — Gemini (HTTP Header Auth)
- Type: Header Auth
- Name it exactly: `Formulytics Gemini`
- Name field: `x-goog-api-key`
- Value field: (user will paste their Gemini API key here)

### Credential 3 — Zernio (HTTP Header Auth)
- Type: Header Auth
- Name it exactly: `Formulytics Zernio`
- Name field: `Authorization`
- Value field: `Bearer ` followed by their Zernio API key (user pastes this)

---

## Constants

- Google Drive folder ID: `1dQ5d6ZhFtf27cVtpuLqtcYW5Xzpgp6my`
- Zernio Instagram account ID: `6a103559520992756d9bdd85`
- Zernio YouTube account ID: `6a1035ed520992756d9be08e`
- Gemini endpoint: `https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:generateContent`
- Zernio post endpoint: `https://zernio.com/api/v1/posts`
- OpenF1 API base: `https://api.openf1.org/v1`

---

## Filename Format Contract

The video producer drops files in this exact format:
`{location}_{year}_{session}.mp4`

Examples:
- `monaco_2026_R.mp4` — Grand Prix Race
- `monaco_2026_S.mp4` — Sprint Race
- `montreal_2026_Q.mp4` — Qualifying
- `las_vegas_2026_R.mp4` — multi-word location
- `TEST_miami_2026_R.mp4` — test render, must be skipped entirely

Session codes:
- `R` = Race
- `S` = Sprint
- `Q` = Qualifying

---

## Already-Posted Tracking

Use Google Drive custom file properties to mark posted videos.
After successfully posting, update the Drive file's properties to add:
`{ "posted": "true" }`

Before posting, check if the file already has `posted: true` in its properties.
This replaces the posted.txt approach entirely — one node instead of nine.

---

## Node Definitions

### Node 1 — Google Drive Trigger
- Type: `n8n-nodes-base.googleDriveTrigger`
- Name: `New Video in Drive`
- Event: File Created
- Folder: `1dQ5d6ZhFtf27cVtpuLqtcYW5Xzpgp6my`
- Credential: `Formulytics Google Drive`
- This fires once each time a new file appears in the folder

### Node 2 — Code Node: Validate Filename
- Type: `n8n-nodes-base.code`
- Name: `Validate File`
- Language: JavaScript
- Logic:
  - Get the filename from the trigger
  - If filename starts with `TEST_` → return `{ skip: true }`
  - If filename does not end with `.mp4` → return `{ skip: true }`
  - Otherwise parse the filename and return all fields

```javascript
const filename = $input.first().json.name;
const fileId = $input.first().json.id;

// Skip test files
if (filename.startsWith('TEST_')) {
  return [{ json: { skip: true, reason: 'Test file', filename } }];
}

// Skip non-mp4
if (!filename.endsWith('.mp4')) {
  return [{ json: { skip: true, reason: 'Not an MP4', filename } }];
}

// Parse filename: {location}_{year}_{session}.mp4
const base = filename.replace('.mp4', '');
const parts = base.split('_');
const session = parts[parts.length - 1];       // R, S, or Q
const year = parts[parts.length - 2];           // 2026
const locationParts = parts.slice(0, -2);       // ["las", "vegas"] or ["monaco"]
const location = locationParts.join('_');        // las_vegas or monaco
const locationDisplay = locationParts.join(' '); // las vegas or monaco (for display)

// Clean hashtag: no spaces, no underscores
const locationTag = locationParts.join('');      // lasvegas or monaco

const sessionLabel = session === 'R' ? 'Race'
  : session === 'S' ? 'Sprint'
  : session === 'Q' ? 'Qualifying'
  : 'Session';

// GP name mapping for proper hashtags
// Common circuits where location != GP name
const gpNameMap = {
  'silverstone': 'british',
  'spa': 'belgian',
  'monza': 'italian',
  'interlagos': 'brazilian',
  'saopaulo': 'brazilian',
  'villeneuve': 'canadian',
  'montreal': 'canadian',
  'redbullring': 'austrian',
  'spielberg': 'austrian',
  'hungaroring': 'hungarian',
  'budapest': 'hungarian',
  'zandvoort': 'dutch',
  'marina_bay': 'singapore',
  'marinabay': 'singapore',
  'yas_marina': 'abudhabi',
  'yasmarina': 'abudhabi',
  'albert_park': 'australian',
  'albertpark': 'australian',
  'bahrain': 'bahrain',
  'jeddah': 'saudi',
  'shanghai': 'chinese',
  'baku': 'azerbaijan',
  'miami': 'miami',
  'monaco': 'monaco',
  'barcelona': 'spanish',
  'suzuka': 'japanese',
  'losangeles': 'usgp',
  'austin': 'usgp',
  'cota': 'usgp',
  'mexico': 'mexican',
  'mexicocity': 'mexican',
  'las_vegas': 'lasvegas',
  'lasvegas': 'lasvegas',
  'qatar': 'qatar',
  'lusail': 'qatar',
};

const gpTag = gpNameMap[location.toLowerCase()] || locationTag;

return [{
  json: {
    skip: false,
    fileId,
    filename,
    location,
    locationDisplay,
    locationTag,
    gpTag,
    year,
    session,
    sessionLabel
  }
}];
```

### Node 3 — IF Node: Skip Invalid Files
- Type: `n8n-nodes-base.if`
- Name: `Should Skip?`
- Condition: `{{ $json.skip }}` equals `true`
- True branch → Stop and Do Nothing (workflow ends cleanly)
- False branch → continue to Node 4

### Node 4 — HTTP Request: Check if Already Posted
- Type: `n8n-nodes-base.httpRequest`
- Name: `Check Posted Status`
- Method: GET
- URL: `https://www.googleapis.com/drive/v3/files/{{ $json.fileId }}?fields=properties`
- Authentication: OAuth2 → `Formulytics Google Drive`
- This returns the file's custom properties

### Node 5 — IF Node: Already Posted?
- Type: `n8n-nodes-base.if`
- Name: `Already Posted?`
- Condition: `{{ $json.properties.posted }}` equals `true`
- True branch → Stop and Do Nothing
- False branch → continue to Node 6

### Node 6 — HTTP Request: Get OpenF1 Session Key
- Type: `n8n-nodes-base.httpRequest`
- Name: `Get OpenF1 Session`
- Method: GET
- URL: `https://api.openf1.org/v1/sessions?year={{ $('Validate File').first().json.year }}&session_type={{ $('Validate File').first().json.session === 'R' ? 'Race' : $('Validate File').first().json.session === 'S' ? 'Sprint' : 'Qualifying' }}`
- No auth required
- Returns array of sessions — take the last one (most recent matching session)

### Node 7 — Code Node: Extract Session Key and Match Location
- Type: `n8n-nodes-base.code`
- Name: `Parse OpenF1 Session`
- Language: JavaScript

```javascript
const sessions = $input.first().json;
const location = $('Validate File').first().json.locationDisplay;

// Find session matching our location (partial match, case insensitive)
let match = null;
if (Array.isArray(sessions)) {
  match = sessions.find(s =>
    s.location && s.location.toLowerCase().includes(location.split(' ')[0].toLowerCase())
  );
  if (!match) match = sessions[sessions.length - 1]; // fallback to last session
}

if (!match) {
  return [{ json: { sessionKey: null, gpName: location, error: 'No session found' } }];
}

return [{
  json: {
    sessionKey: match.session_key,
    gpName: match.meeting_name || location,
    circuitName: match.circuit_short_name || location,
    countryName: match.country_name || '',
    sessionType: match.session_type
  }
}];
```

### Node 8 — HTTP Request: Get Race Results from OpenF1
- Type: `n8n-nodes-base.httpRequest`
- Name: `Get Race Results`
- Method: GET
- URL: `https://api.openf1.org/v1/position?session_key={{ $json.sessionKey }}`
- No auth required
- Returns position data for the session

### Node 9 — HTTP Request: Get Driver Info from OpenF1
- Type: `n8n-nodes-base.httpRequest`
- Name: `Get Drivers`
- Method: GET
- URL: `https://api.openf1.org/v1/drivers?session_key={{ $('Parse OpenF1 Session').first().json.sessionKey }}`
- No auth required

### Node 10 — Code Node: Build Race Summary
- Type: `n8n-nodes-base.code`
- Name: `Build Race Summary`
- Language: JavaScript

```javascript
const positions = $('Get Race Results').first().json;
const drivers = $('Get Drivers').first().json;
const sessionInfo = $('Parse OpenF1 Session').first().json;
const fileInfo = $('Validate File').first().json;

// Build driver lookup by driver_number
const driverMap = {};
if (Array.isArray(drivers)) {
  drivers.forEach(d => {
    driverMap[d.driver_number] = {
      name: d.full_name || d.last_name || `Driver ${d.driver_number}`,
      abbr: d.name_acronym || '',
      team: d.team_name || ''
    };
  });
}

// Get final positions (last entry per driver)
const finalPositions = {};
if (Array.isArray(positions)) {
  positions.forEach(p => {
    finalPositions[p.driver_number] = p.position;
  });
}

// Sort to get podium
const podium = Object.entries(finalPositions)
  .sort((a, b) => a[1] - b[1])
  .slice(0, 3)
  .map(([num, pos]) => ({
    position: pos,
    driver: driverMap[num] || { name: `Driver ${num}`, abbr: '', team: '' },
    driverNumber: num
  }));

const p1 = podium[0] ? podium[0].driver.name : 'Unknown';
const p2 = podium[1] ? podium[1].driver.name : 'Unknown';
const p3 = podium[2] ? podium[2].driver.name : 'Unknown';
const p1Team = podium[0] ? podium[0].driver.team : '';
const p2Team = podium[1] ? podium[1].driver.team : '';

const raceSummary = `GP: ${sessionInfo.gpName} ${fileInfo.year} ${fileInfo.sessionLabel}
P1: ${p1} (${p1Team})
P2: ${p2} (${p2Team})
P3: ${p3 || 'Unknown'}
Session type: ${fileInfo.sessionLabel}`;

return [{
  json: {
    raceSummary,
    p1, p2, p3,
    p1Team, p2Team,
    gpName: sessionInfo.gpName,
    sessionKey: sessionInfo.sessionKey,
    // Pass through file info
    fileId: fileInfo.fileId,
    filename: fileInfo.filename,
    location: fileInfo.location,
    locationDisplay: fileInfo.locationDisplay,
    gpTag: fileInfo.gpTag,
    year: fileInfo.year,
    session: fileInfo.session,
    sessionLabel: fileInfo.sessionLabel
  }
}];
```

### Node 11 — HTTP Request: Generate Caption with Gemini
- Type: `n8n-nodes-base.httpRequest`
- Name: `Generate Caption`
- Method: POST
- URL: `https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:generateContent`
- Authentication: Predefined Credential Type → Header Auth → `Formulytics Gemini`
- Body type: JSON
- Body: JSON with the full Formulytics caption system prompt (see CAPTION_SYSTEM_PROMPT section below)

### Node 12 — Code Node: Extract Caption and Build Title
- Type: `n8n-nodes-base.code`
- Name: `Extract Caption`
- Language: JavaScript

```javascript
const response = $input.first().json;
const parts = response.candidates[0].content.parts;
const fullText = parts.find(p => p.text).text.trim();

// Parse the two-section output from Gemini
const igMatch = fullText.match(/INSTAGRAM REELS\s*\n([\s\S]*?)(?=YOUTUBE SHORTS|$)/i);
const ytMatch = fullText.match(/YOUTUBE SHORTS\s*\n([\s\S]*)/i);

const instagramCaption = igMatch ? igMatch[1].trim() : fullText;
const youtubeBlock = ytMatch ? ytMatch[1].trim() : '';

// First line of YouTube block is the title, rest are hashtags
const youtubeLines = youtubeBlock.split('\n').map(l => l.trim()).filter(l => l.length > 0);
const youtubeTitleRaw = youtubeLines[0] || '';
const youtubeTitle = youtubeTitleRaw.length > 95 ? youtubeTitleRaw.substring(0, 92) + '...' : youtubeTitleRaw;
const youtubeHashtags = youtubeLines.slice(1).join(' ');

const summary = $('Build Race Summary').first().json;

return [{
  json: {
    instagramCaption,
    youtubeTitle,
    youtubeHashtags,
    fileId: summary.fileId,
    filename: summary.filename,
    sessionKey: summary.sessionKey
  }
}];
```

### Node 13 — Google Drive: Download Video
- Type: `n8n-nodes-base.googleDrive`
- Name: `Download Video`
- Operation: download
- File ID: `{{ $json.fileId }}`
- Credential: `Formulytics Google Drive`
- IMPORTANT: This node must be placed immediately before the post node so binary data is still in memory

### Node 14 — HTTP Request: Post to Zernio
- Type: `n8n-nodes-base.httpRequest`
- Name: `Post to Instagram and YouTube`
- Method: POST
- URL: `https://zernio.com/api/v1/posts`
- Authentication: Predefined Credential Type → Header Auth → `Formulytics Zernio`
- Body type: Multipart Form Data
- Fields:
  - `content` = `{{ $('Extract Caption').first().json.instagramCaption }}` (string)
  - `platforms[0][platform]` = `instagram` (string)
  - `platforms[0][accountId]` = `6a103559520992756d9bdd85` (string)
  - `platforms[1][platform]` = `youtube` (string)
  - `platforms[1][accountId]` = `6a1035ed520992756d9be08e` (string)
  - `platforms[1][title]` = `{{ $('Extract Caption').first().json.youtubeTitle }}` (string)
  - `media` = binary data from Node 13 Download Video (set field type to File/Binary)

### Node 15 — HTTP Request: Mark as Posted in Drive
- Type: `n8n-nodes-base.httpRequest`
- Name: `Mark as Posted`
- Method: PATCH
- URL: `https://www.googleapis.com/drive/v3/files/{{ $('Extract Caption').first().json.fileId }}`
- Authentication: OAuth2 → `Formulytics Google Drive`
- Body type: JSON
- Body:
```json
{
  "properties": {
    "posted": "true",
    "postedAt": "{{ new Date().toISOString() }}"
  }
}
```
- This node only runs after a successful post from Node 14
- If Node 14 fails, this node does not run — so the video will be retried next time

---

## Node Connection Map

```
New Video in Drive
  → Validate File
  → Should Skip? (IF)
      TRUE  → Stop
      FALSE → Check Posted Status
                → Already Posted? (IF)
                    TRUE  → Stop
                    FALSE → Get OpenF1 Session
                              → Parse OpenF1 Session
                                → Get Race Results
                                → Get Drivers
                              → Build Race Summary
                                → Generate Caption
                                  → Extract Caption
                                    → Download Video
                                      → Post to Instagram and YouTube
                                        → Mark as Posted
```


---

## CAPTION_SYSTEM_PROMPT (Used in Node 11 Gemini Request Body)

This is the full JSON body for the Generate Caption HTTP request node.
The prompt encodes the complete Formulytics brand voice and caption format.

```json
{
  "tools": [{"google_search": {}}],
  "contents": [{
    "parts": [{
      "text": "You are writing captions for Formulytics (@formulytics on Instagram), an F1 data visualization account. The tone is fan-to-fan. Not corporate. Raw, opinionated, and direct.\n\nConfirmed results from official F1 timing:\n{{ $(\'Build Race Summary\').first().json.raceSummary }}\n\nSession: {{ $(\'Build Race Summary\').first().json.gpName }} {{ $(\'Build Race Summary\').first().json.year }} {{ $(\'Build Race Summary\').first().json.sessionLabel }}\n\nSTEP 1 — RESEARCH\nUse Google Search to find:\n- Full classification P1 to P10 minimum\n- DNFs, DSQs, penalties\n- Key moments: battles, crashes, safety cars, strategy calls\n- Driver quotes if available\n- Championship implications\n- Any records or historic firsts\nAlways search. Results change after penalties and DSQs.\n\nSTEP 2 — ONE PRIMARY STORY\nPriority order:\n1. Historic or record-breaking moment\n2. Dominant or shock result\n3. Title fight implications\n4. Chaos, crashes, strategy disasters\n5. Underdog story\n\nSTEP 3 — INSTAGRAM REELS CAPTION\nExact structure:\n\n[punchy hook with attitude + single emoji]\n\n[facts. dots. no punctuation. 2-3 lines.]\n\n[one narrative thread. 2-3 lines. conversational.]\n\n[question to audience that invites debate]\n\n#f1 #formula1 #{{ $(\'Build Race Summary\').first().json.gpTag }}GP #[main subject driver or team] #f1reels\n\nRules:\n- All lowercase always\n- Short punchy sentences. Dots instead of commas\n- No em dashes. Never use: incredible, amazing, stunning, unbelievable, historic\n- Hook is opinionated and takes a side\n- Facts block: names, positions, gaps, what happened\n- Narrative block: one story in 2-3 lines\n- End with a debate question\n- Max 5 hashtags. Always end #f1reels\n- Fourth hashtag = main subject (driver or team e.g. #landonorris #ferrari)\n- EventGP examples: #AusGP #MiamiGP #MonacoGP #JapaneseGP #CanadianGP\n- Never write as paragraphs, always line breaks\n\nSession tone:\n- Race (R): winner or biggest story, top 5, DNFs, one dramatic moment, championship question\n- Sprint (S): fast and punchy, keep it tight\n- Qualifying (Q): pole or shock result, top 5 grid, one session story\n\nSTEP 4 — YOUTUBE SHORTS TITLE\nFormat: [Race Name] [Year] [Session] [Flag Emoji] [Main Story Title Case] | F1 [Year]\nRules:\n- Race name and year first\n- Flag emoji after session\n- Title case, keyword-rich, for search\n- Max 95 characters\n- Pipe before F1 year\n\nOUTPUT — exactly this format, nothing else:\n\nINSTAGRAM REELS\n[caption]\n\nYOUTUBE SHORTS\n[title]\n#formula1 #f1 #shorts"
    }]
  }]
}
```

---
## Critical Implementation Notes

1. **Binary data must not pass through text nodes.** Download Video (Node 13) must connect directly to Post to Zernio (Node 14). The caption text is pulled by name from Extract Caption using `$('Extract Caption').first().json.caption` — the binary data flows separately.

2. **Node 8 and Node 9 run in parallel** from Node 7 — both need the session key from Parse OpenF1 Session. Node 10 waits for both and merges using node names.

3. **YouTube title max 95 characters** — always use `youtubeTitle` for the YouTube platform title field, never the full caption.

4. **TEST_ files must never reach the post node** — the Should Skip IF node handles this at Node 3, before any API calls are made.

5. **The gpTag in hashtags must have no spaces** — `#las vegasgp` breaks the hashtag. The locationTag and gpNameMap in Node 2 handle this. Always use `gpTag` for the hashtag, never `locationDisplay`.

6. **Mark as Posted only runs on success** — connect Node 15 only to the success output of Node 14. n8n error branch from Node 14 should go to a Stop node so failed posts are retried next time the Drive trigger fires (which it won't unless a new file appears — acceptable for this use case).

7. **OpenF1 data may not be available immediately** — if Node 6 returns empty results, Node 10 should fall back to building a summary with just the filename data and let Gemini use Google Search to fill in the gaps.

8. **All node names must match exactly** — downstream nodes reference upstream nodes by name using `$('Node Name').first().json`.

9. **No API keys in the JSON file** — all authentication uses credential names. The user adds the actual keys inside n8n credentials UI.

10. **Zernio API format** — verify against https://zernio.com/api/v1 docs before finalising the post node body structure. The platforms array format shown above is assumed REST — adjust if their actual schema differs.
