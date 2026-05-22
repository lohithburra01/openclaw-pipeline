# N8N Workflow Build — Formulytics Posting Pipeline

## What You Are Building

A fully automated n8n workflow that:
1. Runs every 10 minutes on a schedule
2. Checks a Google Drive folder for new MP4 videos
3. Skips videos it has already posted (using a tracking file in Drive)
4. Parses the filename to extract GP location, year, session type
5. Calls Gemini 3.5 Flash with Google Search grounding to look up real race results and write a caption
6. Posts the video + caption to Instagram Reels and YouTube Shorts via upload-post.com
7. Marks the video as posted so it never posts it twice

## Credentials Needed in n8n (You Add These Manually in n8n UI)

Before importing the workflow, create these credentials in n8n:
Go to n8n → top-left menu → Credentials → New Credential

### Credential 1 — Google Drive OAuth2
- Type: Google Drive OAuth2 API
- Name it exactly: `Formulytics Google Drive`
- Sign in with the Google account that owns the formulytics-output folder
- Grant Drive access when prompted

### Credential 2 — Gemini API (HTTP Header Auth)
- Type: Header Auth
- Name it exactly: `Formulytics Gemini`
- Name field: `x-goog-api-key`
- Value field: paste your Gemini API key (`<REDACTED — superseded doc; this key was committed earlier and must be rotated>`)

### Credential 3 — upload-post.com (HTTP Header Auth)
- Type: Header Auth
- Name it exactly: `Formulytics UploadPost`
- Name field: `Authorization`
- Value field: `Apikey <REDACTED — superseded doc; this token was committed earlier and must be rotated>`

Note: the word `Apikey` must be included before the token, exactly as shown above.

---

## Constants You Must Know Before Building

- Google Drive folder ID: `1dQ5d6ZhFtf27cVtpuLqtcYW5Xzpgp6my`
- upload-post.com username: `FORMULYTICS_INSTAGRAM`
- Gemini model: `gemini-3.5-flash`
- Gemini API endpoint: `https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:generateContent`
- upload-post.com endpoint: `https://api.upload-post.com/api/upload`

---

## Tracking Already-Posted Videos

To avoid double-posting, the workflow uses a plain text file called `posted.txt`
stored in the same Google Drive folder. Each line in the file is a filename that
has already been posted, e.g.:

```
monaco_2026_R.mp4
miami_2026_S.mp4
```

If the file does not exist yet, the workflow treats all videos as new and creates
the file after the first post.

---

## Workflow JSON to Build

Build this as a single valid n8n workflow JSON file that can be imported directly
via n8n UI → Import from file.

The workflow name is: `Formulytics Posting Pipeline`

### Node 1 — Schedule Trigger
- Type: `n8n-nodes-base.scheduleTrigger`
- Name: `Every 10 Minutes`
- Runs every 10 minutes, 24/7
- No credentials needed

### Node 2 — Get All Files From Drive Folder
- Type: `n8n-nodes-base.googleDrive`
- Name: `Get Drive Files`
- Operation: getAll (list files in a folder)
- Folder ID: `1dQ5d6ZhFtf27cVtpuLqtcYW5Xzpgp6my`
- Filter: only files where name ends with `.mp4`
- Return all matching files
- Credential: `Formulytics Google Drive`

### Node 3 — Get Posted Tracking File
- Type: `n8n-nodes-base.googleDrive`
- Name: `Get Posted List`
- Operation: getAll
- Folder ID: `1dQ5d6ZhFtf27cVtpuLqtcYW5Xzpgp6my`
- Filter: name equals `posted.txt`
- Credential: `Formulytics Google Drive`
- This may return 0 results if posted.txt does not exist yet — handle that in the next node

### Node 4 — Code Node: Read Posted List
- Type: `n8n-nodes-base.code`
- Name: `Parse Posted List`
- Language: JavaScript
- Logic:
  - If the previous node returned a file, download its content and split by newline to get an array of already-posted filenames
  - If no file was returned, return an empty array
  - Output: `{ postedFiles: ["monaco_2026_R.mp4", ...] }`

```javascript
const files = $input.all();
let postedFiles = [];

if (files.length > 0 && files[0].json.id) {
  // File exists — its content will be fetched in next node
  // Pass the file ID forward
  return [{ json: { postedFileId: files[0].json.id, postedFiles: [] } }];
}

return [{ json: { postedFileId: null, postedFiles: [] } }];
```

### Node 5 — Download posted.txt Content (if exists)
- Type: `n8n-nodes-base.googleDrive`
- Name: `Download Posted List`
- Operation: download
- File ID: taken from previous node's `postedFileId` using expression `{{ $json.postedFileId }}`
- Only runs if `postedFileId` is not null — use an IF node before this to branch
- Credential: `Formulytics Google Drive`

### Node 6 — IF Node: Does posted.txt Exist?
- Type: `n8n-nodes-base.if`
- Name: `Posted File Exists?`
- Condition: `{{ $json.postedFileId }}` is not empty / exists
- True branch → go to Download Posted List (Node 5)
- False branch → skip to Node 8 with empty postedFiles array

### Node 7 — Code Node: Parse Downloaded Content
- Type: `n8n-nodes-base.code`
- Name: `Parse Downloaded Posted List`
- Language: JavaScript
- Takes the binary content from the download node, converts to string, splits by newline

```javascript
const binaryData = $input.first().binary?.data;
let postedFiles = [];

if (binaryData) {
  const content = Buffer.from(binaryData.data, 'base64').toString('utf-8');
  postedFiles = content.split('\n').map(f => f.trim()).filter(f => f.length > 0);
}

return [{ json: { postedFiles } }];
```

### Node 8 — Code Node: Find Unposted Videos
- Type: `n8n-nodes-base.code`
- Name: `Find New Videos`
- Language: JavaScript
- Merges the list of all MP4 files from Node 2 with the postedFiles list
- Filters out already-posted files
- Returns only unposted MP4 files, one item per file

```javascript
// Get all MP4 files from Node 2 (Get Drive Files)
const allFiles = $('Get Drive Files').all().map(f => f.json);

// Get posted list (from either branch)
let postedFiles = [];
try {
  postedFiles = $('Parse Downloaded Posted List').first().json.postedFiles;
} catch(e) {
  postedFiles = [];
}

// Filter to unposted MP4s only
const newFiles = allFiles.filter(f => 
  f.name && f.name.endsWith('.mp4') && !postedFiles.includes(f.name)
);

if (newFiles.length === 0) {
  return [{ json: { hasNewFiles: false } }];
}

// Return first unposted file only (process one per run)
const file = newFiles[0];
const nameParts = file.name.replace('.mp4', '').split('_');
const session = nameParts[nameParts.length - 1];
const year = nameParts[nameParts.length - 2];
const location = nameParts.slice(0, -2).join(' ');
const sessionLabel = session === 'R' ? 'Race' : session === 'S' ? 'Sprint' : 'Qualifying';

return [{
  json: {
    hasNewFiles: true,
    fileId: file.id,
    filename: file.name,
    location,
    year,
    session,
    sessionLabel,
    postedFiles
  }
}];
```

### Node 9 — IF Node: Any New Files?
- Type: `n8n-nodes-base.if`
- Name: `New Files Found?`
- Condition: `{{ $json.hasNewFiles }}` equals `true`
- True branch → continue to caption generation
- False branch → workflow ends (no new videos, try again in 10 min)

### Node 10 — Download the Video File
- Type: `n8n-nodes-base.googleDrive`
- Name: `Download Video`
- Operation: download
- File ID: `{{ $json.fileId }}`
- Credential: `Formulytics Google Drive`
- This downloads the MP4 as binary data for uploading to upload-post.com

### Node 11 — HTTP Request: Generate Caption with Gemini
- Type: `n8n-nodes-base.httpRequest`
- Name: `Generate Caption`
- Method: POST
- URL: `https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:generateContent`
- Authentication: Predefined Credential Type → Header Auth → `Formulytics Gemini`
- Body: JSON
- Body content:

```json
{
  "tools": [{"google_search": {}}],
  "contents": [{
    "parts": [{
      "text": "You are writing a social media caption for the F1 account Formulytics on Instagram Reels and YouTube Shorts.\n\nSession details from filename:\n- GP Location: {{ $('Find New Videos').first().json.location }}\n- Year: {{ $('Find New Videos').first().json.year }}\n- Session type: {{ $('Find New Videos').first().json.sessionLabel }}\n\nUse Google Search to find the real results of this F1 session: winner, full podium, gap between P1 and P2, fastest lap holder, any notable incidents, drama, or controversy.\n\nThen write a caption following ALL of these rules:\n1. First line is a punchy hook — use rivalry framing, drama, or a surprising fact. No generic openers.\n2. 2-3 short lines giving context from the race — specific details, not vague summaries.\n3. One engagement question at the end — ask the audience something specific.\n4. Last line is hashtags only: #f1 #formula1 #{{ $('Find New Videos').first().json.location }}gp #f1reels #shorts\n5. Maximum 150 words total.\n6. No em dashes anywhere.\n7. Write differently for session types: Race = full drama, Sprint = fast and punchy, Qualifying = surprise or pole shock.\n\nReturn ONLY the caption text. No explanations, no preamble, no markdown formatting."
    }]
  }]
}
```

### Node 12 — Code Node: Extract Caption Text
- Type: `n8n-nodes-base.code`
- Name: `Extract Caption`
- Language: JavaScript

```javascript
const response = $input.first().json;
const parts = response.candidates[0].content.parts;
const caption = parts.find(p => p.text).text.trim();

return [{
  json: {
    caption,
    fileId: $('Find New Videos').first().json.fileId,
    filename: $('Find New Videos').first().json.filename,
    location: $('Find New Videos').first().json.location,
    year: $('Find New Videos').first().json.year,
    session: $('Find New Videos').first().json.session,
    postedFiles: $('Find New Videos').first().json.postedFiles
  }
}];
```

### Node 13 — HTTP Request: Post to Instagram + YouTube via upload-post.com
- Type: `n8n-nodes-base.httpRequest`
- Name: `Post Video`
- Method: POST
- URL: `https://api.upload-post.com/api/upload`
- Authentication: Predefined Credential Type → Header Auth → `Formulytics UploadPost`
- Body: Form Data (multipart)
- Fields:
  - `user` = `FORMULYTICS_INSTAGRAM` (string)
  - `title` = `{{ $json.caption }}` (string)
  - `description` = `{{ $json.caption }}` (string)
  - `platform[]` = `instagram` (string)
  - `platform[]` = `youtube` (string)
  - `video` = binary data from Node 10 `Download Video` — set field type to "File/Binary", point to the binary output of that node

### Node 14 — Code Node: Update Posted List
- Type: `n8n-nodes-base.code`
- Name: `Update Posted List`
- Language: JavaScript
- Appends the newly posted filename to the postedFiles array and prepares updated content

```javascript
const filename = $('Extract Caption').first().json.filename;
const postedFiles = $('Extract Caption').first().json.postedFiles || [];

postedFiles.push(filename);
const updatedContent = postedFiles.join('\n');

return [{
  json: {
    updatedContent,
    filename,
    postedFiles
  }
}];
```

### Node 15 — IF Node: Does posted.txt Already Exist?
- Type: `n8n-nodes-base.if`
- Name: `Posted.txt Exists Already?`
- Condition: check if `$('Parse Posted List').first().json.postedFileId` is not null
- True branch → update existing file (Node 16)
- False branch → create new file (Node 17)

### Node 16 — Google Drive: Update posted.txt
- Type: `n8n-nodes-base.googleDrive`
- Name: `Update Posted File`
- Operation: update
- File ID: `{{ $('Parse Posted List').first().json.postedFileId }}`
- Content: `{{ $json.updatedContent }}`
- Credential: `Formulytics Google Drive`

### Node 17 — Google Drive: Create posted.txt
- Type: `n8n-nodes-base.googleDrive`
- Name: `Create Posted File`
- Operation: upload
- File name: `posted.txt`
- Parent folder ID: `1dQ5d6ZhFtf27cVtpuLqtcYW5Xzpgp6my`
- Content: `{{ $json.updatedContent }}`
- Credential: `Formulytics Google Drive`

---

## Node Connection Map

Connect nodes in this exact order:

```
Every 10 Minutes
  → Get Drive Files
  → Get Posted List
  → Parse Posted List
  → Posted File Exists? (IF)
      TRUE  → Download Posted List → Parse Downloaded Posted List → Find New Videos
      FALSE → Find New Videos
  → New Files Found? (IF)
      TRUE  → Download Video → Generate Caption → Extract Caption → Post Video → Update Posted List → Posted.txt Exists Already? (IF)
                                                                                                          TRUE  → Update Posted File
                                                                                                          FALSE → Create Posted File
      FALSE → (end, no action)
```

---

## Output File Format

The workflow JSON file must be named: `formulytics_n8n_workflow.json`

It must be valid n8n workflow JSON that can be imported via:
n8n UI → top left menu → Import from file → select the JSON

---

## Important Implementation Notes

1. The workflow processes ONE video per run. If there are 3 unposted videos, it posts 1 now, the next run posts the second, etc. This avoids rate limits on upload-post.com.

2. Google Drive download nodes return binary data. When passing binary data to the upload-post.com HTTP Request node, make sure the video field is set to type "File" and references the binary output from the Download Video node correctly using n8n's binary data handling.

3. The Gemini request includes `"tools": [{"google_search": {}}]` — this enables real-time Google Search grounding so Gemini looks up actual race results before writing the caption. Do not remove this.

4. All node names must match exactly as written because later nodes reference earlier nodes by name using `$('Node Name').first().json`.

5. The schedule trigger should be set to every 10 minutes with no specific time window — it runs 24/7 and self-corrects via the posted.txt tracking file.

6. Do not hardcode any API keys in the JSON. All authentication goes through n8n credentials referenced by name.

7. If upload-post.com returns an error (e.g. free tier limit hit), the workflow should not update posted.txt — so the video will be retried next run. Handle this by only connecting the Update Posted List node after a successful response from Post Video.
