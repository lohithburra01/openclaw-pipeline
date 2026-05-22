"""
get_gdrive_token.py
-------------------
One-time helper that generates the three Google Drive values the race-replay
pipeline needs:  GDRIVE_CLIENT_ID, GDRIVE_CLIENT_SECRET, GDRIVE_REFRESH_TOKEN.

How to use:
  1. In Google Cloud Console, create an OAuth client of type "Desktop app"
     and download its JSON file.
  2. Save that JSON next to this script and rename it to:  client_secret.json
  3. Install the one dependency:   pip install google-auth-oauthlib
  4. Run:                          python get_gdrive_token.py
  5. A browser window opens. Sign in with the Google account that owns your
     Drive folder. If you see "Google hasn't verified this app", click
     "Advanced", then "Go to ... (unsafe)", then "Allow".
  6. The script prints three values. Paste each into the matching GitHub
     repository secret (Settings -> Secrets and variables -> Actions).
"""

import json
import os
import sys

CLIENT_FILE = "client_secret.json"
SCOPES = [
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/spreadsheets",
]


def main():
    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError:
        sys.exit("Missing dependency. Run this first:\n"
                 "  pip install google-auth-oauthlib")

    if not os.path.exists(CLIENT_FILE):
        sys.exit(f"'{CLIENT_FILE}' not found.\n"
                 f"Download your OAuth client JSON from Google Cloud Console "
                 f"and save it next to this script as {CLIENT_FILE}.")

    with open(CLIENT_FILE, encoding="utf-8") as f:
        cfg = json.load(f)

    flow = InstalledAppFlow.from_client_config(cfg, SCOPES)
    creds = flow.run_local_server(port=0, prompt="consent", access_type="offline")

    block = cfg.get("installed") or cfg.get("web") or {}

    if not creds.refresh_token:
        sys.exit("No refresh token was returned. Re-run the script and make "
                 "sure you fully complete the approval in the browser.")

    line = "=" * 64
    print("\n" + line)
    print("  SUCCESS - paste each value into the matching GitHub secret")
    print("  (repo Settings -> Secrets and variables -> Actions)")
    print(line)
    print("\nSecret name:  GDRIVE_CLIENT_ID")
    print("Value:        " + block.get("client_id", "(not found)"))
    print("\nSecret name:  GDRIVE_CLIENT_SECRET")
    print("Value:        " + block.get("client_secret", "(not found)"))
    print("\nSecret name:  GDRIVE_REFRESH_TOKEN")
    print("Value:        " + creds.refresh_token)
    print("\n" + line)


if __name__ == "__main__":
    main()
