"""Capture the README screenshots from a running app.

    python run.py                        # in one terminal
    python docs/capture_screenshots.py   # in another

Drives headless Chrome over the DevTools protocol rather than using
`chrome --screenshot`, because the one-shot flag fires as soon as the HTML
loads - long before Streamlit has finished streaming a Gemini answer. Here we
wait for a marker string to appear in the page, then capture.

Images are written to docs/screenshots/.
"""
from __future__ import annotations

import base64
import json
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "screenshots"
PORT = 9333
BASE = "http://localhost:8530"

CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "google-chrome",
    "chromium",
]

# (filename, url, text marker, viewport height, JS that must return truthy)
PLOT_PAINTED = "document.querySelectorAll('.js-plotly-plot .main-svg').length > 0"
TABLE_FILLED = (
    "document.querySelectorAll('[data-testid=\"stDataFrame\"] [role=\"row\"]').length > 1"
)

SHOTS = [
    (
        "ask-answer.png",
        BASE + "/?q=" + urllib.parse.quote("What are the main threats to the Amazon rainforest?"),
        "Why Are They Close?",
        2500,
        PLOT_PAINTED,
    ),
    (
        "dashboard.png",
        BASE + "/",
        "Indexed documents",
        1250,
        TABLE_FILLED,
    ),
    (
        "arabic-rtl.png",
        BASE + "/?lang=ar",
        "لوحة التحكم",
        1250,
        TABLE_FILLED,
    ),
]


def find_chrome() -> str:
    for candidate in CHROME_CANDIDATES:
        if Path(candidate).exists():
            return candidate
        found = shutil.which(candidate)
        if found:
            return found
    raise SystemExit("Chrome or Edge not found - install one, or edit CHROME_CANDIDATES.")


class CDP:
    """A very small DevTools-protocol client."""

    def __init__(self, ws):
        self.ws = ws
        self.n = 0

    def send(self, method: str, **params):
        self.n += 1
        self.ws.send(json.dumps({"id": self.n, "method": method, "params": params}))
        while True:
            message = json.loads(self.ws.recv())
            if message.get("id") == self.n:
                if "error" in message:
                    raise RuntimeError(method + ": " + str(message["error"]))
                return message.get("result", {})

    def text(self) -> str:
        result = self.send(
            "Runtime.evaluate",
            expression="document.body ? document.body.innerText : ''",
            returnByValue=True,
        )
        return result.get("result", {}).get("value", "") or ""

    def truthy(self, expression: str) -> bool:
        """Charts and tables paint after their text appears - wait for them."""
        if not expression:
            return True
        result = self.send(
            "Runtime.evaluate", expression=expression, returnByValue=True
        )
        return bool(result.get("result", {}).get("value"))


def _trim_tail(path: Path, pad: int = 40) -> None:
    """Crop the empty page tail so the image is not mostly blank."""
    try:
        from PIL import Image, ImageChops
    except ImportError:
        return
    image = Image.open(path).convert("RGB")
    width, height = image.size
    # ignore the sidebar column when looking for the last row of content
    content = image.crop((int(width * 0.22), 0, width, height))
    blank = Image.new("RGB", content.size, content.getpixel((content.size[0] - 5, 5)))
    box = ImageChops.difference(content, blank).getbbox()
    if box and box[3] + pad < height:
        image.crop((0, 0, width, box[3] + pad)).save(path)


def capture(chrome: str) -> list:
    from websockets.sync.client import connect

    OUT.mkdir(parents=True, exist_ok=True)
    profile = ROOT / ".chrome-profile"
    process = subprocess.Popen(
        [
            chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars",
            "--remote-debugging-port=" + str(PORT),
            "--user-data-dir=" + str(profile),
            "--no-first-run", "--no-default-browser-check",
            "--force-device-scale-factor=2",      # retina-sharp output
            "about:blank",
        ],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    written = []
    try:
        endpoint = None
        for _ in range(60):
            try:
                with urllib.request.urlopen(
                    "http://localhost:" + str(PORT) + "/json/version", timeout=1
                ) as response:
                    endpoint = json.load(response)["webSocketDebuggerUrl"]
                break
            except Exception:
                time.sleep(0.5)
        if not endpoint:
            raise SystemExit("Chrome did not expose a debugging endpoint.")

        for name, url, marker, height, ready_js in SHOTS:
            print("capturing", name, "...", flush=True)
            # current Chrome requires PUT on /json/new
            request = urllib.request.Request(
                "http://localhost:" + str(PORT)
                + "/json/new?" + urllib.parse.quote(url, safe=":/?=&%"),
                method="PUT",
            )
            with urllib.request.urlopen(request, timeout=10) as response:
                target = json.load(response)

            with connect(target["webSocketDebuggerUrl"], max_size=80 * 1024 * 1024) as ws:
                cdp = CDP(ws)
                cdp.send("Page.enable")
                cdp.send("Runtime.enable")
                cdp.send(
                    "Emulation.setDeviceMetricsOverride",
                    width=1500, height=height, deviceScaleFactor=2, mobile=False,
                )
                # light theme reads better on a README in either GitHub mode
                cdp.send("Emulation.setEmulatedMedia",
                         features=[{"name": "prefers-color-scheme", "value": "light"}])

                deadline = time.time() + 240
                while time.time() < deadline:
                    body = cdp.text()
                    ready = marker in body and "Loading the embedding model" not in body
                    if ready and cdp.truthy(ready_js):
                        break
                    time.sleep(1.5)
                else:
                    print("  ! timed out waiting for:", marker)

                time.sleep(4)   # let fonts and chart animations settle
                shot = cdp.send("Page.captureScreenshot", format="png")
                path = OUT / name
                path.write_bytes(base64.b64decode(shot["data"]))
                _trim_tail(path)
                written.append(path)
                print("  ->", path.name, round(path.stat().st_size / 1024), "KB")
    finally:
        process.terminate()
        shutil.rmtree(ROOT / ".chrome-profile", ignore_errors=True)
    return written


def main() -> int:
    try:
        urllib.request.urlopen(BASE, timeout=5)
    except Exception:
        print("The app is not running at " + BASE + ". Start it with: python run.py")
        return 1
    capture(find_chrome())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
