"""Take a real screenshot of the running Streamlit UI through Chrome's DevTools protocol (waits for the
websocket-driven render). Usage: python scripts/12_ui_screenshot.py <ui_url_with_query> <out.png>"""
import sys, json, time, base64, subprocess, urllib.request
from websockets.sync.client import connect
url, out = sys.argv[1], sys.argv[2]
CH = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
proc = subprocess.Popen([CH, "--headless=new", "--disable-gpu", "--remote-debugging-port=9333", "--window-size=1500,1900", "--hide-scrollbars", "about:blank"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    for _ in range(40):
        try:
            tabs = json.load(urllib.request.urlopen("http://127.0.0.1:9333/json")); break
        except Exception:
            time.sleep(0.5)
    ws_url = [t for t in tabs if t["type"] == "page"][0]["webSocketDebuggerUrl"]
    with connect(ws_url, max_size=None) as ws:
        i = 0
        def call(method, **params):
            global i
            i += 1; ws.send(json.dumps({"id": i, "method": method, "params": params}))
            while True:
                m = json.loads(ws.recv())
                if m.get("id") == i: return m.get("result")
        call("Page.enable"); call("Emulation.setDeviceMetricsOverride", width=1500, height=1900, deviceScaleFactor=1, mobile=False)
        call("Page.navigate", url=url); time.sleep(30)
        r = call("Page.captureScreenshot", format="png")
        open(out, "wb").write(base64.b64decode(r["data"])); print("saved", out)
finally:
    proc.terminate()
