"""Local chat client.   python chat.py [session-name]"""
import sys
import requests

BASE = "http://localhost:8000"
session = sys.argv[1] if len(sys.argv) > 1 else "local"

token = None
try:
    token = requests.post(f"{BASE}/session", timeout=10).json()["session_token"]
except Exception as e:
    print("Could not reach the server:", e)
    raise SystemExit(1)

headers = {"X-Session-Token": token}
print(f"session: {session}  (ctrl-c to quit,  /facts  /forget)\n")

while True:
    try:
        q = input("you > ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if not q:
        continue
    if q == "/facts":
        print(requests.get(f"{BASE}/session/state", headers=headers).json(), "\n")
        continue
    if q == "/forget":
        print(requests.post(f"{BASE}/session/forget", headers=headers).json(), "\n")
        continue

    r = requests.post(f"{BASE}/ask", json={"question": q}, headers=headers, timeout=120)
    if r.status_code != 200:
        print("bot > ERROR", r.status_code, r.text, "\n")
        continue
    d = r.json()
    flags = f"[{d['route']}{' DEGRADED' if d.get('degraded') else ''} {d['latency_ms']}ms]"
    print(f"bot > {d['response']}\n      {flags}\n")
