"""tools/strata_menubar.py - a macOS menu bar icon for Strata (Apple Silicon / MLX backend).

Shows a small icon in the menu bar and, without a terminal, lets you: see whether the model is loaded, open the web
app, and start/stop/restart the server.  It talks to the server over HTTP (/health) and starts it with the same
command the run script uses; it never imports the model itself, so it stays small.

    .venv/bin/python tools/strata_menubar.py        (or double-click strata-menubar.command)

Needs `rumps` (pip install rumps), installed by setup-macos.sh.  Add strata-menubar.command to Login Items for the
icon at every login.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PORT = int(os.environ.get("STRATA_PORT", "8080"))
CONFIG = os.environ.get("STRATA_CONFIG", "strata-qwen25-7b-1m.json")
BASE = f"http://127.0.0.1:{PORT}"
LOG = ROOT / f"{Path(CONFIG).stem}.log"
STATUS = ROOT / "strata-status.json"      # the server writes loading -> warming -> ready while it starts

try:
    import rumps
except ImportError:                                  # a clear message instead of a traceback
    print("Strata menu bar needs rumps: .venv/bin/pip install rumps  (or run ./setup-macos.sh)", file=sys.stderr)
    raise SystemExit(1)

APP_SUPPORT = Path.home() / "Library" / "Application Support" / "Strata"


def _icon_file():
    """A template menu-bar icon (three stacked bars = strata), drawn once with Pillow.  Template images are
    black+alpha, so macOS tints them for light/dark.  None (a text title) when Pillow is missing."""
    path = APP_SUPPORT / "menubar.png"
    if path.exists():
        return str(path)
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return None
    APP_SUPPORT.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGBA", (44, 44), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    for y, w in ((9, 26), (20, 20), (31, 14)):
        x0 = (44 - w) // 2
        d.rounded_rectangle([x0, y, x0 + w, y + 5], radius=2, fill=(0, 0, 0, 255))
    img.save(path)
    return str(path)


def _health(timeout=2.0):
    """The server's /health as a dict, or None when it is not answering."""
    try:
        with urllib.request.urlopen(f"{BASE}/health", timeout=timeout) as r:
            return json.load(r)
    except Exception:
        return None


def _server_pids():
    try:
        out = subprocess.run(["pgrep", "-f", "serve.server --engine mlx"], capture_output=True, text=True).stdout
        return [int(x) for x in out.split()]
    except (OSError, ValueError):
        return []


class StrataMenubar(rumps.App):
    def __init__(self):
        icon = _icon_file()
        self._icon = icon
        try:
            super().__init__("Strata", icon=icon, title=None if icon else "Strata", template=bool(icon),
                             quit_button=None)
        except TypeError:                            # an older rumps without `template`
            super().__init__("Strata", icon=icon, title=None if icon else "Strata", quit_button=None)
        # Build the menu ONCE and only update the status label afterwards: rumps' `menu` setter calls Menu.update(),
        # which ADDS items and never replaces them, so rebuilding it would duplicate every entry on each tick.
        self.state_item = rumps.MenuItem("Strata — starting…")
        self.state_item.set_callback(None)           # a status line, not a command
        self.menu = [self.state_item, None,
                     rumps.MenuItem("Open the web app", callback=self.open_web), None,
                     rumps.MenuItem("Start", callback=self.start),
                     rumps.MenuItem("Stop", callback=self.stop),
                     rumps.MenuItem("Restart", callback=self.restart), None,
                     rumps.MenuItem("Quit", callback=self.quit_app)]
        self.timer = rumps.Timer(self._tick, 3)
        self.timer.start()
        self._tick(None)

    # ---- actions -------------------------------------------------------------------------------------
    def open_web(self, _=None):
        subprocess.run(["open", BASE + "/"], check=False)

    def start(self, _=None):
        if _health():
            return
        with open(LOG, "a", encoding="utf-8") as log:
            subprocess.Popen([sys.executable, "-m", "serve.server", "--engine", "mlx", "--config", CONFIG,
                              "--port", str(PORT)], cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, start_new_session=True)
        try:
            rumps.notification("Strata", "Starting the model", "It will be ready in a few seconds")
        except Exception:
            pass

    def stop(self, _=None):
        for pid in _server_pids():
            try:
                os.kill(pid, 15)                     # SIGTERM: the server releases the model and exits cleanly
            except OSError:
                pass
        self.state_item.title = "Strata — stopped"

    def restart(self, _=None):
        self.stop()
        self.timer.stop()
        self.timer = rumps.Timer(lambda t: (t.stop(), self.start(), self.timer.start()), 2)
        self.timer.start()

    def quit_app(self, _=None):
        self.timer.stop()
        rumps.quit_application()

    # ---- state ---------------------------------------------------------------------------------------
    @staticmethod
    def _phase():
        """The server's startup phase from its status file ("loading", "warming" or "ready"), or None if there is
        no recent one.  It only matters while /health is not answering yet."""
        try:
            data = json.loads(STATUS.read_text(encoding="utf-8"))
            if time.time() - float(data.get("at", 0)) < 600:
                return data.get("phase")
        except (OSError, ValueError):
            pass
        return None

    def _tick(self, _):
        h = _health()
        if h:
            state = f"{h.get('model', '?')} · {int(h.get('max_context', 0)) // 1000}K"
        else:
            state = {"loading": "starting…", "warming": "warming up…"}.get(self._phase() or "", "stopped")
        if not self._icon:
            self.title = f"Strata ({state})" if state == "stopped" else "Strata"
        self.state_item.title = f"Strata — {state}"


if __name__ == "__main__":
    StrataMenubar().run()
