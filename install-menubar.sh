#!/bin/bash
# install-menubar.sh - install/uninstall the Strata menu bar icon as a macOS LaunchAgent.
#
#   ./install-menubar.sh            install and start it (also at every login)
#   ./install-menubar.sh --uninstall
#
# It runs tools/strata_menubar.py with this folder's .venv; the icon then shows in the menu bar and survives
# logout/login.  Quitting it from its own menu stops it until the next login (it is not kept alive).
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"
PY="$ROOT/.venv/bin/python"
LABEL="com.strata.menubar"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
UID_NUM="$(id -u)"

unload() {
  launchctl bootout "gui/$UID_NUM/$LABEL" 2>/dev/null || true
  launchctl unload "$PLIST" 2>/dev/null || true
}

if [ "${1:-}" = "--uninstall" ]; then
  unload
  rm -f "$PLIST"
  pkill -f "tools/strata_menubar.py" 2>/dev/null || true
  echo "Strata menu bar uninstalled."
  exit 0
fi

if [ ! -x "$PY" ]; then
  echo "No .venv here: run ./setup-macos.sh first (it installs mlx-lm and rumps)." >&2
  exit 1
fi
"$PY" -c "import rumps" 2>/dev/null || {
  echo "Installing rumps into .venv ..."
  "$PY" -m pip install -q rumps
}

mkdir -p "$HOME/Library/LaunchAgents"
cat > "$PLIST" <<PLIST_EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>$LABEL</string>
    <key>ProgramArguments</key>
    <array>
        <string>$PY</string>
        <string>$ROOT/tools/strata_menubar.py</string>
    </array>
    <key>WorkingDirectory</key>
    <string>$ROOT</string>
    <key>RunAtLoad</key>
    <true/>
    <key>LimitLoadToSessionType</key>
    <string>Aqua</string>
    <key>StandardOutPath</key>
    <string>/tmp/strata-menubar.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/strata-menubar.log</string>
</dict>
</plist>
PLIST_EOF

unload
# stop any hand-started copy so only the LaunchAgent's icon remains
pkill -f "tools/strata_menubar.py" 2>/dev/null || true
sleep 1

if ! launchctl bootstrap "gui/$UID_NUM" "$PLIST" 2>/dev/null; then
  launchctl load -w "$PLIST"
fi

sleep 2
if pgrep -f "tools/strata_menubar.py" >/dev/null; then
  echo "Strata menu bar installed and running (it will also start at login)."
  echo "Look for the icon (three stacked bars) in the menu bar."
else
  echo "Installed, but the icon is not running; see /tmp/strata-menubar.log" >&2
  exit 1
fi
