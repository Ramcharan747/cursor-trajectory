#!/bin/bash
# ═══════════════════════════════════════════════════════
#  CursorCapture — One-Click Installer for macOS
#  Double-click this file to install.
# ═══════════════════════════════════════════════════════

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BINARY_NAME="cursor_capture"
BINARY_PATH="$SCRIPT_DIR/$BINARY_NAME"
INSTALL_DIR="$HOME/Applications/CursorCapture"
PLIST_PATH="$HOME/Library/LaunchAgents/CursorCapture.plist"
DATA_DIR="$HOME/cursor_capture_data"
LOG_FILE="$DATA_DIR/cursor_capture.log"

clear
echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║     CursorCapture — macOS Installer             ║"
echo "╠══════════════════════════════════════════════════╣"
echo "║                                                 ║"
echo "║  This installs a tiny background app that       ║"
echo "║  records cursor movement for research.          ║"
echo "║                                                 ║"
echo "║  • Runs silently in the background              ║"
echo "║  • Auto-starts on every login                   ║"
echo "║  • Uses < 5MB RAM                               ║"
echo "║  • Data saved to ~/cursor_capture_data/         ║"
echo "║  • No screenshots, no keystrokes — just         ║"
echo "║    cursor position and time                     ║"
echo "║                                                 ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# Check if binary exists
if [ ! -f "$BINARY_PATH" ]; then
    echo "  ✗ Error: Cannot find $BINARY_NAME in the same folder."
    echo "    Make sure this script is next to the cursor_capture binary."
    echo ""
    echo "  Press Enter to exit..."
    read -r
    exit 1
fi

echo "  [1/5] Creating install directory..."
mkdir -p "$INSTALL_DIR"
cp "$BINARY_PATH" "$INSTALL_DIR/$BINARY_NAME"
chmod +x "$INSTALL_DIR/$BINARY_NAME"
echo "        ✓ Installed to $INSTALL_DIR/"

echo ""
echo "  [2/5] Creating data directory..."
mkdir -p "$DATA_DIR"
echo "        ✓ Data will be saved to $DATA_DIR/"

echo ""
echo "  [3/5] Setting up auto-start daemon..."

# Stop existing daemon if running
launchctl unload "$PLIST_PATH" 2>/dev/null || true

# Write the plist
cat > "$PLIST_PATH" << PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
  <dict>
    <key>Label</key>
    <string>CursorCapture</string>
    <key>ProgramArguments</key>
    <array>
      <string>$INSTALL_DIR/$BINARY_NAME</string>
      <string>run</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>$LOG_FILE</string>
    <key>StandardErrorPath</key>
    <string>$LOG_FILE</string>
    <key>EnvironmentVariables</key>
    <dict>
      <key>RUST_LOG</key>
      <string>info</string>
    </dict>
  </dict>
</plist>
PLIST

echo "        ✓ LaunchAgent configured (starts on every login)"

echo ""
echo "  [4/5] Granting Accessibility permission..."
echo ""
echo "  ┌──────────────────────────────────────────────┐"
echo "  │  A System Settings window will open.         │"
echo "  │                                              │"
echo "  │  Please:                                     │"
echo "  │  1. Click the + button                       │"
echo "  │  2. Navigate to:                             │"
echo "  │     ~/Applications/CursorCapture/            │"
echo "  │  3. Select 'cursor_capture'                  │"
echo "  │  4. Make sure the toggle is ON               │"
echo "  │                                              │"
echo "  │  Then come back here and press Enter.        │"
echo "  └──────────────────────────────────────────────┘"
echo ""

# Open Accessibility settings
open "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"

echo "  Press Enter once you've granted permission..."
read -r

echo ""
echo "  [5/5] Starting daemon..."
launchctl load "$PLIST_PATH"

# Verify it's running
sleep 2
if launchctl list | grep -q CursorCapture; then
    echo "        ✓ Daemon is running!"
else
    echo "        ⚠ Daemon may not have started. Try logging out and back in."
fi

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║                                                 ║"
echo "║  ✓ Installation complete!                       ║"
echo "║                                                 ║"
echo "║  CursorCapture is running in the background     ║"
echo "║  and will auto-start on every login.            ║"
echo "║                                                 ║"
echo "║  You can close this window now.                 ║"
echo "║                                                 ║"
echo "║  To check status:                               ║"
echo "║    ~/Applications/CursorCapture/cursor_capture status"
echo "║                                                 ║"
echo "║  To uninstall:                                  ║"
echo "║    ~/Applications/CursorCapture/cursor_capture uninstall"
echo "║                                                 ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""
echo "  Press Enter to close..."
read -r
