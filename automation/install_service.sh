#!/usr/bin/env bash
# Install the recording + clog-detection service as a systemd user service,
# so it starts in the background and restarts if it crashes.
#
#   bash automation/install_service.sh          install, enable and start
#   systemctl --user status ovad-anomaly        check it
#   journalctl --user -u ovad-anomaly -f        follow its log
#   systemctl --user stop ovad-anomaly          stop it
#
# The service stops when you log out unless lingering is enabled:
#   loginctl enable-linger "$USER"
set -e
REPO="$(cd "$(dirname "$0")/.." && pwd)"
UNIT_DIR="$HOME/.config/systemd/user"
mkdir -p "$UNIT_DIR"
cat > "$UNIT_DIR/ovad-anomaly.service" <<EOF
[Unit]
Description=O-VAD pipette recording and clog detection (HTTP :8100)
After=network-online.target

[Service]
ExecStart=/bin/bash "$REPO/automation/start_service.sh"
Restart=on-failure
RestartSec=10
# give ffmpeg time to finish an open recording on stop
TimeoutStopSec=30
KillSignal=SIGINT

[Install]
WantedBy=default.target
EOF
systemctl --user daemon-reload
systemctl --user enable --now ovad-anomaly
sleep 3
systemctl --user --no-pager status ovad-anomaly | head -5
curl -s localhost:8100/health || echo "service not answering yet; see: journalctl --user -u ovad-anomaly"
