#!/usr/bin/env bash
# Check the camera once it is plugged in: list video devices and the formats
# the camera supports, then record a short test clip with the settings in
# config.yaml.
#
#   bash automation/check_camera.sh [device] [seconds]
set -e
CFG="$(dirname "$0")/config.yaml"
DEV="${1:-$(grep -E '^\s*device:' "$CFG" | awk '{print $2}')}"
SECS="${2:-5}"

echo "== video devices"
ls -l /dev/video* 2>/dev/null || { echo "no /dev/video* devices: camera not detected"; exit 1; }
for d in /dev/video*; do
  echo "-- $d: $(cat /sys/class/video4linux/$(basename "$d")/name 2>/dev/null)"
done

echo; echo "== formats on $DEV"
ffmpeg -hide_banner -f v4l2 -list_formats all -i "$DEV" 2>&1 | grep -E "Raw|Compressed" || true

fmt=$(grep -E '^\s*input_format:' "$CFG" | awk '{print $2}')
size=$(grep -E '^\s*video_size:' "$CFG" | awk '{print $2}')
rate=$(grep -E '^\s*framerate:' "$CFG" | awk '{print $2}')
out="/tmp/camera_test_$(date +%H%M%S).mp4"
echo; echo "== recording ${SECS}s test clip ($fmt $size @ $rate fps) -> $out"
ffmpeg -hide_banner -loglevel warning -y -f v4l2 -input_format "$fmt" -video_size "$size" \
  -framerate "$rate" -i "$DEV" -t "$SECS" -vf fps=10 -c:v libx264 -preset veryfast -pix_fmt yuv420p "$out"
ffprobe -v error -show_entries stream=width,height,r_frame_rate:format=duration -of default=nw=1 "$out"
echo "OK. If the settings differ from config.yaml, update its camera section and restart the service."
