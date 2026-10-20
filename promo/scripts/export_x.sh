#!/usr/bin/env bash
# Re-encode a master for posting on X: H.264 High level 4.2, 1080p30, AAC-LC 48 kHz, video capped at 12 Mbps with a
# one-second buffer, faststart, colour tags kept as BT.709 (the masters are rendered with --color-space=bt709).
#
# Audio is encoded once, from the lossless score, after a look-ahead true-peak limiter (ceiling -2 dBTP, which leaves
# about -1.3 dBTP after AAC so that X's own re-encode cannot clip). Re-encoding the master's AAC instead would stack
# two lossy generations and came out at +0.1 dBTP.
#
#   usage: scripts/export_x.sh MASTER.mp4 OUT.mp4 [SCORE.wav]
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
in="${1:?master mp4 (the picture is taken from it)}"; out="${2:?output file}"
wav="${3:-$here/public/audio/score.wav}"
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT

if [[ -f "$wav" ]]; then
  python3 "$here/audio/limit_truepeak.py" "$wav" "$tmp/score-x.wav" -2.0
  audio_in=(-i "$tmp/score-x.wav"); audio_map=(-map 1:a:0); audio_filter=()
else
  echo "no lossless score found; falling back to the master's audio with -1 dB of headroom" >&2
  audio_in=(); audio_map=(-map 0:a:0); audio_filter=(-af "volume=-1.0dB")
fi

# (an empty array expands to nothing only with the ${arr[@]+...} form under `set -u`)
ffmpeg -hide_banner -y -i "$in" ${audio_in[@]+"${audio_in[@]}"} -map 0:v:0 "${audio_map[@]}" \
  -c:v libx264 -profile:v high -level 4.2 -preset slow -crf 19 -maxrate 12M -bufsize 12M \
  -pix_fmt yuv420p -r 30 -g 60 \
  -colorspace bt709 -color_primaries bt709 -color_trc bt709 -color_range tv \
  -movflags +faststart -metadata title="The Missing Textbooks" \
  -c:a aac -b:a 192k -ar 48000 -ac 2 ${audio_filter[@]+"${audio_filter[@]}"} -shortest "$out"
