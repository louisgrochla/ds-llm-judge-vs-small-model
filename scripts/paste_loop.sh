#!/bin/bash
# Sequential paste-and-save helper for batched Claude Code evaluation.
#
# Usage: bash scripts/paste_loop.sh [PROMPT_VERSION] [DATASET]
#   defaults: v1 test
#
# Setup once: open Claude Code in any empty directory (NOT inside any
# project with a CLAUDE.md), run /model claude-sonnet-4-6, confirm
# Sonnet 4.6 in the status bar. Keep that window open.
#
# Per batch: this script copies the next batch prompt to your clipboard,
# you paste into Claude Code, save the JSON response to the indicated
# file, then run /clear in Claude Code to wipe the conversation before
# the next batch. /clear isolates each batch from the previous one's
# context — non-negotiable for clean results.
#
# Skips batches that already have responses, so Ctrl+C is safe — re-run
# to resume.

set -e

VERSION="${1:-v1}"
DATASET="${2:-test}"

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BATCH_DIR="$REPO_DIR/results/eval_batches/$VERSION/$DATASET"
RESPONSE_DIR="$REPO_DIR/results/eval_responses/$VERSION/$DATASET"

if [ ! -d "$BATCH_DIR" ]; then
  echo "No batch directory at $BATCH_DIR"
  echo "Run notebook 02 section 2 first to generate batches."
  exit 1
fi

mkdir -p "$RESPONSE_DIR"

total=$(ls "$BATCH_DIR"/batch_*.txt 2>/dev/null | wc -l | tr -d ' ')
if [ "$total" -eq 0 ]; then
  echo "No batches in $BATCH_DIR"
  exit 1
fi

echo "Found $total batches in $BATCH_DIR"
echo "Responses save to    $RESPONSE_DIR"
echo ""

for batch_file in "$BATCH_DIR"/batch_*.txt; do
  n=$(basename "$batch_file" .txt | sed 's/batch_//')
  response_file="$RESPONSE_DIR/batch_$n.txt"

  if [ -f "$response_file" ]; then
    echo "batch $n: already saved, skipping"
    continue
  fi

  echo ""
  echo "=== batch $n / $total ==="
  cat "$batch_file" | pbcopy
  echo "Prompt copied to clipboard."
  echo ""
  echo "  1. In your Claude Code window, run /clear (skip if this is batch 01)"
  echo "  2. Paste, hit enter, wait for JSON response"
  echo "  3. Copy the JSON, save to:"
  echo "       $response_file"
  echo ""
  printf "Press enter when saved (or Ctrl+C to stop, resume by re-running)... "
  read -r
done

echo ""
echo "All batches have responses."
echo "Re-run notebook 02 sections 3-6 to parse and score."
