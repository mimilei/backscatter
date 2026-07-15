#!/bin/bash
# Move each .npz file out of its trial<N> subfolder into the parent
# directory, renaming it to data_<N>.npz.
#
# Usage:
#   ./flatten_trial_npz.sh [target_dir]
#
# target_dir defaults to the egain folder for 20260708 if not given.

set -euo pipefail

DIR="${1:-/Users/michang/Documents/research/microfluidic_rf/code/backscatter/1c-tag-main/data/20260708/egain}"

if [[ ! -d "$DIR" ]]; then
  echo "Error: $DIR is not a directory" >&2
  exit 1
fi

shopt -s nullglob

for trial_dir in "$DIR"/trial*/; do
  trial_dir="${trial_dir%/}"
  trial_name="$(basename "$trial_dir")"

  if [[ ! "$trial_name" =~ ^trial([0-9]+)$ ]]; then
    echo "Skipping $trial_dir: name doesn't match trial<N>"
    continue
  fi
  trial_num="${BASH_REMATCH[1]}"

  npz_files=("$trial_dir"/*.npz)
  if [[ ! -e "${npz_files[0]:-}" ]]; then
    echo "Skipping $trial_dir: no .npz files found"
    continue
  fi

  if [[ ${#npz_files[@]} -gt 1 ]]; then
    echo "Warning: $trial_dir has ${#npz_files[@]} .npz files; only the first will be moved to data_${trial_num}.npz"
  fi

  dest="$DIR/data_${trial_num}.npz"
  if [[ -e "$dest" ]]; then
    echo "Skipping ${npz_files[0]}: $dest already exists"
    continue
  fi

  echo "Moving ${npz_files[0]} -> $dest"
  mv "${npz_files[0]}" "$dest"

  if [[ -z "$(ls -A "$trial_dir")" ]]; then
    echo "Removing empty $trial_dir"
    rmdir "$trial_dir"
  fi
done
