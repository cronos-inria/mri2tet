#!/usr/bin/env bash
set -euo pipefail

ref="${1:?Pass a CGAL tag, branch, or full commit SHA}"
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_dir="$project_root/.pixi/cgal/source"

mkdir -p "$(dirname "$source_dir")"
if [[ ! -d "$source_dir/.git" ]]; then
  rm -rf "$source_dir"
  git clone --filter=blob:none --no-checkout https://github.com/CGAL/cgal.git "$source_dir"
fi

git -C "$source_dir" fetch --force --depth 1 origin "$ref"
git -C "$source_dir" checkout --detach FETCH_HEAD

resolved_commit="$(git -C "$source_dir" rev-parse HEAD)"
printf 'Using CGAL ref %s (%s)\n' "$ref" "$resolved_commit"
