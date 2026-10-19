#!/usr/bin/env bash
# Extract the vendor's matching C library locally; no sudo or system SDK changes.
set -euo pipefail
rig_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -ne 1 || ! -f "$1" ]]; then
  echo "Usage: $0 /path/to/matching/libspinnaker-c_VERSION_amd64.deb" >&2
  exit 1
fi
mkdir -p "$rig_root/.runtime/spinnaker"
dpkg-deb -x "$1" "$rig_root/.runtime/spinnaker"
echo "Local C library extracted. Use the same version as the installed Spinnaker runtime."
