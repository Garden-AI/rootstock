#!/bin/bash
# Install the rootstock styles (fix rootstock, pair_style rootstock) into a
# LAMMPS source tree.
#
# Usage:
#   ./install.sh /path/to/lammps/src
#
# Then rebuild LAMMPS with CMake (see README.md).

set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 <lammps-src-dir>"
    echo "Example: $0 ~/lammps/src"
    exit 1
fi

LAMMPS_SRC="$1"

if [ ! -d "$LAMMPS_SRC" ]; then
    echo "Error: $LAMMPS_SRC is not a directory"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

cp "$SCRIPT_DIR/rootstock_ipi.h"    "$LAMMPS_SRC/"
cp "$SCRIPT_DIR/rootstock_ipi.cpp"  "$LAMMPS_SRC/"
cp "$SCRIPT_DIR/fix_rootstock.h"    "$LAMMPS_SRC/"
cp "$SCRIPT_DIR/fix_rootstock.cpp"  "$LAMMPS_SRC/"
cp "$SCRIPT_DIR/pair_rootstock.h"   "$LAMMPS_SRC/"
cp "$SCRIPT_DIR/pair_rootstock.cpp" "$LAMMPS_SRC/"

echo "Installed fix rootstock and pair_style rootstock into $LAMMPS_SRC"
echo "Rebuild LAMMPS to pick up the new styles."
