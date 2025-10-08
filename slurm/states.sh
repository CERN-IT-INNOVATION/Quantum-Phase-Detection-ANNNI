#!/usr/bin/env bash

# Fixed parameters
SIDE=20
CHI=64
DEPTH=10

# Loops
for spins in {6..20}; do
    echo "Running: spins=$spins"
    python states.py \
        --side "$SIDE" \
        --spins "$spins" \
        --chi "$CHI" \
        --depth "$DEPTH" \
        --diag --fid
done