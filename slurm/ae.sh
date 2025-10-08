#!/usr/bin/env bash

# Fixed parameters
SIDE=20
CHI=64
ANSA=demo
LR=0.01
NUM_TRIALS=20

# Loops
for spins in {6..14}; do
    # Compute numtrash as spins // 2
    numtrash=$((spins / 2))

    # Compute depth mathematically: increases by 1 every 3 spins
    depth=$(( (spins - 6) / 3 + 1 ))

    for label in diag brick10; do
        echo "Running: spins=$spins, label=$label, numtrash=$numtrash, depth=$depth"
        python ae.py \
            --side "$SIDE" \
            --spins "$spins" \
            --chi "$CHI" \
            --ansatz "$ANSA" \
            --lr "$LR" \
            --label "$label" \
            --numtrials "$NUM_TRIALS" \
            --numtrash "$numtrash" \
            --depth "$depth"
    done
done