#!/usr/bin/env bash

# Fixed parameters
SIDE=20
CHI=64
ANSA=demo
LR=0.01
NUM_TRIALS=20

# Loops
for spins in {8..14}; do
    for label in diag brick10; do
        for T in 0.1 0.25 0.5 1; do
            echo "Running: spins=$spins, label=$label, T=$T"
            python qcnn.py \
                --side "$SIDE" \
                --spins "$spins" \
                --chi "$CHI" \
                --ansatz "$ANSA" \
                --lr "$LR" \
                --label "$label" \
                --numtrials "$NUM_TRIALS" \
                --T "$T"
        done
    done
done