#!/usr/bin/env bash
# Wrapper for parallel join_vtk

NPROCS=4
RANGE="35:734:1"
BASENAME="gc"
INDIR="."
OUTDIR="."
ATHENADIR="/home/smoon/Dropbox/gc/Athena-TIGRESS"
SCRIPT="$ATHENADIR/vtk/join_parallel.py"
mpirun -np $NPROCS python $SCRIPT -b $BASENAME -i $INDIR -o $OUTDIR -r $RANGE
