#!/bin/bash

PYTHON_SCRIPT="../src/object_classification.py"

DATA_DIR="../data/"
G_OPTION="all"
M_OPTION="knn"

for i in {0..15}
do
    python $PYTHON_SCRIPT -d $DATA_DIR -g $G_OPTION -m $M_OPTION -s $i -o ../scipt/tmp.txt
done