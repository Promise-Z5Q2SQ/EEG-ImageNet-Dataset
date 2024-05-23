#!/bin/bash

PYTHON_SCRIPT="../src/object_classification.py"

DATA_DIR="../data/"
G_OPTION="all"
M_OPTION="rgnn"
B_OPTION=80
P_OPTION="mlp_s15_3l_c_0.pth"
S_OPTION=0
O_OPTION="../output/"

#python $PYTHON_SCRIPT -d $DATA_DIR -g $G_OPTION -m $M_OPTION -b $B_OPTION -p $P_OPTION -s $S_OPTION -o $O_OPTION
python $PYTHON_SCRIPT -d $DATA_DIR -g $G_OPTION -m $M_OPTION -b $B_OPTION -s $S_OPTION -o $O_OPTION

#for i in {0..15}
#do
#    python $PYTHON_SCRIPT -d $DATA_DIR -g $G_OPTION -m $M_OPTION -s $i -o $O_OPTION
#done