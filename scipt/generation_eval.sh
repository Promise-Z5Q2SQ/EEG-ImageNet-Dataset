#!/bin/bash

PYTHON_SCRIPT="../src/gen_eval.py"

DATA_DIR="../data/"
G_OPTION="all"
M_OPTION="mlp_sd"
B_OPTION=10
S_OPTION=4
P_OPTION="mlpsd_s${S_OPTION}_tmp.pth"
O_OPTION="../output/"

python $PYTHON_SCRIPT -d $DATA_DIR -g $G_OPTION -m $M_OPTION -b $B_OPTION -p $P_OPTION -s $S_OPTION -o $O_OPTION
#python $PYTHON_SCRIPT -d $DATA_DIR -g $G_OPTION -m $M_OPTION -b $B_OPTION -s $S_OPTION -o $O_OPTION

#for i in {0..15}
#do
#    P_OPTION1="rgnn_s${i}_200_2_bn2x_2.pth"
##    python $PYTHON_SCRIPT -d $DATA_DIR -g $G_OPTION -m $M_OPTION -b $B_OPTION -s $i -o $O_OPTION
#    python $PYTHON_SCRIPT -d $DATA_DIR -g $G_OPTION -m $M_OPTION -b $B_OPTION -p $P_OPTION1 -s $i -o $O_OPTION
#done