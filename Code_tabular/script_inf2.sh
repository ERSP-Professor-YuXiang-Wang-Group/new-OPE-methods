#!/bin/sh
i=$@
size=1
mix=0.2
tra=400000
gamma=0.98
python all_compare2.py $i $size $mix $tra $gamma

