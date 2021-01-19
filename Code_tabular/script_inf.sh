#!/bin/sh
i=$@
size=1
mix=0.8
tra=20000
gamma=0.98
python all_compare.py $i $size $mix $tra $gamma

