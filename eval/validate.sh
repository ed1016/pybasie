#!/bin/bash

configfile=$1
timestamp=$(date +%Y%m%d_%H%M%S)

outfile="$timestamp.txt"

python3 validate_python_matlab.py $configfile > results/$outfile
