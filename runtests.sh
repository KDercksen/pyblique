#!/usr/bin/env bash
set -e

python test.py -f 5 iris
python test.py -f 10 iris
python test.py -f 15 iris

python test.py -f 5 isolet_compressed
python test.py -f 10 isolet_compressed
python test.py -f 15 isolet_compressed

python test.py -f 5 isolet
python test.py -f 10 isolet
python test.py -f 15 isolet
