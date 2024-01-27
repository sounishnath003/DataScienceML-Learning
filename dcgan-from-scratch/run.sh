#/bin/bash

set -eu
. ./venv/bin/activate;

rm -fr lightning_logs;
mkdir -p lightning_logs;

clear;
black .;

time python3 -m main --train True;

echo "<==== execution finished =====>"
