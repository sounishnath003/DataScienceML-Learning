#!/bin/bash

clear
black .
rm -fr lightning_logs
rm -fr mlruns
clear
echo "######### execution is starting off ######### "
python -m torchf.main
echo "######## executiion completed ########## "
