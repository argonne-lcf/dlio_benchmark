#!/bin/bash

python3 src/dlio_postprocessor.py --num-proc 8 --output-folder output/ --batch-size 4 --epochs 2 --do-eval y --name UNET3D --debug y

# Print out the report if postprocessing was successful
if [ $? -eq 0 ]; then cat output/DLIO_UNET3D_report.txt; fi

