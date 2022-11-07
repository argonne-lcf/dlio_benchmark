#!/bin/bash

python3 src/dlio_postprocessor.py --num-proc 1 --output-folder output/ --batch-size 48 --epochs 1 --do-eval y --name BERT --debug y

# Print out the report if postprocessing was successful
if [ $? -eq 0 ]; then cat output/DLIO_BERT_report.txt; fi
