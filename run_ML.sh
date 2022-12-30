#!/bin/bash                                                                                                             
# This is a simple script to run ML code                                                                                                   
# The scripts assumes that the testdata is located at                                                                                    

### RETRIEVE LATEST DATA                                                                                                
#cd testdata/latest/                                                                                                    
#./retrieval_script.sh                                                                                                  
#cd ../..                                                                                                               

### DEBUGGING CALLS                                                                                                     
python3 ./xgb_training.py
#python3 ./xgb_.py
