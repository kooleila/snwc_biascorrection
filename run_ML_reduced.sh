#!/bin/bash                                                                                                             
# This is a simple script to run ML code                                                                                                   
# The scripts assumes that the testdata is located at                                                                                    

### RETRIEVE LATEST DATA                                                                                                
#cd testdata/latest/                                                                                                    
#./retrieval_script.sh                                                                                                  
#cd ../..                                                                                                               
cd /home/users/hietal/statcal/python_projects/MNWC_data/snwc_biascorrection.git/ 
export TMPDIR=/data/hietal/tmp_temp

### DEBUGGING CALLS                                                                                                     
#python3 ./xgb_reduced.py --param 'T2m'
#python3 ./xgb_reduced.py --param 'RH'
#python3 ./xgb_reduced.py --param 'WS'
python3 ./xgb_reduced.py --param 'WG'

