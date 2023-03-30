#!/bin/bash                                                                                                             
# This is a simple script to run ML code                                                                                                   
# The scripts assumes that the testdata is located at                                                                                    

### RETRIEVE LATEST DATA                                                                                                
cd /home/users/hietal/statcal/python_projects/MNWC_data/snwc_biascorrection.git/                                                                                                    
#./retrieval_script.sh                                                                                                  
#cd ../..                                                                                                               

### DEBUGGING CALLS                                                                                                     
python3 xgb_bayes_new.py --param 'WS'
#python3 xgb_bayes_new.py --param 'WG'
#python3 xgb_bayes_new.py --param 'RH'
