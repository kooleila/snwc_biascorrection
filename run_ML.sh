#!/bin/bash                                                                                                             
# This is a simple script to run ML code                                                                                                   
# The scripts assumes that the testdata is located at nowcasting_fcst/testdata/$HOD                                     

          
                                                                                
#mkdir -p figures/fields                                                                                                
#mkdir -p figures/jumpiness_absdiff                                                                                     
#mkdir -p figures/jumpiness_meandiff                                                                                    
#mkdir -p figures/jumpiness_ratio                                                                                       
#mkdir -p figures/linear_change                                                                                         
#mkdir -p figures/linear_change3h                                                                                       
#mkdir -p figures/linear_change4h                                                                                       

### RETRIEVE LATEST DATA                                                                                                
#cd testdata/latest/                                                                                                    
#./retrieval_script.sh                                                                                                  
#cd ../..                                                                                                               

### DEBUGGING CALLS                                                                                                     
python3 ./xgb_training.py
#python3 ./xgb_.py
