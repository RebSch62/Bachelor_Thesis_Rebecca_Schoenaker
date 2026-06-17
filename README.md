# Classifying tremor from voluntary rhythmic movements

This repository contains the code to investigate the signal difference between tremor and voluntary rhythmic movement (VRM) windows, as discussed in the bachelor thesis "Reducing tremor detection false positives: distinguishing Parkinsonian rest tremor from voluntary rhythmic movements using deep learning".





## Overview

The code contained in this project contains the entire pipeline from importing and preprocessing the data until evaluation. The code is separated over distinct files with different functions, namely:

0\. constants.py: contains all the constants to direct the pipeline in main.py;

1. main.py: contains all the steps of the entire pipeline; 
2. data.py: imports/preprocesses data;
3. training.py: takes care of cross-validation, hyperparameter tuning, and training of the models;
4. model\_performance\_evaluation.py: measures the performance of the model;
5. model\_evaluations.py: inspects the model's temporal and spectral filters; 
6. signal\_evaluations.py: investigates several physical characteristics of the signal.





## Running the code

The data used for this project is not publicly available. 



The code uses the requirements as specified in requirements.txt. There are two directories defined within the code. Change DIR to the directory the code files are located, and DATA\_ROOT to where the local or external data is located.



To run individual parts of the pipeline, change the corresponding constants as specified in constants.py. After training, the parameters are saved to easily run the evauations after.







