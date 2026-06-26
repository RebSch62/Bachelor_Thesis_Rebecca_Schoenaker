# Distinguishing tremor from voluntary rhythmic movements

This repository contains the code to investigate the signal difference between tremor and voluntary rhythmic movement (VRM) windows, as discussed in the bachelor thesis "Reducing false positives in tremor detection: distinguishing Parkinsonian rest tremor from voluntary rhythmic movements using deep learning".





## Overview

The code contained in this project contains the entire pipeline from importing and preprocessing the data until evaluation. The code is separated over distinct files with different functions, namely:
1. config.py: contains all the constants to direct the pipeline in main.py;
2. main.py: contains all the steps of the entire pipeline;
3. data.py: imports/preprocesses data;
4. training.py: takes care of cross-validation, hyperparameter tuning, and training of the models;
5. model_performance_evaluation.py: measures the performance of the model;
6. model_evaluations.py: inspects the model's temporal and spectral filters;
7. signal_evaluations.py: investigates several physical characteristics of the signal;
8. statistic_functions.py: handles the significance tests.





## Running the code

The data used for this project is not publicly available. 



The code uses the requirements as specified in requirements.txt. There are two directories defined within the code. Change DIR to the directory the code files are located, and DATA\_ROOT to where the local or external data is located.



To run individual parts of the pipeline, change the corresponding constants as specified in constants.py. After training, the parameters are saved to easily run the evauations after.







