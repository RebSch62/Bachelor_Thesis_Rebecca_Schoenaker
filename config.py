from pathlib import Path
import torch
from colorspace import sequential_hcl
import json

# Paths
DIR = Path(r"C:\Users\rebec\BCI\Thesis")
DATA_ROOT = Path(r"D:\Internships\Rebecca")

with open(r"C:\Users\rebec\BCI\Thesis\patients.json") as f:
    PATIENTS = json.load(f)
with open(r"C:\Users\rebec\BCI\Thesis\controls.json") as f:
    CONTROLS = json.load(f)
with open(r"C:\Users\rebec\BCI\Thesis\participants.json") as f:
    PARTICIPANTS = json.load(f)
 


# Data
NORM = False
FS = 100
WINDOW_S = 400
GYRO_AXES = ['gyroscope_x', 'gyroscope_y', 'gyroscope_z']

# Training
BATCH_SIZE = 64
EPOCHS_INNER = 1
EPOCHS_OUTER = 10
EPOCHS_PERM = 20
INNER_SPLITS = 5
N_TRIALS = 5
N_FOLDS = 4
SEED = 42

# Constant
N_PSD_WINDOWS     = 129   
N_DURATION_WINDOWS = 200  
N_SHAP_BACKGROUND  = 100
N_SHAP_WINDOWS     = 50

# Data information
COUNT_DATA_MEASUREMENT_TIME = False

# Actions
IMPORTING = False
TRAINING = False

# Model evaluation
ROC = False
LOSS_CURVES = False
PERMUTATION_DISTRIBUTION = False

PERTURBATION = False
FILTER_VISUALISATION = False
TEMPORAL_FILTERS = False
SPATIAL_FILTERS = False

OSCILLATION_DURATION = False
PSD = False


# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Color palette
C1 = '#0072B2'  
C2 = '#D55E00'    
C3 = '#009E73'  
C4 = '#CC79A7'  
C5 = '#56B4E9'  
C6 = '#E69F00'  
C7 = "#07B989"  