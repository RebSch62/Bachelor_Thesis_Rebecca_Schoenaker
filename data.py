from collections import Counter
from scipy.signal import butter, filtfilt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from config import *
from statistic_functions import * 


def load_paths(paths, is_patient):
    """
    Reads data and preprocesses it
    """
    X, y, p_ids = [], [], []
    i=0

    for path in paths:
        if not path.exists():
            continue
        participant_id = path.name
        print(f"Loading path: {path}.")

        for parquet_file in path.glob("**/*_tremor_predictions.parquet"):
            df = pd.read_parquet(
                parquet_file, columns=["gyroscope_x", "gyroscope_y", "gyroscope_z", "pred_tremor_checked"]
            )

            # Take only FP/TP windows
            df = df[df["pred_tremor_checked"] == 1]

            # Skip file if no flagged windows present
            if len(df) == 0:
                continue

            # Stack the remaining values into an array
            x = np.stack([
                np.stack(df["gyroscope_x"].values),
                np.stack(df["gyroscope_y"].values),
                np.stack(df["gyroscope_z"].values),
            ], axis=2).astype(np.float32)   
            x = np.array([preprocessing_window(w) for w in x])
            X.append(x)
            
            # Save ID of patient for masking
            p_ids.append(np.array([participant_id] * len(df)))

            # For patients, assign value of pred_tremor_checked, for controls always 0
            if is_patient:
                y.append(np.ones(len(df), dtype=np.int64))
            else:
                y.append(np.zeros(len(df), dtype=np.int64))
            
        i += 1

    
    return np.concatenate(X, axis=0), np.concatenate(y, axis=0), np.concatenate(p_ids, axis=0)
            

def load_group(train_patient, train_control):
    """
    Reading data and concatenating it into lists
    """  

    # Build paths to file destinations
    patient_paths = [DATA_ROOT/"parkinson"/"0"/p for p in train_patient]
    control_paths = [DATA_ROOT/"controls"/w/c for c in train_control for w in ['0', '2']]

    # Load data from different classes
    X_p, y_p, part_ids_p = load_paths(patient_paths, is_patient=True)
    X_c, y_c, part_ids_c = load_paths(control_paths, is_patient=False)
    
    # Concatenate into data, labels, and the corresponding participant id
    X = np.concatenate([X_p, X_c], axis=0)
    y = np.concatenate([y_p, y_c], axis=0)
    part_ids = np.concatenate([part_ids_p, part_ids_c], axis=0)

    return X, y, part_ids



def count_group(paths):
    """
    Counts the summed duration of total tremor labeled windows.
    """
    tremor_labeled = 0
    duration = 0

    # Iterate over all paths
    for path in paths:
        if not path.exists():
            continue
        
        # Load the data and sum
        for parquet_file in path.glob("**/*_tremor_predictions.parquet"):
            df = pd.read_parquet(
                parquet_file, columns=["pred_tremor_checked" ]
            )

            tremor_labeled += np.sum(df["pred_tremor_checked"] == 1)
            duration += len(df.index)

    return tremor_labeled, duration


def count_raw_data(train_patient, train_control):
    """
    Counts the duration of measurements.
    """
    # Define the paths the data is saved on
    patient_paths = [DATA_ROOT/"parkinson"/"0"/p for p in train_patient]
    control_paths = [DATA_ROOT/"controls"/w/c for c in train_control for w in ['0', '2']]

    # Calculate durations
    patient_nr, dur_patients = count_group(patient_paths)
    control_nr, dur_controls = count_group(control_paths)
    total_duration = dur_patients + dur_controls

    print(f"    Real tremor in dataset: {patient_nr}. False positives: {control_nr}")
    print(f"    Total recording time: {total_duration*4} seconds.")
    print(f"    Total patient: {dur_patients}")
    print(f"    Total controls: {dur_patients}")
    print(f"    Total average measurement time per patient per day: {dur_patients/(97*7*60*15)} hours")

    return patient_nr, control_nr, total_duration


def download_data():
    """
    Imports data from external device.
    """
    X, y, part_ids = load_group(PATIENTS, CONTROLS)
    print(f"    X shape: {X.shape}")
    print(f"    y shape: {y.shape}")

    np.save('X.npy', X)
    np.save('y.npy', y)
    np.save('part_ids.npy', part_ids)



def import_stored_data():
    """
    Imports saved data.
    """
    if NORM == True:
        X  = np.load(DIR / 'X.npy')
        y  = np.load(DIR / 'y.npy')
        part_ids = np.load(DIR / 'part_ids.npy')
    else:
        X  = np.load(DIR / 'X_nonorm.npy')
        y  = np.load(DIR / 'y_nonorm.npy')
        part_ids = np.load(DIR / 'part_ids_nonorm.npy')

    print(f"    Shape of data loaded: {X.shape}, labels: {Counter(y)}")
    
    return X, y, part_ids



def preprocessing_window(signal, low=0.5, high=25.0,order=4):
    """
    Preprocessing steps: frequency filtering and normalisation
    """
    # Frequency filtering
    nyquist = FS /2 
    b,a = butter(order, [low/nyquist, high/nyquist], btype = 'band')
    filtered = filtfilt(b, a, signal, axis=0)

    if NORM == True:
    # Per-window z-score normalisation
        mean = filtered.mean(axis=0, keepdims=True)
        std = filtered.std(axis=0, keepdims=True)+1e-8
        preprocessed = (filtered-mean)/std

    return preprocessed.T.astype(np.float32)


def balance_data(X, y, part_ids):
    """
    Ensures balanced classes within the data.
    """
    # Find the corrersponding indices
    n_controls = (y==0).sum()
    patient_idx = np.where(y==1)[0]
    control_idx = np.where(y==0)[0]

    # Undersample patient windows
    keep_patients = np.random.choice(patient_idx, n_controls, replace=False)
    keep_idx      = np.concatenate([keep_patients, control_idx])

    # Balanced data
    X        = X[keep_idx]
    y        = y[keep_idx]
    part_ids = part_ids[keep_idx]

    print(f"    After balancing: {(y==1).sum()} patient windows, {(y==0).sum()} control windows")

    return X, y, part_ids



def amplitude_differences(X, y, part_ids):
    """
    Computes the difference between amplitudes of gyroscope signals
    """
    y_t = np.where(y == 1)[0]
    y_v = np.where(y == 0)[0]

    # Compute per-window amplitude 
    amplitudes_t = np.array([np.abs(X[i]).mean() for i in y_t])
    amplitudes_v = np.array([np.abs(X[i]).mean() for i in y_v])

    # Find subject ids
    subject_ids_t = part_ids[y_t]
    subject_ids_v = part_ids[y_v]

    # Aggregate values per subject
    t_per_subject = np.array([np.mean(amplitudes_t[subject_ids_t == subj]) for subj in np.unique(subject_ids_t)])
    v_per_subject = np.array([np.mean(amplitudes_v[subject_ids_v == subj])for subj in np.unique(subject_ids_v)])

    # Run stats on subject aggregated values
    run_stats(t_per_subject, v_per_subject)

    return t_per_subject, v_per_subject
    

def amplitude_distribution(t_per_subject, v_per_subject):
    """
    Visualises the amplitude distribution.
    """
    df = pd.DataFrame({
        'Mean absolute amplitude': np.concatenate([t_per_subject, v_per_subject]),
        'Class': ['Tremor'] * len(t_per_subject) + ['Voluntary rhythmic movement'] * len(v_per_subject)
    })

    fig, ax = plt.subplots(figsize=(6, 5), layout='constrained')
    sns.boxplot(data=df, x='Class', y='Mean absolute amplitude', width=0.5, palette=[C5, C6], ax=ax)
    ax.set_xlabel('')
    plt.savefig('figures/amplitude_analysis.pdf')
    plt.show()
    