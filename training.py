import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from braindecode.models import EEGNet
from sklearn.model_selection import StratifiedGroupKFold
import os
import json
import pickle


from training import *
from config import *


def outer_folds_definition(X,y,part_ids):
    """
    Randomly selects groups for outer folds using entire participants.
    """
    sgkf_outer = StratifiedGroupKFold(n_splits=EPOCHS_OUTER, shuffle=True, random_state=SEED)
    outer_folds = list(sgkf_outer.split(X, y, groups=part_ids))

    return outer_folds


def hyperparameter_suggestions(trial):
    """
    Selects hyperparameter values for Optuna trial.
    """

    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    drop_prob = trial.suggest_float("drop_prob", 0.1, 0.5)
    kernel_length = trial.suggest_categorical("kernel_length", [32, 64, 128])

    return lr, drop_prob, kernel_length


def train_validation_split(X_train, y_train, part_ids_train):
    """
    Splits data into a 80/20 train/val folds and selects a single split as train/validation data.
    """

    sgkf_inner = StratifiedGroupKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=SEED)
    train_idx_inner, val_idx = next(sgkf_inner.split(X_train, y_train, groups=part_ids_train))

    return train_idx_inner, val_idx



def optuna_call(optuna, objective, i):
    """
    Creates and performs an Optuna study.
    """
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=N_TRIALS)
    best = study.best_params
    roc = study.best_value
    print(f"Best params fold {i+1}: {best}, roc: {roc:.3f}")

    return best, roc


def objective(X_train_outer, y_train_outer, part_ids_train, trial):
    """
    Function used by Optuna
    """

    # Hyperparameter suggestions
    lr, drop_prob, kernel_length = hyperparameter_suggestions(trial)

    # Defining the inner folds
    train_idx_inner, val_idx = train_validation_split(X_train_outer, y_train_outer, part_ids_train)

    X_train_i, y_train_i = X_train_outer[train_idx_inner], y_train_outer[train_idx_inner]
    X_val_i, y_val_i  = X_train_outer[val_idx], y_train_outer[val_idx]

    # Model defition
    m = EEGNet(n_chans=3, n_outputs=2, n_times=WINDOW_S, F1=8,D=2, 
                kernel_length=kernel_length, drop_prob=drop_prob).to(DEVICE)
    
    # ROC-AUC score calculation
    roc, *_ = train_evaluate(
        m, X_train_i, y_train_i, X_val_i, y_val_i, lr=lr, 
        batch_size=BATCH_SIZE, epochs=EPOCHS_INNER, device=DEVICE, verbose=False
    )

    # Free memory
    del m
    torch.cuda.empty_cache()
    
    # Return the average across both folds
    return roc


def convert_to_loaders(X_tr_t, X_val_t, y_tr_t, y_val_t, batch_size):
    """
    Converts data to DataLoaders to be used by the model.
    """
    dataset = TensorDataset(torch.tensor(X_tr_t, dtype=torch.float32), torch.tensor(y_tr_t, dtype=torch.long))
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    X_val_tensor = torch.tensor(X_val_t, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val_t, dtype=torch.long)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader



def train_evaluate(model, X_tr_t, y_tr_t, X_val_t, y_val_t, lr, batch_size, epochs, device, verbose=True):
    """
    Trains a model and returns ROC-AUC score
    """
    train_loss, val_loss= [], []

    # Define a model, optimiser and criterion
    optimizer = torch.optim.AdamW(model.parameters(),lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Loaders
    train_loader, val_loader = convert_to_loaders(X_tr_t, X_val_t, y_tr_t, y_val_t, batch_size)

    i=0
    # Training phase
    for epoch in range(epochs):
        model.train()
        running_loss, running_acc, total = 0.0, 0.0, 0.0

        # Iterate over batches
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device,non_blocking=True), y_batch.to(device,non_blocking=True)

            # zero param gradients
            optimizer.zero_grad()

            # forward, backward and optimisation
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            running_acc += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
        train_loss.append(running_loss/len(train_loader))
            

        # Evaluation phase
        model.eval()
        running_vloss= 0.0
        probs, preds, labels = [], [], []

        # Disable gradients
        with torch.no_grad():
            for  X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device,non_blocking=True)
                y_batch = y_batch.to(device,non_blocking=True)

                # forward
                outputs = model(X_batch)
                running_vloss += criterion(outputs, y_batch).item()
                _, predicted = torch.max(outputs, 1)

                preds.extend(predicted.cpu().numpy())
                probs.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())
                labels.extend(y_batch.cpu().numpy())

        val_loss.append(running_vloss / len(val_loader))

    # Calculate the AUC score
    roc_auc = roc_auc_score(labels, probs)
    if verbose:
        tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0,1]).ravel()
        print(f"\n  Confusion matrix:")
        print(f"  TP: {tp:5d}  FP: {fp:5d}")
        print(f"  FN: {fn:5d}  TN: {tn:5d}")
        print(f"  Sensitivity (TPR): {tp/(tp+fn):.3f}")
        print(f"  Specificity (TNR): {tn/(tn+fp):.3f}")
        print(f"  ROC-AUC:           {roc_auc:.3f}")


    return roc_auc, train_loss, val_loss, np.array(probs), np.array(labels), np.array(preds)



def print_info_outer_fold(i, best, roc, outer_results):
    """
    Prints information for outer fold.
    """

    print(f"\n{'='*50}")
    print(f"OUTER FOLD {i+1}/10 COMPLETE")
    print(f"  Best params:  {best}")
    print(f"  AUC:    {roc:.3f}")
    print(f"  Running mean: {np.mean(outer_results):.3f} +/- {np.std(outer_results):.3f}")
    print(f"{'='*50}\n")


def clean_files():
    """
    Removes files from a previous training session.
    """
    files_to_delete = [
        DIR/"best_params.json",
        DIR/"outer_results.npy",
        DIR/"all_probs.pkl",
        DIR/"all_labels.pkl",
        DIR/"all_preds.pkl",
        DIR/"all_train_loss.pkl",
        DIR/"all_val_loss.pkl",
    ] + [DIR/f"model_fold{i}.pth" for i in range(1, 11)]

    for f in files_to_delete:
        if os.path.exists(f):
            os.remove(f)
            print(f"Deleted: {f}")


def save_parameters(fold, final_model, outer_results,  all_val_loss, all_train_loss, all_probs, all_preds, all_labels_list, all_best_params):
    """
    Function to retrieve saved parameters
    """
    
    torch.save(final_model.state_dict(), DIR/f"model_fold{fold+1}.pth")
    np.save(DIR/"outer_results.npy", outer_results)
    with open(DIR/"all_val_loss.pkl", "wb") as f:
        pickle.dump(all_val_loss, f)
    with open(DIR/"all_train_loss.pkl", "wb") as f:
        pickle.dump(all_train_loss, f)
    with open(DIR/"all_probs.pkl", "wb") as f:
        pickle.dump(all_probs, f)
    with open(DIR/"all_preds.pkl", "wb") as f:
        pickle.dump(all_preds, f)
    with open(DIR/"all_labels.pkl", "wb") as f:
        pickle.dump(all_labels_list, f)
    with open(DIR/"best_params.json", "w") as f:
        json.dump(all_best_params, f)


def load_saved_parameters(): 
    """
    Function that saves parameters after training
    """   
    with open(DIR/"best_params.json") as f:
        all_best_params = json.load(f)
        print(all_best_params)

    # Reconstruct each fold's model from saved state_dict
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = []
    for i, params in enumerate(all_best_params):
        m = EEGNet(
            n_chans=3, n_outputs=2, n_times=400,
            F1=8, D=2,
            kernel_length=params["kernel_length"],
            drop_prob=params["drop_prob"],
        ).to(device)
        m.load_state_dict(torch.load(
            DIR/f"model_fold{i+1}.pth",
            map_location=device, weights_only=True
        ))
        m.eval()
        models.append(m)

    # Retrieve other results
    outer_results   = list(np.load(DIR/"outer_results.npy"))
    with open(DIR/"all_val_loss.pkl", "rb") as f:
        all_val_loss = pickle.load(f)
    with open(DIR/"all_train_loss.pkl", "rb") as f:
        all_train_loss = pickle.load(f)
    with open(DIR/"all_probs.pkl", "rb") as f:
        all_probs = pickle.load(f)
    with open(DIR/"all_preds.pkl", "rb") as f:
        all_preds = pickle.load(f)
    with open(DIR/"all_labels.pkl", "rb") as f:
        all_labels_list = pickle.load(f)
    with open(DIR/"best_params.json") as f:
        all_best_params = json.load(f)
    
    return models, outer_results, all_val_loss, all_train_loss, all_probs, all_preds, all_labels_list, all_best_params

    