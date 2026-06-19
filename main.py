import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
t = time.time()
import numpy as np
import random
import sys


from config import *
from statistic_functions import *
from data import *
from preprocessing import *
from training import *
from signal_evaluations import *
from model_evaluations import *
from model_performance_evaluations import *

if TRAINING:
    import optuna
    import gc
    import torch
    from tqdm import tqdm

print(f"Imports: {time.time()-t:.2f}s")


if __name__ == '__main__':
    start_time = time.time()

    # ========== SET UP =============
    print("\n\n\n=== SETTING UP ===")

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    random.seed(SEED)
    print(f"Device: {DEVICE}")



    # ========== LOAD DATA =============
    print("\n\n\n=== LOADING DATA ===")

    # ---- LOADING THE DATA ----
    if COUNT_DATA_MEASUREMENT_TIME:
        print("  Counting data measurement time for both groups.")
        count_raw_data(PATIENTS, CONTROLS)

    # Loading data from disk
    if IMPORTING == True:

        # Load function
        print("  Importing data from external storage.")
        download_data()


    print("  Importing locally stored data")
    X_unbalanced, y_unbalanced, part_ids_unbalanced = import_stored_data()


    if NORM == False:
        # Non-normalised data for amplitude analysis
        print("    Amplitude analysis")
        mean_amplitudes_t, mean_amplitudes_v = amplitude_differences(X_unbalanced, y_unbalanced, part_ids_unbalanced)
        amplitude_distribution(mean_amplitudes_t, mean_amplitudes_v)

        # Exit to not get unexpected results
        sys.exit("\nFor the rest of the code run, the data should be normalised and, therefore, this script will be exited. To run the rest of the analysis, set NORM to True.")

                
    
    # ---- BALANCING CLASSES ----
    print("  Balancing classes for comparison.")
    X, y, part_ids = balance_data(X_unbalanced, y_unbalanced, part_ids_unbalanced)
    

    

    # ========== TRAINING PHASE =============
    print("\n\n\n=== TRAINING PHASE ===")

    # Defining the outer folds
    print("  Defining outer folds.")
    outer_folds = outer_folds_definition(X,y,part_ids)

    # Only train if specified
    if TRAINING == True:

        # Remove previous runs
        print("  Cleaning files.")
        clean_files()

        # Define lists to save results
        outer_results = []
        all_train_loss     = []
        all_val_loss       = []
        all_probs          = []
        all_labels_list    = []
        all_best_params = []
        all_preds = []


        # Outer loop
        for i, (train_idx, test_idx) in tqdm(enumerate(outer_folds)):

            # Outer train and test data
            X_train_outer = X[train_idx]           
            y_train_outer = y[train_idx]
            X_test_outer       = X[test_idx]
            y_test_outer       = y[test_idx]
            part_ids_train = part_ids[train_idx]

    

            # Creating an Optuna study
            print("  Creating Optuna study.")
            best, roc = optuna_call(optuna, objective, i)
            
            # Load best inner model
            print("  Retrieve model with optimal hyperparameters.")
            final_model = EEGNet(
                n_chans=3, n_outputs=2, n_times=WINDOW_S,F1=8, D=2,
                kernel_length=best["kernel_length"], drop_prob=best["drop_prob"]).to(DEVICE)

            # Train on outer data
            print("  Train and test model on outer folds.")
            roc, train_loss, val_loss, probs, labels, preds = train_evaluate(
                final_model, X_train_outer, y_train_outer, X_test_outer,  y_test_outer, lr=best["lr"], 
                batch_size=BATCH_SIZE, epochs=EPOCHS_OUTER, device=DEVICE
            )



            # Save all results
            all_train_loss.append(train_loss)
            all_val_loss.append(val_loss)
            all_probs.append(probs)
            all_labels_list.append(labels)
            all_best_params.append(best)
            all_preds.append(preds)
            outer_results.append(roc)


            # Cleaning nemory
            gc.collect()
            torch.cuda.empty_cache()

            # Saving parameters
            save_parameters(i, final_model, outer_results,  all_val_loss, all_train_loss, all_probs, 
                    all_preds, all_labels_list, all_best_params)
        
            # Print statistics about outer fold
            print_info_outer_fold(i, best, roc, outer_results)

            



    # ========== EVALUATION PHASE =============
    print("\n\n\n=== EVALUATION PHASE ===")

    # Loading saved parameters for evaluation
    print("  Retrieved saved parameters.")
    models, outer_results, all_val_loss, all_train_loss, all_probs, all_preds, all_labels_list, all_best_params = load_saved_parameters()
    

    # # Loading best model
    best_fold  = np.argmax(outer_results)
    final_model = models[best_fold]


    final_model.eval()
    print(f"  \n\nFinal model: \n{final_model}")


    # Define some variables to be used by evaluation with same value orderings
    print("   Flattening and indicing lists for evaluation.")
    labels_flat = np.concatenate(all_labels_list)
    preds_flat  = np.concatenate(all_preds)
    probs_flat = np.concatenate(all_probs)
    X_test_all = np.concatenate([X[test_idx] for _, test_idx in outer_folds])
    y_test_all = np.concatenate([y[test_idx] for _, test_idx in outer_folds])
    part_ids_all = np.concatenate([part_ids[test_idx] for _, test_idx in outer_folds])

    # Indices into X 
    t_idx = np.where(y_test_all == 1)[0]
    v_idx = np.where(y_test_all == 0)[0]

    # Indices into X based on predicted labels
    pt_idx = np.where(preds_flat == 1)[0]   
    pv_idx = np.where(preds_flat == 0)[0]

    random_t_idx = np.random.choice(t_idx, 100, replace=False)
    random_v_idx = np.random.choice(v_idx, 100, replace=False)

    # Indices same as X_test_all
    fn_idx = np.where((labels_flat == 1) & (preds_flat == 0))[0]
    fp_idx = np.where((labels_flat == 0) & (preds_flat == 1))[0]
    tp_idx = np.where((labels_flat == 1) & (preds_flat == 1))[0]
    tn_idx = np.where((labels_flat == 0) & (preds_flat == 0))[0]

    b, a = butter(4, [3/(FS/2), 7/(FS/2)], btype='band')



    
    # ---- ROC CURVES ----
    if ROC:
        print("\n  Visualising the ROC-curves.")
        plot_roc_curves(all_probs, all_labels_list, outer_results)

    
    # ---- LOSS CURVES ----
    if LOSS_CURVES:
        print("\n   Visualising the loss curves.")
        loss_curves_plot(all_train_loss, all_val_loss)


    # ---- PERMUTATION ----
    if PERMUTATION_DISTRIBUTION:
        permutation_distribution(X, y, outer_folds, np.mean(outer_results), fold=best_fold)


    # ---- FREQUENCY PERTURBATION ----
    if PERTURBATION:
        print("\n  Frequency perturbation analysis:")

        # What bands are important for classification
        print("     Calculating the importance of each frequency band on prediction.")
        mean_perturbed_t, stds_perturbed_t = frequency_perturbation_importance(final_model, X_test_all[random_t_idx], class_idx=1)
        mean_perturbed_v, stds_perturbed_v = frequency_perturbation_importance(final_model, X_test_all[random_v_idx], class_idx=0)

        print("     Visualising the resulting prediction drops.")
        perturbed_plot(mean_perturbed_t, mean_perturbed_v, stds_perturbed_t, stds_perturbed_v)
        

    # ---- FILTER VISUALISATION ----
    if FILTER_VISUALISATION:

        # EEGNet filters visualisations
        print("\n  Visualising temporal and spatial filters.")
        plot_temporal_filters(final_model)
        plot_spatial_filters(final_model)

    
    # ---- TEMPORAL FILTER ANALYSIS ---
    if TEMPORAL_FILTERS:
        print("\n   Temporal filter analysis: ")

        # Calculate the raw filter activations
        print("     Raw filter activations")
        acts_t, acts_v = temporal_filter_activations(final_model, X_test_all, t_idx, v_idx)
        mean_t, mean_v = average_activation_per_class(acts_t, acts_v)

        print("     LIFT analysis.")
        lift_ratio_t, lift_ratio_v, lift_diff = lift_evaluation(acts_t, acts_v)

        temporal_filter_activation_difference(final_model,  mean_t, mean_v, X_test_all, t_idx, v_idx)
        lift_plot(lift_ratio_t, lift_ratio_v)

        print("     Zeroing individual temporal filters.")
        importance = temporal_filter_zeroing(final_model, X_test_all, labels_flat)
        plot_temporal_filter_zeroing(importance)

    
    # ---- SPATIAL FILTER ANALYSIS ----
    if SPATIAL_FILTERS:
        print("\n  Spatial filter analysis:")

        print("     Zeroing individual spatial filters.")
        importance = spatial_filter_zeroing(final_model, X_test_all, y_test_all)
        plot_spatial_filter_zeroing(importance)
        
        # Calculate SHAP example values and visualise
        print("     Calculate SHAP estimations attributions to axes.")
        shap_t, shap_v = calculate_shap_values(final_model, X_test_all,  t_idx, v_idx)
        plot_shap_per_axis(shap_t, shap_v)




    # ---- OSCILLATION DURATION ----
    if OSCILLATION_DURATION:
        print("\n  Envelope analysis:")

        # Calculate the envelopes
        print("     Calculation envelopes")
        envelopes_t, envelopes_v, envelopes_pt, envelopes_pv = envelope_calculations(X_test_all, t_idx, v_idx, pt_idx, pv_idx, b, a)

        # Compute one global threshold from all envelopes combined
        global_threshold = compute_global_threshold(envelopes_t, envelopes_v)
        print(f"     Global threshold: {global_threshold:.3f}")

        # Compute the durations for each category
        print("     Compare durations against global threshold.")
        t_durations, v_durations, pred_t_durations ,pred_v_durations = duration_calculations(global_threshold, envelopes_t, envelopes_v, envelopes_pt, envelopes_pv)

        # Sample the categories to speed up computations
        t_sample, v_sample, pt_sample, pv_sample = category_samples(t_durations, v_durations, pred_t_durations ,pred_v_durations)

        # Run statistics and visualisation
        run_stats(t_sample, v_sample, "True tremor versus true VRM oscillation duration:")   
        run_stats(pt_sample, pv_sample, "Predicted tremor versus predicted VRM oscillation duration:")        
        oscillation_duration_boxplots(t_sample, v_sample, pt_sample, pv_sample)

        # Do the same for confusion matrix categories
        envelopes_tp, envelopes_tn, envelopes_fp, envelopes_fn = envelope_calculations(X_test_all, tp_idx, tn_idx, fp_idx, fn_idx, b, a)
        
        tp_durations, tn_durations, fp_durations ,fn_durations = duration_calculations(global_threshold, envelopes_tp, envelopes_tn, envelopes_fp, envelopes_fn)
        tp_sample, tn_sample, fp_sample, fn_sample = category_samples(tp_durations, tn_durations, fp_durations ,fn_durations)
        
        run_stats(tp_sample, fp_sample, "Correctly versus incorrectly tremor oscillation durations:") 
        run_stats(tn_sample, fn_sample, "Correctly versus incorrectly VRM oscillation durations:")    
        
        oscillation_duration_boxplots_classes(tp_sample, fp_sample, tn_sample, fn_sample)




    
    # ---- PSD ----
    if PSD:
        print("\n  Power spectral density analysis:")

        # Find the mean aggregated PSD values per class and visualise the PSD
        print("     Calculate the aggregated PSD curves per class")
        f_axis, mean_t, mean_v, t_per_subject, v_per_subject = psds_tremor_vs_voluntary(X_test_all, y_test_all, part_ids_all, t_idx, v_idx,low=3, high=7)
        run_stats(t_per_subject, v_per_subject, "Tremor vs VRM PSD curves")

        print("     Visualise the tremor vs VRM PSD curves.")
        psds_tremor_vs_voluntary_plot(f_axis, mean_t, mean_v, 3, 7)       


        # Compare the different predicted classes
        print("     Visualise the different categorisation PSD curves.")
        psd_comparison(t_idx, v_idx, tp_idx, fp_idx, tn_idx, fn_idx,  X_test_all, low=3, high=7)

    
    


    


            













    

    





