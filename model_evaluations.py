import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from scipy.signal import butter, filtfilt
from sklearn.metrics import roc_auc_score


from training import *
from config import *



def bandstop_filter(signal, low, high, order=4):
    """
    Remove a specific frequency band from the signal.
    """
    nyq  = FS / 2
    b, a = butter(order, [low/nyq, high/nyq], btype='bandstop')
    return filtfilt(b, a, signal, axis=-1)



def frequency_perturbation_importance(model, X_windows, class_idx):
    """
    Remove frequency bands and evaluate prediction drop per category. 
    """
    model.eval()
    
    # Define frequency bands to test
    bands = {
        '0-3Hz': (0.5,  3.0),  
        '3-4Hz': (3.0,  4.0),
        '4-5Hz': (4.0, 5.0),  
        '5-6Hz': (5.0, 6.0),   
        '6-7Hz': (6.0, 7.0),
        '7-8Hz': (7.0, 8.0),
        '8-9Hz': (8.0, 9.0),
        '9-10Hz': (9.0, 10.0),
        '10-15Hz': (10.0, 15.0),   
        '15-25Hz': (15.0, 25.0), 
    }
    
    results = {band: [] for band in bands}
    
    # Loop over all data
    for i in range(len(X_windows)):
        x = torch.tensor(X_windows[i], dtype=torch.float32).unsqueeze(0).to(DEVICE)

        # Model prediction
        with torch.no_grad():
            model_prob = torch.softmax(model(x), dim=1)[0, class_idx].item()
        
        # Remove each band and measure performance drop
        for band_name, (low, high) in bands.items():

            # Apply bandstop filter to remove this frequency band
            x_perturbed = X_windows[i].copy()          
            x_perturbed = bandstop_filter(x_perturbed, low, high)
            
            # Calculate the performance drop
            x_t = torch.tensor(x_perturbed.copy(), dtype=torch.float32).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                perturbed_prob = torch.softmax(model(x_t), dim=1)[0, class_idx].item()
            
            # Calculate prediction drop and save
            drop = model_prob - perturbed_prob
            results[band_name].append(drop)

    # For each band, calculate the mean drop and std
    means = {band: np.mean(drops) for band, drops in results.items()}
    stds  = {band: np.std(drops)  for band, drops in results.items()}

    return means, stds


def perturbed_plot(mean_perturbed_t, mean_perturbed_v, stds_perturbed_t, stds_perturbed_v ):
    """
    Perturbation plot, shows prediction drop per class for the chosen removed frequency bands.
    """

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), layout='constrained', sharex=True, sharey=True)

    bands = list(mean_perturbed_t.keys())
    x_pos = np.arange(len(bands))

    
    axes[0].bar(x_pos, list(mean_perturbed_t.values()), yerr=list(stds_perturbed_t.values()), capsize=3, color=C5)
    axes[0].axhline(0, color='k', linewidth=0.5)
    axes[0].axvspan(0.5, 4.5, alpha=0.1, color='gray', label='Tremor band')
    axes[0].set_ylabel('P(tremor) drop')
    axes[0].set_title('Tremor prediction drop in frequency band removal.')
    axes[0].legend()

    axes[1].bar(x_pos, list(mean_perturbed_v.values()), yerr=list(stds_perturbed_v.values()), capsize=3, color=C6)
    axes[1].axhline(0, color='k', linewidth=0.5)
    axes[1].axvspan(0.5, 4.5, alpha=0.1, color='gray', label='Tremor band')
    axes[1].set_ylabel('P(voluntary) drop')
    axes[1].set_title('VRM prediction drop in frequency band removal.')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(bands, rotation=45, ha='right')
    axes[1].set_xlabel('Frequency band removed')
    axes[1].legend()

    plt.savefig('figures/frequency_perturbation_both_classes.pdf')
    plt.show()


def temporal_filter_zeroing(model, X_test_all, y_test_all):
    """
    Measures AUC drop when each temporal filter is zeroed out.
    """
    model.eval()
    
    # Baseline AUC with all filters
    with torch.no_grad():
        out = model(torch.tensor(X_test_all, dtype=torch.float32).to(DEVICE))
        probs_base = torch.softmax(out, dim=1)[:,1].cpu().numpy()
    baseline_auc = roc_auc_score(y_test_all, probs_base)
    print(f"Baseline AUC: {baseline_auc:.3f}")

    importance = {}

    # Copy original weights
    original_weights = model.conv_temporal.weight.data.clone()

    for i in range(8):
        
        # Zero out filter i
        model.conv_temporal.weight.data[i] = 0

        # Calculate probabilities with zeroed out filter
        with torch.no_grad():
            out = model(torch.tensor(X_test_all, dtype=torch.float32).to(DEVICE))
            probs = torch.softmax(out, dim=1)[:,1].cpu().numpy()
        
        # Calculate the AUC and performance drop for each filter
        auc = roc_auc_score(y_test_all, probs)
        importance[f'F{i+1}'] = baseline_auc - auc
        print(f"Filter {i+1}: AUC drop = {baseline_auc - auc:.4f}")

        # Restore original weights
        model.conv_temporal.weight.data = original_weights.clone()

    return importance



def plot_temporal_filter_zeroing(importance):
    """
    Visualises performance drop in zeroing of individual temporal filters.
    """
    fig, ax = plt.subplots(figsize=(8, 4), layout='constrained')
    sns.barplot(x=list(importance.keys()), y=list(importance.values()), color=C1, ax=ax)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Filter')
    ax.set_ylabel('AUC drop when temporal filter zeroed')
    plt.savefig('figures/filter_importance.pdf', dpi=150)
    plt.show()




def temporal_filter_activations(model, X_test_all, t_idx, v_idx):
    """
    Calculates the raw activation per temporal filter for both classes.
    """
    # Calculates the activations 
    activations = {}
    def hook(module, input, output):
        activations['out'] = output.detach().cpu().numpy()
    
    handle = model.conv_temporal.register_forward_hook(hook)
    model.eval()

    # Measure activation for a sample of both classes
    with torch.no_grad():
        model(torch.tensor(X_test_all[t_idx[:100]], dtype=torch.float32).to(DEVICE))
        acts_t = activations['out']
        
        model(torch.tensor(X_test_all[v_idx[:100]], dtype=torch.float32).to(DEVICE))
        acts_v = activations['out']

    handle.remove()

    return acts_t, acts_v




def temporal_filter_activation_difference(model, mean_t, mean_v, X_test_all, t_idx, v_idx):
    """
    Shows mean activation magnitude per temporal filter for tremor vs VRM windows.
    """
    
    fig = plt.figure(figsize=(14, 4), layout='constrained')
    x = np.arange(8)
    labels = [f'TF{i+1}' for i in range(8)]

    #  raw activations
    sns.barplot(x=np.concatenate([x, x]), y=np.concatenate([mean_t, mean_v]),
                hue=['Tremor']*8 + ['Voluntary rhythmic movement']*8, palette=[C5, C6])
    plt.xticks(x, labels = labels)
    plt.xlabel('Filter')
    plt.ylabel('Mean activation magnitude')
    plt.legend(loc= "lower right")
    plt.savefig('figures/temporal_activation_difference.pdf', dpi=150)
    plt.show()


def average_activation_per_class(acts_t, acts_v):
    """
    Calculates the mean average activation per class.
    """
    # Mean activation per filter 
    mean_t = np.abs(acts_t).mean(axis=(0, 2, 3))
    mean_v = np.abs(acts_v).mean(axis=(0, 2, 3))

    return mean_t, mean_v



def lift_evaluation(mean_t, mean_v):
    """
    Performs a LIFT analysis to compare all filter activations against each other.
    """

    # average baseline
    average = (mean_t + mean_v) / 2   

    # Relative activation
    lift_t = mean_t / mean_t.sum()    
    lift_v = mean_v / mean_v.sum()    
    baseline = average / average.sum() 

    # LIFT 
    lift_ratio_t = lift_t / baseline   
    lift_ratio_v = lift_v / baseline 

    # Difference between classes on standardised scale
    lift_diff = lift_ratio_t - lift_ratio_v 

    return lift_ratio_t, lift_ratio_v, lift_diff




def lift_plot(lift_ratio_t, lift_ratio_v):
    """
    Visualises the LIFT evaluation of temporal filter activations.
    """
    x = np.arange(8)
    labels = [f'TF{i+1}' for i in range(8)]

    fig = plt.figure(figsize=(14, 4), layout='constrained')
    sns.barplot(x=np.concatenate([x, x]), y=np.concatenate([lift_ratio_t, lift_ratio_v]),
                hue=['Tremor']*8 + ['Voluntary']*8, palette=[C5, C6])
    plt.axhline(1.0, color='k', linewidth=1, linestyle='--', label='Baseline (uniform)')
    plt.xticks(x, labels= labels)
    plt.xlabel('Filter')
    plt.ylabel('LIFT ratio')
    plt.legend(loc = 'lower right')

    plt.savefig('figures/lift.pdf', dpi=150)
    plt.show()




def plot_temporal_filters(model):
    """
    Plots the temporal filters of EEGNet and their corresponding spectral equivalent.

    """
    # Derive the EEGNet filters
    temporal_filters = model.get_submodule('conv_temporal').weight.squeeze(2).squeeze(1).cpu().detach().numpy()
    
    # Filter information
    f1, kernel_length = temporal_filters.shape
    fig = plt.figure(constrained_layout=True, figsize=(25,8))


    fig_temp, fig_freq_res = fig.subfigures(2, 1, wspace=0.07)
    
    # Plot temporal filters:
    axes = fig_temp.subplots(1, f1, squeeze=False,sharey=True)
    t = np.arange(kernel_length)/FS
    for i, (ax, f) in enumerate(zip(axes[0], temporal_filters)):
        ax.plot(t, f, color=C2)
        ax.set_xlabel('Time [s]')
        ax.set_title(f'Temp. filter {i+1}')
        if i==0:
            ax.set_ylabel('Filter weight')
        
    # Plot the frequency response of the temporal filters
    axes = fig_freq_res.subplots(1, f1, squeeze=False, sharey=True)
    for i, (ax, f) in enumerate(zip(axes[0], temporal_filters)):
        f_norm = f / f.sum()
        freq_res = np.abs(np.fft.fft(f_norm))[0:f.shape[0]//2 + 1]
        freqs = np.linspace(0, FS // 2, freq_res.shape[0])
        
        ax.plot(freqs, freq_res, color=C2)
        ax.set_xlabel('Frequency [Hz]')
        ax.set_title(f'Freq. response')
        ax.axvspan(3, 7, alpha=0.1, color=C5, label='Tremor band')
        ax.axvline(3, color=C5)
        ax.axvline(7, color=C5)
        
        if i == 0:
            ax.set_ylabel('Magnitude response')

    plt.savefig('figures/EEGNet_filter_explanations_temp_spect.pdf')
    plt.show()



def plot_spatial_filters(model):
    """
    Visualises the EEGNet's spatial filters and patterns.
    """
    spatial_filters = model.get_submodule('conv_spatial').weight.squeeze(3).squeeze(1).cpu().detach().numpy().reshape(model.F1, model.D, model.n_chans) 
    F1, D, in_chans = spatial_filters.shape
    

    fig = plt.figure(constrained_layout=True, figsize=(12,8))

    # # Plot spatial filters:
    axis_names = ['Gyro X', 'Gyro Y', 'Gyro Z']
    spatial_patterns = np.linalg.pinv(spatial_filters.reshape(F1*D, in_chans)).T.reshape(F1, D, in_chans)
    axes = fig.subplots(D, F1, squeeze=False, sharey=True, sharex=True)
    
    for i, (axx, ff) in enumerate(zip(axes.T, spatial_patterns)):
        
        for j, (ax, f) in enumerate(zip(axx, ff)):
            # if i ==0:
            #     ax.set_ylabel('Pattern weight')
            ax.bar(axis_names, f, color=[C1,C2,C3])
            if i==0:
                ax.set_ylabel(f'Spat. pattern {j+1}\nweight')

    plt.savefig('figures/EEGNet_filter_explanations_spat.pdf')
    plt.show()




def spatial_filter_zeroing(model, X_test_all, y_test_all):
    """
    Iteratively sets the weights of spatial filters to 0and measures performance drop.
    """
    for name, module in model.named_modules():
        print(name, type(module))

    # Calculate the baseline performance    
    with torch.no_grad():
        out = model(torch.tensor(X_test_all, dtype=torch.float32).to(DEVICE))
        probs = torch.softmax(out, dim=1)[:,1].cpu().numpy()
    baseline_auc = roc_auc_score(y_test_all, probs)

    print(f"baseline auc {baseline_auc}")

    # Save original weights to restore later
    original_weights = model.conv_spatial.parametrizations.weight.original.data.clone()
    
    importance = {}

    # Iterate over each weight
    for i in range(model.conv_spatial.weight.shape[0]):

        # Set weight to 0
        model.conv_spatial.parametrizations.weight.original.data[i] = 0

        # Measure performance drop
        with torch.no_grad():
            out   = model(torch.tensor(X_test_all, dtype=torch.float32).to(DEVICE))
            probs = torch.softmax(out, dim=1)[:,1].cpu().numpy()
        auc = roc_auc_score(y_test_all, probs)
        importance[f'SF{i+1}'] = baseline_auc - auc
        print(f"SF{i+1}: AUC drop = {baseline_auc - auc:.4f}")

        # Restore original weight
        model.conv_spatial.parametrizations.weight.original.data = original_weights.clone()

    return importance



def plot_spatial_filter_zeroing(importance):
    """
    Visualises the model's performance drop after zeroing individual filters.
    """
    fig, ax = plt.subplots(figsize=(12, 4), layout='constrained')
    
    colors = []
    for i in range(16):
        # Alternate colors per temporal filter pair
        temporal_filter = i // 2
        colors.append(C2 if temporal_filter % 2 == 0 else C3)
    
    sns.barplot(x=list(importance.keys()), y=list(importance.values()), palette=colors, ax=ax)
    
    # Add temporal filter labels
    for i, tf in enumerate(range(1, 9)):
        ax.axvline(i*2 - 0.5, color='gray', linestyle='--', lw=0.5, alpha=0.5)
        ax.text(i*2 + 0.5, ax.get_ylim()[1]*0.95, f'TF{tf}', ha='center', fontsize=7, color='gray')
    
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Spatial filter')
    ax.set_ylabel('AUC drop when filter zeroed')
    plt.savefig('figures/spatial_filter_importance.pdf', dpi=150)
    plt.show()



def calculate_shap_values(final_model, X_test_all, t_idx, v_idx):
    """
    Defines SHAP background and explainer and calculates SHAP values for a subset of windows.  
    """
    # Define background and explainer
    background  = torch.tensor(X_test_all[np.random.choice(len(X_test_all), 100)]).float().to(DEVICE)
    explainer   = shap.DeepExplainer(final_model, background)
    
    # Random sample tremor and VRM indices
    rng    = np.random.default_rng(42)
    t_samp = rng.choice(t_idx, size=min(50, len(t_idx)), replace=False)
    v_samp = rng.choice(v_idx, size=min(50, len(v_idx)), replace=False)

    X_t = torch.tensor(X_test_all[t_samp], dtype=torch.float32).to(DEVICE)
    X_v = torch.tensor(X_test_all[v_samp], dtype=torch.float32).to(DEVICE)

    # Retrieve estimated SHAP values
    shap_t = explainer.shap_values(X_t)
    shap_v = explainer.shap_values(X_v)

    return shap_t, shap_v


    

def shap_evaluation_variables(shap_class):
    """
    Calculates the mean shap values over the gyroscope X-axis.
    """

    # Average over axis
    per_gyro  = np.abs(shap_class).sum(axis=2)  

    # Retrieve statistics
    mean_shap = per_gyro.mean(axis=0)             
    std_shap  = per_gyro.std(axis=0)

    return mean_shap, std_shap



def plot_shap_per_axis(shap_t, shap_v):
    """
    Plots the axes importance of FP against TP according to SHAP values.
    """

    print(np.shape(shap_t))
    shap_class_t  = shap_t[:, :, :, 1]   
    shap_class_v = shap_v[:, :, :, 1]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, layout='constrained')

    for ax, shap_class, title, color in [
        (axes[0], shap_class_t,  'Real tremor windows',       C5),
        (axes[1], shap_class_v, 'Voluntary movement windows', C6),
    ]:
        mean_shap, std_shap = shap_evaluation_variables(shap_class)
        ax.bar(['Gyro X', 'Gyro Y', 'Gyro Z'], mean_shap, yerr=std_shap, color=color, capsize=5)
        ax.set_title(title)
        ax.set_ylabel('Mean |SHAP value|')

    plt.savefig('figures/shap_comparison.pdf')
    plt.show()







