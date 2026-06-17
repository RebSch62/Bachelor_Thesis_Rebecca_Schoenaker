import numpy as np
from scipy.signal import welch, filtfilt, hilbert
import matplotlib.pyplot as plt

from config import *
from statistics import *

def envelope_calculations(X_test_all, t_idx, v_idx, pt_idx, pv_idx, b, a):
    """
    Calculate envelopes for tremor, VRM, predicted tremor and predicted VRM.
    """
    envelope_categories = []

    # Iterate over the four categories
    for category in [t_idx, v_idx, pt_idx, pv_idx]:
        envelopes = []

        # Calculate the envelope for each window
        for window in X_test_all[category]:
            signal   = window[0, :]                    
            filtered = filtfilt(b, a, signal)
            envelope = np.abs(hilbert(filtered))
            envelopes.append(envelope)
        envelope_categories.append(envelopes)

    return tuple(envelope_categories)


def oscillation_duration_global(envelope, threshold):
        """
        Calculates how long the envelopes values are above the set global threshold.
        """
        above = envelope > threshold
        max_duration = 0
        current      = 0
        for is_active in above:
            if is_active:
                current += 1
                max_duration = max(max_duration, current)
            else:
                current = 0
        return max_duration / FS


def duration_calculations(global_threshold, envelopes_t, envelopes_v, envelopes_pt, envelopes_pv):
    """
    Calls the duration calculator for each category.
    """
    all_durations = []
    for category in [envelopes_t, envelopes_v, envelopes_pt, envelopes_pv]:
        category_durations = []
        for window in category:
            duration = oscillation_duration_global(window, global_threshold)
            category_durations.append(duration)
        all_durations.append(np.array(category_durations))


    return tuple(all_durations)


def category_samples(t_durations, v_durations, pred_t_durations ,pred_v_durations):
    """
    Selects samples (n=1000) for visualisation and statistics calculation. 
    """
    samples_categories = []

    # Sample for each category
    for category in [t_durations, v_durations, pred_t_durations ,pred_v_durations]:
        sample = np.random.choice(category,  min(1000, len(t_durations)),  replace=False)
        samples_categories.append(sample)
    
    return tuple(samples_categories)


def compute_global_threshold(envelopes_t, envelopes_v):
    """
    Computes the global threshold used in the duration analysis.
    """
    all_env = np.concatenate([envelopes_t, envelopes_v])
    global_threshold = np.percentile(all_env, 75)  
    return global_threshold



def oscillation_duration_boxplots(t_durations, v_durations, pred_t_durations, pred_v_durations):
    """
    Visualises the difference in frequency band oscillation duration between (predicted) tremor and VRM windows. 
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), layout='constrained', sharey=True)

    # True class 
    parts1 = axes[0].violinplot([t_durations, v_durations], positions=[1, 2], showmedians=True, showextrema=False)
    parts1['bodies'][0].set_facecolor(C5)
    parts1['bodies'][1].set_facecolor(C6)
    axes[0].set_xticks([1, 2])
    axes[0].set_xticklabels(['Tremor', 'VRM'])
    axes[0].set_ylabel('Oscillation duration (s)')
    axes[0].set_title('True class')

    # Predicted class
    parts2 = axes[1].violinplot([pred_t_durations, pred_v_durations], positions=[1, 2], showmedians=True, showextrema=False)
    parts2['bodies'][0].set_facecolor(C5)
    parts2['bodies'][1].set_facecolor(C6)
    axes[1].set_xticks([1, 2])
    axes[1].set_xticklabels(['Predicted tremor', 'Predicted VRM'])
    axes[1].set_title('Predicted class')

    plt.savefig('figures/oscillation_duration_boxplots.pdf')
    plt.show()







    
def psd_single_class(idx_group, data, low, high):
    psds = []
    band_powers = []
    for i in idx_group:
        f_axis, psd = welch(data[i].mean(axis=0), fs=FS, nperseg=256) 
        psds.append(psd)
        band_mask = (f_axis >= low) & (f_axis <= high)
        mean_power = np.mean(psd[band_mask])
        band_powers.append(np.mean(psd[band_mask]))


    return band_powers, f_axis, psds 

def psds_tremor_vs_voluntary(X_test_all, y_test_all, part_ids, t_idx, v_idx, low=3, high=7):
    """
    Plots mean PSD for tremor vs voluntary movement windows, aggregated per sample.
    """
    psds_t, psds_v = [], []
    tremor_band_t, tremor_band_v = [], []

    # Calculate power spectrum per window and the average value within the tremor frequency band
    for i in np.where(y_test_all == 1)[0]:
        f_axis, psd = welch(X_test_all[i].mean(axis=0), fs=FS, nperseg=256)
        band_mask   = (f_axis >= low) & (f_axis <= high)
        psds_t.append(psd)
        tremor_band_t.append(np.mean(psd[band_mask]))

    for i in np.where(y_test_all == 0)[0]:
        f_axis, psd = welch(X_test_all[i].mean(axis=0), fs=FS, nperseg=256)
        band_mask   = (f_axis >= low) & (f_axis <= high)
        psds_v.append(psd)
        tremor_band_v.append(np.mean(psd[band_mask]))

    # Convert to arrays
    psds_t = np.array(psds_t)
    psds_v = np.array(psds_v)
    tremor_band_t = np.array(tremor_band_t)
    tremor_band_v = np.array(tremor_band_v)

    mean_t = psds_t.mean(axis=0)
    std_t  = psds_t.std(axis=0)
    mean_v = psds_v.mean(axis=0)
    std_v  = psds_v.std(axis=0)

    # Calculate the mean tremor band value of all participant
    t_per_subject = np.array([np.mean(tremor_band_t[part_ids[t_idx] == subj]) for subj in np.unique(part_ids[t_idx])])
    v_per_subject = np.array([np.mean(tremor_band_v[part_ids[v_idx] == subj]) for subj in np.unique(part_ids[v_idx])])

    return f_axis, mean_t, mean_v, t_per_subject, v_per_subject


def psds_tremor_vs_voluntary_plot(f_axis, mean_t, mean_v, low, high):
    """
    Visualises the average PSD curves aggregated over each participant for both classes.
    """
    fig, ax = plt.subplots(figsize=(8, 5), layout='constrained')

    ax.plot(f_axis, mean_t, color=C1, label='Tremor')
    ax.plot(f_axis, mean_v, color=C2, label='Voluntary rhythmic movement (VRM)')
    ax.fill_between(f_axis, mean_t, mean_v, where=(mean_t >= mean_v),
                    color=C1, alpha=0.2, interpolate=True, label='Tremor > VRM')
    ax.fill_between(f_axis, mean_t, mean_v, where=(mean_t < mean_v),
                    color=C2, alpha=0.2, interpolate=True, label='VRM > Tremor')
    ax.axvspan(low, high, alpha=0.1, color=C3, label=f'Investigated frequency band {low}-{high} Hz.')
    ax.axvline(low, color=C3)
    ax.axvline(high, color=C3)
    ax.set_xlim(0, 25)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power spectral density [V**2/Hz]')
    ax.legend()

    plt.savefig('figures/psd.pdf', dpi=150)
    plt.show()

    



def psd_comparison(t_idx, v_idx, tp_idx, fp_idx, tn_idx, fn_idx, X_test_all, low, high):
    """
    Compares TP/TN/FP/FN and all negatives and positives on averaged PSD values
    """

    all_band_powers = {}

    # Loop over the 6 cases
    fig, ax = plt.subplots(figsize=(8, 5))
    for idx_group, data, ls, label, color, alpha in [
        (tp_idx, X_test_all, '-', 'Correct tremor (TP)',  C1, 1),
        (fn_idx, X_test_all, '--', 'Missed tremor (FN)',   C5, 1),
        (fp_idx, X_test_all,  '--', 'Missed VRM (FP)',     C6, 1),
        (tn_idx, X_test_all,  '-', 'Correct VRM (TN)', C2, 1),
        (t_idx, X_test_all, '-', 'Voluntary rhytmic movement (VRM)', 'black', 0.5 ),
        (v_idx, X_test_all, '--', 'Tremor', 'black', 0.5)
    ]:
        
        # Calculate psds for every class
        band_powers, f_axis, psds = psd_single_class(idx_group, data, low, high)
        all_band_powers[label] = np.array(band_powers)
        mean_psd = np.mean(psds, axis=0)
        ax.plot(f_axis, mean_psd, color=color, label=label, linestyle = ls, alpha=alpha)
    
    ax.axvspan(3, 7, alpha=0.05, color=C3, label='Tremor band')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power spectral density')
    ax.legend()
    ax.set_xlim(0, 25)
    plt.savefig('figures/psd_classes.pdf', dpi=150)
    plt.show()
    


