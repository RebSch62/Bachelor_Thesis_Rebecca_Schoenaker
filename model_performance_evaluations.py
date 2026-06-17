
import numpy as np
import json
import torch
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.metrics import roc_curve, auc

from braindecode.models import EEGNet


from training import *
from config import *



def plot_roc_curves(probs, labels, outer_results):
    """
    Visualises ROC curves based on calculated ROC-AUC scores for each outer fold.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    base_fpr = np.linspace(0, 1, 101)
    tprs     = []

    # Loop over the folds
    for fold_i, (p, l) in enumerate(zip(probs, labels)):

        # Calculate the ROC Curve and AUC score
        fpr, tpr, _ = roc_curve(l, p)
        fold_auc = auc(fpr, tpr)

        # interpolate for nicer curve
        tpr_interp  = np.interp(base_fpr, fpr, tpr)
        tprs.append(tpr_interp)
        ax.plot(fpr, tpr, alpha=0.3, color = f'C{fold_i%5}', label=f'Fold {fold_i+1} (AUC={fold_auc:.2f})')

    # Take the mean
    tprs  = np.mean(tprs, axis=0)
    mean_auc  = np.mean(outer_results)
    std_auc = np.std(outer_results)

    ax.plot(base_fpr, tprs, 'b-', lw=2, label=f'Mean (AUC={mean_auc:.2f} +/- {std_auc:.2f})')
    ax.plot([0,1],[0,1], 'k--', label='Chance', alpha=0.5)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(fontsize=7, loc='lower right')

    plt.savefig('figures/roc_curve.pdf', dpi=150)
    plt.show()


def permutation_test(X, y, outer_folds, mean_roc, fold, n_permutations=300):
    """
    Run the permutation test for significance comparison.
    """
    perm_aucs = []

    # Load fold-specific params
    path = DIR / f'model_fold{fold+1}.pth'
    state = torch.load(path, map_location=DEVICE, weights_only=True)
    actual_kernel = state['conv_temporal.weight'].shape[-1]

    with open(DIR / f'best_params.json') as f:
        params = json.load(f)[fold]

    # Pick just one fold
    train_idx, test_idx = outer_folds[fold]

    # Loop over the number of permutations
    for p in range(n_permutations):
        print("Number of permutation: ", p)
        y_p = np.random.permutation(y)
        permutation_folds_auc = []

        perm_model = EEGNet(
            n_chans=3, n_outputs=2, n_times=400,
            F1=8, D=2,
            kernel_length=actual_kernel,
            drop_prob=params['drop_prob']
        ).to(DEVICE)

        roc, *_ = train_evaluate(perm_model, X[train_idx], y_p[train_idx], X[test_idx],  y_p[test_idx],
            lr=params['lr'], batch_size=BATCH_SIZE, epochs=EPOCHS_INNER, device=DEVICE)
        permutation_folds_auc.append(roc)

        del perm_model
        torch.cuda.empty_cache()

        # Save the aucs
        perm_aucs.append(np.mean(permutation_folds_auc))
        print(f"Permutation {p+1}: AUC={perm_aucs[-1]:.3f}")

    # Find the CI threshold
    threshold = np.percentile(perm_aucs, 95)
    p_perm    = np.mean(np.array(perm_aucs) >= mean_roc)
    print(f"95th percentile: {threshold:.3f}, p={p_perm:.4f}")

    return perm_aucs, threshold




def permutation_distribution(X,y, outer_folds, mean_roc, fold):
    """
    Visualises an estimate of the null distribution.
    """

    # Perform the permutation test
    perm_aucs, threshold = permutation_test(X,y,outer_folds,mean_roc, fold,300)
    with open(DIR/"perm_aucs.json", "w") as f:
        json.dump(perm_aucs, f)
    with open(DIR/"perm_aucs.json", "w") as f:
        json.dump(perm_aucs, f)
    fig, ax = plt.subplots(figsize=(8, 4), layout='constrained')
    sns.histplot(perm_aucs,bins=10, color=C1, alpha=0.6, ax=ax, label ='Permuted AUCs')
    plt.axvline(mean_roc,  color=C3,    lw=2, alpha=0.7, label=f'Real AUC={mean_roc:.3f}')
    plt.axvline(threshold, color=C4, lw=2, alpha=0.7, linestyle='--', label=f'95th percentile={threshold:.3f}')
    ax.axvline(0.5, color='black', alpha=0.7, lw=1, linestyle='--', label='Chance level = 0.5')
    plt.xlim(0.40, 1.0)
    plt.xlabel('ROC-AUC')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig('figures/permutation_test2.pdf')
    plt.show()


def loss_curves_plot(all_train_loss, all_val_loss):
    """
    Visualises training and validation curves.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout='constrained', sharey=True)
    for fold_train, fold_val in zip(all_train_loss, all_val_loss):
        axes[0].plot(fold_train, alpha=0.4, color=C1)
        axes[1].plot(fold_val, alpha=0.4, color=C2)
    axes[0].set_title('Training loss per fold')
    axes[1].set_title('Validation loss per fold')
    for ax in axes:
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
    plt.savefig('figures/loss_curves.pdf', dpi=150)
    plt.show()
