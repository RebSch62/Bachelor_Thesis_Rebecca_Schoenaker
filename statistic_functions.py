import numpy as np
from scipy.stats import mannwhitneyu

from config import *


def effect_size_and_CI(group1, group2, n_boot=1000, ci=95):
    """
    Compute effect size with bootstrap confidence interval.
    """
    n1, n2 = len(group1), len(group2)
    
    # Observed effect size
    U, _ = mannwhitneyu(group1, group2, alternative='two-sided')
    effect_size = U / (n1 * n2)
    
    # Bootstrap CI
    boot_cles = []
    for _ in range(n_boot):
        b1 = np.random.choice(group1, n1, replace=True)
        b2 = np.random.choice(group2, n2, replace=True)
        U_boot, _ = mannwhitneyu(b1, b2, alternative='two-sided')
        boot_cles.append(U_boot / (n1 * n2))
    
    ci_low  = np.percentile(boot_cles, (100 - ci) / 2)
    ci_high = np.percentile(boot_cles, 100 - (100 - ci) / 2)
    
    return effect_size, ci_low, ci_high


def run_stats(t_sample, v_sample, title=""):
    """
    Runs Mann-Whitney U statistics.
    """
    stat, p_val = mannwhitneyu(t_sample, v_sample, alternative='two-sided')
    effect_size, ci_low, ci_high = effect_size_and_CI(t_sample, v_sample)
    print(f"\n      {title}Mann-Whitney U: stat={stat:.3f}, p={p_val:.4f}")
    print(f"        Effect size = {effect_size:.3f} [95% CI: {ci_low:.3f}, {ci_high:.3f}]")
    print(f"        Difference between tremor ({np.mean(t_sample)}) and VRM ({np.mean(v_sample)}) power values: {np.mean(t_sample)-np.mean(v_sample)}.")

    



