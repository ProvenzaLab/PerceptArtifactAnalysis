from statsmodels.tsa.stattools import acf
import numpy as np
from scipy import stats

def estimate_effective_sample_size(data, nlags=200):
    """
    Estimate effective sample size based on autocorrelation.
    Formula:
        Neff ≈ N / (1 + 2 * sum_{k=1}^{K} rho_k)
    where rho_k is the autocorrelation at lag k.
    """
    N = len(data)
    acf_vals = acf(data, nlags=nlags, fft=False)
    
    # sum positive autocorrelations until first negative
    sum_rho = 0
    for k in range(1, len(acf_vals)):
        if acf_vals[k] <= 0:
            break
        sum_rho += acf_vals[k]

    neff = N / (1 + 2 * sum_rho)
    return int(neff)

def downsample_to_eff_size(data, eff_N):
    """
    Downsample data to approximate independence by selecting every k-th point.
    k = round(len(data) / eff_N)
    """
    k = max(1, round(len(data) / eff_N))
    return data[::k]

def ks_test_with_downsampling(data1, data2, nlags=200):
    """
    Performs a KS test after downsampling each dataset to its estimated effective sample size.
    """
    neff1 = estimate_effective_sample_size(data1, nlags)
    neff2 = estimate_effective_sample_size(data2, nlags)

    ds1 = downsample_to_eff_size(np.asarray(data1), neff1)
    ds2 = downsample_to_eff_size(np.asarray(data2), neff2)

    ks_result = stats.ks_2samp(ds1, ds2)

    return {
        'KS_stat': ks_result.statistic,
        'p_value': ks_result.pvalue,
        'n1_eff': neff1,
        'n2_eff': neff2,
        'n1_actual': len(ds1),
        'n2_actual': len(ds2)
    }

def spearman_with_downsampling(data1, data2, nlags=200):
    """
    Computes Spearman correlation between two autocorrelated variables
    after downsampling each to their effective sample size.

    Parameters
    ----------
    data1, data2 : array-like
        Aligned 1D arrays of the same length.
    nlags : int
        Maximum lag to use in autocorrelation for ESS estimation.

    Returns
    -------
    dict with:
        - spearman_r: Spearman rank correlation
        - p_value: two-sided p-value
        - n1_eff: effective sample size of data1
        - n2_eff: effective sample size of data2
        - n_actual: length of downsampled arrays (minimum of both)
    """
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)

    if len(data1) != len(data2):
        raise ValueError("Input arrays must be the same length and time-aligned.")

    n1_eff = estimate_effective_sample_size(data1, nlags)
    n2_eff = estimate_effective_sample_size(data2, nlags)
    n_eff = min(n1_eff, n2_eff)

    ds1 = downsample_to_eff_size(data1, n_eff)
    ds2 = downsample_to_eff_size(data2, n_eff)

    # Truncate to match length if unequal due to rounding
    n_actual = min(len(ds1), len(ds2))
    ds1 = ds1[:n_actual]
    ds2 = ds2[:n_actual]

    rho, p = stats.spearmanr(ds1, ds2)

    return {
        'spearman_r': rho,
        'p_value': p,
        'n1_eff': n1_eff,
        'n2_eff': n2_eff,
        'n_actual': n_actual
    }
