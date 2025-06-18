import pandas as pd
import numpy as np
from scipy.interpolate import PchipInterpolator

def threshold_outliers(group: pd.DataFrame, cols_to_fill: list, threshold: float=((2**32)-1)/60) -> pd.DataFrame:
    """
    Replace outliers in the group DataFrame with NaN. These may then be left empty or optionally interpolated in a supplemental step.
    
    Parameters:
        group (pd.DataFrame): Input DataFrame with potential outliers.
        cols_to_fill (list): List of column names to fill outliers in.
        threshold (float, optional): Threshold to define the outliers. Default is max 32-bit int value divided by 60.
    
    Returns:
        pd.DataFrame: DataFrame with outliers filled.
    """
    new_cols = [(col[:-4] if col.endswith('_raw') else col) + '_threshold' for col in cols_to_fill]
    result = pd.DataFrame(index=group.index, columns=new_cols)
    result[new_cols] = group[cols_to_fill].values.copy()  # Copy original values to result DataFrame
    for col, new_col in zip(cols_to_fill, new_cols):
        mask_over_threshold = group[col] >= threshold
        result.loc[mask_over_threshold, new_col] = np.nan
    return result

def fill_outliers_OvER(group: pd.DataFrame, cols_to_fill: list):
    """
    Replace outliers in the group DataFrame caused by overvoltage readings. When the Percept device records a LFP value above its
    acceptable range, it places the maximum integer value in its place. Then, when the 10 minute interval is averaged,
    the abnormally high value dominates the average and causes non-physical outliers in the data. When multiple overages
    are observed in a single 10 minute interval, the outlier is even higher. Here, we estimate how many overages were
    recorded during each 10 minute interval, then remove them and recalculate the averaged LFP without the abnormal values.
    The overvoltage recordings may be caused by movement artifacts or something else.

    Parameters:
        - group (pd.DataFrame): DataFrame containing contiguous LFP data, potentially with outliers and holes.
        - cols_to_fill (list): List of column names to fill outliers in.

    Returns:
        - pd.DataFrame: DataFrame with outliers filled and containing number of overages in each cell.
    """
    new_filled_cols = [(col[:-4] if col.endswith('_raw') else col) + '_OvER' for col in cols_to_fill]
    new_num_overages_cols = [(col[:-4] if col.endswith('_raw') else col) + '_num_overages' for col in cols_to_fill]
    result_df = pd.DataFrame(index=group.index, columns=new_filled_cols + new_num_overages_cols)
    n = 60 # Number of samples per 10 minute average
    v = 2**32 - 1

    for col, new_filled_col, new_num_overages_col in zip(cols_to_fill, new_filled_cols, new_num_overages_cols):
        data = group[col].values
        num_overages = data // (v/n) # Estimate how many voltage overages we had during each 10 minute interval

        # If all samples within the interval are overages, place a NAN in. This will be filled in later when the missing values are filled.
        # This edge case never actually happens in our dataset, but we handle it just in case.
        valid_mask = num_overages < n
        corrected_data = np.empty_like(data, dtype=float)
        corrected_data[valid_mask] = (n * data[valid_mask] - v * num_overages[valid_mask]) / (n - num_overages[valid_mask])
        corrected_data[~valid_mask] = np.nan
        # print(corrected_data)

        result_df[new_filled_col] = corrected_data
        result_df[new_num_overages_col] = num_overages

    return result_df


def interpolate_holes(group: pd.DataFrame, cols_to_fill: list, max_gap: int=12) -> pd.DataFrame:
    """
    Fill missing values (NaNs) in the specified columns of the group DataFrame using PCHIP interpolation, for gaps up to max_gap size.
    
    Parameters:
        group (pd.DataFrame): Input DataFrame with missing values (NaNs).
        cols_to_fill (list): List of column names to fill.
        max_gap (int, optional): Maximum gap size to fill. Default is 12 (2 hours).
    
    Returns:
        pd.DataFrame: DataFrame with missing values filled in the specified columns.
    """
    new_cols = [col + '_interpolate' for col in cols_to_fill]
    filled_df = pd.DataFrame(index=group.index, columns=new_cols)
    filled_df[new_cols] = group[cols_to_fill].values.copy()  # Copy original values to filled DataFrame
    for new_col in new_cols:
        # Identify NaN indices
        nan_indices = np.where(filled_df[new_col].isna())[0]
        not_nan_indices = np.where(filled_df[new_col].notna())[0]
        valid_values = filled_df.loc[filled_df.index[not_nan_indices], new_col].values

        if len(not_nan_indices) < 2: # Not enough data to interpolate
            filled_df[new_col] = np.nan
            continue
        if len(nan_indices) == 0: # Nothing to interpolate
            continue

        # Create the PCHIP interpolator
        interpolator = PchipInterpolator(not_nan_indices, valid_values)
        gaps = np.split(nan_indices, np.where(np.diff(nan_indices) != 1)[0] + 1)
        if len(filled_df) - 1 in gaps[-1]:
            gaps = gaps[:-1]
        if (len(gaps) > 0) and (0 in gaps[0]):
            gaps = gaps[1:]

        for gap in gaps:
            if len(gap) <= max_gap:
                filled_df.loc[filled_df.index[gap], new_col] = interpolator(gap)
        
    return filled_df
