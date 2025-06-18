from datetime import datetime, timedelta
import numpy as np
from zoneinfo import ZoneInfo
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from datetime import timedelta, datetime, date, timezone
from datetime import time as dttime

central_time = ZoneInfo('America/Chicago')
epoch=date(1970, 1, 1)
epoch_dt=datetime.combine(epoch, datetime.min.time(), tzinfo=timezone.utc)

def get_dbs_on_date(dbs_on_date: str) -> datetime.date:
    """
    Retrieves date that stimulation was turned on from info dict.

    Parameters:
    - pt_data (dictionary): patient info dictionary containing DBS on date in string format.
    
    Returns:
    - dbs_on_date (datetime.date): DBS on date in the datetime.date type.
    """
    return datetime.strptime(dbs_on_date, '%Y-%m-%d').date()

def add_empty_rows(group, pt_id, lead_location, time_bin_col='time_bin', dbs_on_date=None):
    """
    Fills in dataframe holes so that all possible empty time bins contain NaN. Useful for interpolation later.

    Parameters:
    - group (pd.DataFrame): DataFrame containing a single patient's data from a single lead.
    - lead_location (str): Lead location for this lead.
    - time_bin_col (str, optional): Name of the DataFrame column containing the datas' time bins timestamps.
    - dbs_on_date (datetime, optional): Date when the DBS was turned on. If provided, will fill the "days_since_dbs" column.

    Returns:
    - interp_df (pd.DataFrame): DataFrame containing empty rows where no data was recorded by the Percept device.
    """

    # Get sizes of gaps of missing data in terms of number of missing data points.
    gap_sizes = (np.diff(group[time_bin_col]) // timedelta(minutes=10)).astype(int)
    small_gap_start_inds = np.where(gap_sizes >= 2)[0]
    gap_sizes = gap_sizes[small_gap_start_inds]
    
    # Get the last time bin timestamp before each unfilled data gap so we know where to start filling from.
    gap_start_times = group.loc[group.index[small_gap_start_inds], time_bin_col]
    times_to_fill = [gap_start_time + timedelta(minutes=10) * i for (gap_start_time, gap_size) in zip(gap_start_times, gap_sizes) for i in range(1, gap_size)]
    
    # Create new dataframe and fill in information in relevant columns.
    if len(times_to_fill) != 0:
        interp_df = pd.DataFrame()
        interp_df['timestamp'] = times_to_fill # Timestamp is set to time bin
        interp_df[time_bin_col] = times_to_fill # Time bin is where the data was not recorded/missing from the device.
        interp_df['CT_timestamp'] = interp_df['timestamp'].dt.tz_convert(central_time)
        if dbs_on_date is not None:
            interp_df['days_since_dbs'] = [dt.days for dt in (interp_df['CT_timestamp'].dt.date - dbs_on_date)]
        interp_df['lead_location'] = lead_location # Use same lead model and location as original df.
        interp_df['lead_model'] = np.repeat(group.loc[group.index[small_gap_start_inds], 'lead_model'].values, gap_sizes-1)
        interp_df['pt_id'] = pt_id
        interp_df['source_file'] = 'interp' # Denote filled rows as interpolated so we know they aren't real data.
        interp_df['interpolated'] = True
        return interp_df

def zscore_group(group, cols_to_zscore=['lfp_left_raw', 'lfp_right_raw']):
    """
    Calculate Z-scored version of specified data.

    Parameters:
    - group (pd.DataFrame): group dataframe to Z-score across.
    - cols_to_zscore (list): columns to calculate and return Z-scored version of.

    Returns:
    - cols_z_scored_df (pd.DataFrame): DataFrame containing only Z-scored version of provided columns. Output is NaN wherever input was NaN.
    """
    new_cols = {}
    for col in cols_to_zscore:
        # Write new column name for easy merging later.
        if '_raw' in col:
            zscored_col_name = col.replace('_raw', '_z_scored')
        elif '_filled' in col:
            zscored_col_name = col.replace('_filled', '_z_scored')
        else:
            zscored_col_name = col + '_z_scored'

        if pd.notna(group[col]).sum() > 1:
            # Z-score data: for each data point, subtract mean and divide by standard deviation of entire series.
            mean = np.nanmean(group[col], axis=0)
            std = np.nanstd(group[col], axis=0)
            new_cols[zscored_col_name] = (group[col] - mean) / std if std != 0 else [np.nan] * len(group[col])
        else:
            # If all values are nan, return np.nan
            new_cols[zscored_col_name] = [np.nan] * len(group)
    
    # Create new dataframe containing only the new Z-scored columns and return it. Keep the same indices as the input group for easy merging.
    return pd.DataFrame(new_cols, index=group.index)

def gaussian_smooth(data: pd.Series, sigma: float=3.0) -> pd.Series:
    """
    Apply Gaussian smoothing to a pandas Series.
    
    Parameters:
        data (pd.Series): Input data to smooth.
        sigma (float, optional): Standard deviation for Gaussian kernel. Default is 3.0.
    
    Returns:
        pd.Series: Smoothed data.
    """
    if len(data) < 2:
        return data
    if data.isna().all():
        return pd.Series([np.nan] * len(data), index=data.index)
    interpolated = data.interpolate(method='linear', limit_direction='both')
    smoothed = gaussian_filter1d(interpolated.values, sigma=sigma, mode='wrap')
    return pd.Series(smoothed, index=data.index)

def transform_timestamp_to_days(pt_df, ax, tick_spacing_days=None, rotation=0):
    """
    Transforms the x-axis of a plot to represent days since DBS.

    Parameters:
    - pt_df (pd.DataFrame): DataFrame containing patient data with 'CT_timestamp' and 'days_since_dbs' columns.
    - ax (matplotlib.axes.Axes): The axes object to modify.
    - tick_spacing_days (int, optional): The spacing between ticks in days. If None, it will be calculated based on the data range.
    - rotation (float): Rotate the tick labels by this number of degrees.
    """

    xlim_orig = ax.get_xlim()
    days_range = pt_df['days_since_dbs'].max() - pt_df['days_since_dbs'].min()
    if tick_spacing_days is None:
        if days_range < 4:
            tick_spacing_days = 1
        elif days_range < 7:
            tick_spacing_days = 2
        elif days_range < 30:
            tick_spacing_days = 5
        elif days_range < 60: 
            tick_spacing_days = 10
        elif days_range < 100:
            tick_spacing_days = 30
        elif days_range < 250:
            tick_spacing_days = 50
        elif days_range < 500:
            tick_spacing_days = 100
        else:
            tick_spacing_days = 150

    dbs_on_date = pt_df.iloc[0]['CT_timestamp'].date() - timedelta(days=int(pt_df.iloc[0]['days_since_dbs']))
    dbs_on_since_epoch = (dbs_on_date - epoch).days

    min_time = pt_df['CT_timestamp'].min()
    min_time_midnight = min_time.replace(hour=0, minute=0, second=0, microsecond=0)
    min_time_midnight_as_days_since = (min_time_midnight - epoch_dt) / timedelta(days=1)
    max_time = pt_df['CT_timestamp'].max()
    max_time_midnight = max_time.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    max_time_midnight_as_days_since = (max_time_midnight - epoch_dt) / timedelta(days=1)

    min_fractional_offset = min_time_midnight_as_days_since % 1
    max_fractional_offset = max_time_midnight_as_days_since % 1
    min_tick_adjusted = int(np.floor((min_time_midnight_as_days_since - dbs_on_since_epoch) / tick_spacing_days)) * tick_spacing_days + min_fractional_offset + dbs_on_since_epoch
    max_tick_adjusted = int(np.floor((max_time_midnight_as_days_since - dbs_on_since_epoch) / tick_spacing_days)) * tick_spacing_days + max_fractional_offset + dbs_on_since_epoch

    if max_tick_adjusted < max_time_midnight_as_days_since:
        max_tick_adjusted += tick_spacing_days
    
    ticks = np.arange(min_tick_adjusted, max_tick_adjusted + 1, tick_spacing_days)
    tick_labels = [int(np.round(tick - dbs_on_since_epoch)) for tick in ticks]
    # ax.set_xlim(min_tick_adjusted, max_tick_adjusted)
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels, rotation=rotation)
    ax.set_xlim(xlim_orig[0], xlim_orig[1])

def round_time_up(t: dttime, num_secs=600) -> dttime:
    total_seconds = t.hour * 3600 + t.minute * 60 + t.second
    rounded_seconds = (total_seconds + (num_secs-1)) // num_secs * num_secs
    if rounded_seconds >= 86400:
        return dttime(0, 0, 0)
    else:
        rounded_hour = rounded_seconds // 3600
        rounded_minute = (rounded_seconds % 3600) // 60
        return dttime(rounded_hour, rounded_minute, 0)

def interpolate_x_from_y(x_vals, y_vals, y_target):
    """
    Interpolates the x value corresponding to a given y value using linear interpolation.
    
    Parameters:
    - x_vals: Array of x values.
    - y_vals: Array of y values.
    - y_target: The y value for which to find the corresponding x value.

    Returns:
    - x_target: The interpolated x value corresponding to y_target.
    """
    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)

    # Eliminate indices where x_vals is equal to 0
    non_zero_indices = x_vals != 0
    x_vals = x_vals[non_zero_indices]
    y_vals = y_vals[non_zero_indices]

    # Ensure inputs are sorted by y (ascending)
    sort_idx = np.argsort(y_vals)
    x_vals = x_vals[sort_idx]
    y_vals = y_vals[sort_idx]

    # Find index where y_target would be inserted
    idx = np.searchsorted(y_vals, y_target)

    if idx == 0 or idx == len(y_vals):
        raise ValueError("y_target is out of bounds")

    return x_vals[idx]

def make_violin_plot_pretty(parts, color, median, ax, alpha=0.5):
    """
    Enhances the appearance of a violin plot by customizing its body color, transparency, 
    and adding a horizontal median line.

    Parameters:
    - parts: The components of the violin plot, obtained from the `ax.violinplot()` method.
    - color: The color to set for the violin body.
    - median: The y-coordinate of the median line to be drawn.
    - ax: The axes object where the violin plot is drawn.
    - alpha: The transparency level for the violin body (default is 0.5).
    """

    # Customize body
    body = parts['bodies'][0]
    body.set_facecolor(color)
    body.set_edgecolor(color)

    vertices = parts['bodies'][0].get_paths()[0].vertices
    ax.plot(vertices[:, 0], vertices[:, 1], color=color, lw=0.5, alpha=np.mean([alpha, alpha, 1.0]))
    x, y = vertices[:, 0], vertices[:, 1]

    x_target = interpolate_x_from_y(x, y, median)
    if np.isclose(x.min(), 0) or np.isclose(x.max(), 0):
        ax.hlines(median, xmin=x_target, xmax=0, color=color, lw=1, alpha=np.mean([alpha, alpha, 1.0]))
    else:
        ax.hlines(median, xmin=x_target, xmax=-x_target, color=color, lw=1, alpha=np.mean([alpha, alpha, 1.0]))
