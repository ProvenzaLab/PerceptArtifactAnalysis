import numpy as np
import pandas as pd
import json

def get_json_filenames(root_dir):
    """
    Retrieves a list of all json files containing LFP data in the provided root directory.

    Parameters:
        root_dir (pathlib.Path): Directory from which to get JSON descendents.

    Returns:
        all_files (list): JSON descendents of root directory containing LFP data.
    """
    if not root_dir.exists():
        return []
    
    all_files = []
    if np.any([f.is_dir() for f in root_dir.iterdir()]):
        for subdir in root_dir.iterdir():
            if not subdir.is_dir():
                continue
            p = subdir.glob('[!.]*.json')
            # all_files.extend([x for x in p if x.name[:3] == x.parents[2].name])
            all_files.extend(p)
    else:
        p = root_dir.glob('[!.]*.json')
        all_files.extend(p)
    return all_files


def chronic_lfp_from_json(filename) -> pd.DataFrame:
    """
    Processes a Percept JSON file to extract chronic LFP power data recorded with BrainSense Timeline and returns it as a DataFrame.

    Parameters:
        filename (pathlib.Path): Path to the JSON file.
    
    Returns:
        processed_df (pd.DataFrame): DataFrame containing the processed LFP data.
    """

    with open(filename, 'r') as f:
        js = json.load(f)
    if 'DiagnosticData' not in js or 'LFPTrendLogs' not in js['DiagnosticData']:
        return pd.DataFrame()
    
    left_lead_config = next((d for d in js['LeadConfiguration']['Final'] if d['Hemisphere'] == 'HemisphereLocationDef.Left'), None)
    right_lead_config = next((d for d in js['LeadConfiguration']['Final'] if d['Hemisphere'] == 'HemisphereLocationDef.Right'), None)
    left_lead_location = left_lead_config['LeadLocation'].split('.')[-1].upper() if left_lead_config else None
    right_lead_location = right_lead_config['LeadLocation'].split('.')[-1].upper() if right_lead_config else None
    left_lead_model = left_lead_config['Model'].split('.')[-1].upper() if left_lead_config else None
    right_lead_model = right_lead_config['Model'].split('.')[-1].upper() if right_lead_config else None

    if left_lead_location in ['VCVS', 'VC/VS', 'VC', 'VS', 'AIC', 'ALIC', 'BNST']:
        left_lead_location = 'VC/VS'
    if right_lead_location in ['VCVS', 'VC/VS', 'VC', 'VS', 'AIC', 'ALIC', 'BNST']:
        right_lead_location = 'VC/VS'
    
    data = js['DiagnosticData']['LFPTrendLogs']
    data_left, data_right = [], []
    
    if 'HemisphereLocationDef.Left' in data:
        for key in data['HemisphereLocationDef.Left']:
            data_left.extend(data['HemisphereLocationDef.Left'][key])
    
    if 'HemisphereLocationDef.Right' in data:
        for key in data['HemisphereLocationDef.Right']:
            data_right.extend(data['HemisphereLocationDef.Right'][key])
    
    if len(data_left) > 0:
        left_timestamp, left_lfp, left_stim = map(list, zip(*((d['DateTime'], d['LFP'], d['AmplitudeInMilliAmps']) for d in data_left)))
    if len(data_right) > 0:
        right_timestamp, right_lfp, right_stim = map(list, zip(*((d['DateTime'], d['LFP'], d['AmplitudeInMilliAmps']) for d in data_right)))

    if len(data_left) > 0:
        df_left = pd.DataFrame({
            'timestamp': left_timestamp,
            'lfp_left': left_lfp,
            'stim_left': left_stim,
        })
    else:
        df_left = pd.DataFrame({
            'timestamp': right_timestamp,
            'lfp_left': np.nan,
            'stim_left': np.nan
        })
    if len(data_right) > 0:
        df_right = pd.DataFrame({
            'timestamp': right_timestamp,
            'lfp_right': right_lfp,
            'stim_right': right_stim,
        })
    else:
        df_right = pd.DataFrame({
            'timestamp': left_timestamp,
            'lfp_right': np.nan,
            'stim_right': np.nan
        })
    
    final_df = pd.merge(df_left, df_right, on='timestamp', how='outer')
    
    final_df['source_file'] = filename.name
    final_df['left_lead_location'] = left_lead_location
    final_df['right_lead_location'] = right_lead_location
    final_df['left_lead_model'] = left_lead_model
    final_df['right_lead_model'] = right_lead_model
    return final_df


