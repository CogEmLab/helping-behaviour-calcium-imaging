import pandas as pd
import numpy as np

def update_table(table):
    # Create a new column with np.nan for Identity Switch
    table['Identity Switch'] = np.nan
    # Create a new column for Identity Switch Value
    table['Identity Switch Value'] = np.nan
    # Create a new column for Preserved Identity Value
    table['Preserved Identity Value'] = np.nan

    # Get only tracked data
    tracked_only = table[table['Tracked'] == True]

    # Iterate over all tracked data
    for i, row in tracked_only.iterrows():
        # Get the index of the current row
        idx = row['Index']
        # Get the Reg ID of the current row
        reg_id = row['Reg ID']
        # Get only rows with the same Reg ID
        rows = tracked_only[tracked_only['Reg ID'] == reg_id]
        # Convert day to int
        rows['Day'] = rows['Day'].astype(int)
        # Sort rows according to the day
        rows = rows.sort_values('Day')
        # Go through all rows starting from the second
        for j in range(1, len(rows)):
            # Get the previous row
            prev_row = rows.iloc[j-1]
            # Get the current row
            curr_row = rows.iloc[j]
            # Compare groups of the previous and current row
            if prev_row['Group'] != curr_row['Group']:
                # If groups are different, set Identity Switch to True
                tracked_only.loc[curr_row.name, 'Identity Switch'] = True
                # Assign Identity Switch Value
                tracked_only.loc[curr_row.name, 'Identity Switch Value'] = assign_identity_switch(prev_row['Group'], curr_row['Group'])
            else:
                # If groups are the same, set Identity Switch to False
                tracked_only.loc[curr_row.name, 'Identity Switch'] = False
                # Assign Identity Switch Value
                tracked_only.loc[curr_row.name, 'Identity Switch Value'] = 0
                # Assign Preserved Identity Value
                tracked_only.loc[curr_row.name, 'Preserved Identity Value'] = curr_row['Group']
                
    # Update table 
    table.loc[tracked_only.index, 'Identity Switch'] = tracked_only['Identity Switch']
    table.loc[tracked_only.index, 'Identity Switch Value'] = tracked_only['Identity Switch Value']
    table.loc[tracked_only.index, 'Preserved Identity Value'] = tracked_only['Preserved Identity Value']

    return table

# Define identity switch assignment function
def assign_identity_switch(prev_value, new_value):
    if prev_value == new_value:
        print('here')
        print('---------------------------------------------------------')
        return 0
    elif prev_value == 'Pre-responsive' and new_value == 'Post-responsive':
        return 1
    elif prev_value == 'Post-responsive' and new_value == 'Pre-responsive':
        return 2
    elif prev_value == 'Pre-responsive' and new_value == 'Mixed':
        return 3
    elif prev_value == 'Mixed' and new_value == 'Pre-responsive':
        return 4
    elif prev_value == 'Post-responsive' and new_value == 'Mixed':
        return 5
    elif prev_value == 'Mixed' and new_value == 'Post-responsive':
        return 6

# Load data
metadata = pd.read_csv('processed-data/metadata_cross_comp.csv')
# data = np.load('data/dHPC/collected_data.npy')

# Assign day according to the scheme
session_to_day = {
    'HBTD1': 1,
    'HBTD2': 2,
    'HBTD3': 3,
    'HBTD4': 4,
    'HBTD5': 5,
    'HBTD6': 6,
    'RECNvlCtx': 7,
    'Stranger': 7,
    'REMNvlCtx': 7,
    'REMHBTD': 7}

# Assign day to metadata
metadata['Day'] = metadata['Session'].map(session_to_day)

# Print where nan
print(metadata[metadata['Day'].isna()])

# Create a column 'Tracked' in metadata, which is True if Reg ID is in mask
metadata['Tracked'] = False

# Create a new column with np.nan for Identity Switch
metadata['Identity Switch'] = np.nan

# Create a new column for Identity Switch Value
metadata['Identity Switch Value'] = np.nan

# Create a new column for Identity Switch Value
metadata['Preserved Identity Value'] = np.nan

# Get all unique mouse IDs
mouse_ids = metadata['Mouse'].unique()
print(f'Number of unique mouse IDs: {len(mouse_ids)}')

# Iterate over all mouse IDs
for mouse_id in mouse_ids:
    # Get all data for the current mouse ID
    mouse_metadata = metadata[metadata['Mouse'] == mouse_id]
    print(f'Mouse ID: {mouse_id}, number of data points: {mouse_metadata.shape[0]}')

    reg_ids = mouse_metadata['Reg ID'].values
    ids, counts = np.unique(reg_ids, return_counts=True)
    print(f'Number of unique Reg IDs: {len(ids)}')

    mask = counts != 1
    # Add column tracked to metadata, which is True if Reg ID is in mask
    mouse_metadata['Tracked'] = mouse_metadata['Reg ID'].isin(ids[mask])
    # Update metadata
    metadata.loc[metadata['Mouse'] == mouse_id, 'Tracked'] = mouse_metadata['Tracked']

    # Update metadata with Identity Switch
    mouse_metadata = update_table(mouse_metadata)
    print(mouse_metadata['Preserved Identity Value'].value_counts())
    # Update metadata
    metadata.loc[metadata['Mouse'] == mouse_id, 'Identity Switch'] = mouse_metadata['Identity Switch']
    metadata.loc[metadata['Mouse'] == mouse_id, 'Identity Switch Value'] = mouse_metadata['Identity Switch Value']
    metadata.loc[metadata['Mouse'] == mouse_id, 'Preserved Identity Value'] = mouse_metadata['Preserved Identity Value']

# Save metadata
metadata.to_csv('processed-data/metadata_cross_comp_updated.csv', index=False)

        