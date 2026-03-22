import pandas as pd

file_path = '../Data/GSE282122_metadata.txt'

# Initialize a list of dictionaries to construct the transposed DataFrame
samples = []

with open(file_path, 'r') as f:
    for line in f:
        if line.startswith('!Sample_'):
            # Split the tab-separated file and remove quotes
            parts = [p.strip().strip('"') for p in line.strip().split('\t')]
            key = parts[0].replace('!Sample_', '')
            values = parts[1:]
            
            # Initialize a dictionary for each sample column on the first pass
            if not samples:
                samples = [{} for _ in range(len(values))]
            
            # Populate attributes into the correct sample's dictionary
            for i, val in enumerate(values):
                if i < len(samples):
                    # Expand the embedded key-value traits into distinct columns
                    if key == 'characteristics_ch1' and ':' in val:
                        sub_key, sub_val = val.split(':', 1)
                        samples[i][sub_key.strip()] = sub_val.strip()
                    else:
                        samples[i][key] = val

# Create the full structured DataFrame
df = pd.DataFrame(samples)

# Apply study-specific constraints: Crohn's Disease (CD) and Baseline (Pre-treatment)
df_filtered = df[(df['disease'] == 'CD') & (df['treatment'] == 'Pre')]

# Isolate the high-value features for the prediction target and demographic normalization
columns_to_keep = [
    'geo_accession', 'title', 'patient', 'site', 'disease duration', 
    'inflammation', 'inflammation score', 'age', 'Sex', 'cellsloaded', 'batch'
]

final_cols = [c for c in columns_to_keep if c in df_filtered.columns]
df_final = df_filtered[final_cols]

# Export to working directory
df_final.to_csv('CD_PreTreatment_Metadata.csv', index=False)
print("Data successfully converted and saved to CD_PreTreatment_Metadata.csv!")