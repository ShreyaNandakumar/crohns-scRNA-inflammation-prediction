import os
import shutil
import pandas as pd

# paths
metadata_file = "CD_PreTreatment_Metadata.csv"
source_dir = "/Users/shreyanandakumar/Downloads/filtered_processed_data"
target_dir = "/Users/shreyanandakumar/Downloads/selected_49_samples"

# make target folder
os.makedirs(target_dir, exist_ok=True)

# read metadata
meta = pd.read_csv(metadata_file)

# convert title like CID003378-1-reup -> CID003378-1
meta["folder_name"] = meta["title"].str.replace("-reup", "", regex=False)

copied = []
missing = []

for folder in meta["folder_name"]:
    src = os.path.join(source_dir, folder)
    dst = os.path.join(target_dir, folder)

    if os.path.exists(src):
        if not os.path.exists(dst):
            shutil.copytree(src, dst)
        copied.append(folder)
    else:
        missing.append(folder)

print(f"Copied {len(copied)} folders.")
print(f"Missing {len(missing)} folders.")

if missing:
    print("Missing folders:")
    for x in missing:
        print(x)