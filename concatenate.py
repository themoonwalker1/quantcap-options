import os
import pandas as pd

# Define the output file name
output_file = "consolidated_data.csv"

# Initialize a list to store merged DataFrames from all subfolders
all_subfolder_data = []

# Traverse through all subfolders in the current directory
for root, dirs, files in os.walk(os.getcwd()):
    infotable_df = None
    coverpage_df = None
    print(root)

    for file in files:
        file_path = os.path.join(root, file)

        # Process INFOTABLE.tsv
        if file == "INFOTABLE.tsv":
            try:
                print(f"Processing: {file_path}")
                infotable_df = pd.read_csv(file_path, sep="\t", low_memory=False)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

        # Process COVERPAGE.tsv
        if file == "COVERPAGE.tsv":
            try:
                print(f"Processing: {file_path}")
                coverpage_df = pd.read_csv(file_path, sep="\t", low_memory=False)
                # Keep only necessary columns
                coverpage_df = coverpage_df[["ACCESSION_NUMBER", "FILINGMANAGER_NAME", "DATEREPORTED"]]
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    # Merge INFOTABLE and COVERPAGE on ACCESSION_NUMBER if both are present
    if infotable_df is not None and coverpage_df is not None:
        try:
            merged_subfolder_data = pd.merge(infotable_df, coverpage_df, on="ACCESSION_NUMBER", how="inner")
            all_subfolder_data.append(merged_subfolder_data)
        except Exception as e:
            print(f"Error merging files in {root}: {e}")

# Combine all merged subfolder DataFrames into a single DataFrame
if all_subfolder_data:
    consolidated_data = pd.concat(all_subfolder_data, ignore_index=True)
    consolidated_data.to_csv(output_file, index=False)
    print(f"Consolidation complete. File saved as {output_file}")
else:
    print("No data to process.")
