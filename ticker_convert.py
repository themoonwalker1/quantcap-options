# Reference CSV download here: https://github.com/johnflem/cusip_tickers/blob/main/cusip_tickers.csv

import pandas as pd

# Read the reference csv 
df_ref = pd.read_csv("/Users/jinschofield/Downloads/cusip_tickers.csv", dtype=str)

# Check columns exist
if not {"CUSIP", "Ticker"}.issubset(df_ref.columns):
    raise ValueError("reference csv has to have 'CUSIP' and 'Ticker' columns.")

# Conver to dict
cusip_to_ticker = df_ref.set_index("CUSIP")["Ticker"].to_dict()

original_csv = "/Users/jinschofield/Downloads/consolidated_data2.csv"
output_csv   = "/Users/jinschofield/Downloads/consolidated_data3.csv"

chunksize = 10**5
first_chunk = True

with pd.read_csv(original_csv, dtype=str, chunksize=chunksize) as reader:
    for chunk_index, chunk in enumerate(reader):
        print(f"processing chunk {chunk_index+1}...")

        
        if "CUSIP" not in chunk.columns:
            raise ValueError("original csv is missing a 'CUSIP' column.")

        # convert cusip to ticker 
        chunk["CUSIP"] = chunk["CUSIP"].map(cusip_to_ticker).fillna(chunk["CUSIP"])

        
        chunk.rename(columns={"CUSIP": "Ticker"}, inplace=True)

        mode   = 'w' if first_chunk else 'a'
        header = first_chunk
        chunk.to_csv(output_csv, index=False, mode=mode, header=header)


        if first_chunk:
            first_chunk = False

print(f"updated csv is in '{output_csv}'.")

