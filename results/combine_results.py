import os 

# combine all results in results/ into a single csv file
import pandas as pd
import glob
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="results/fairness_combined.csv")
    args = parser.parse_args()
    output = args.output
    all_files = glob.glob("results/fairness_*.csv")
    df_list = []
    for filename in all_files:
        df = pd.read_csv(filename)
        df_list.append(df)
    combined_df = pd.concat(df_list, ignore_index=True)
    combined_df.to_csv(output, index=False)
    print(f"Combined {len(all_files)} files into {output}")