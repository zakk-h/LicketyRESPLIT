import run_one_fairness
import argparse


algos = ["lickety", "treefarms"]
depths = [3, 4, 5]
regs = [0.01, 0.05, 0.1]
mults = [0.01, 0.03, 0.05]
data_paths = [
    "data/compas_w_demographics.csv"]
slacks = [0.0]  # no slack for fairness experiments

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--rerun", action='store_true')
    args = parser.parse_args()
    index = args.index
    rerun = args.rerun
    algo = algos[index % len(algos)]
    index //= len(algos)
    depth = depths[index % len(depths)]
    index //= len(depths)
    reg = regs[index % len(regs)]
    index //= len(regs)
    mult = mults[index % len(mults)]
    index //= len(mults)
    multiplicative_slack = slacks[index % len(slacks)]
    index //= len(slacks)
    data_path = data_paths[index % len(data_paths)]
    index //= len(data_paths)
    # check if file exists
    output_file = f"results/fairness_{data_path.split('/')[-1].replace('.csv','')}_{algo}_{depth}_{reg}_{mult}.csv"
    import os
    if os.path.exists(output_file) and not rerun:
        print(f"File {output_file} already exists, skipping...")
        exit(0)    
    # decide other hyperparameters based on index
    run_one_fairness.main(
        data_path=data_path,
        algo=algo,
        reg=reg,
        depth=depth,
        mult=mult,
        multiplicative_slack=multiplicative_slack
    )