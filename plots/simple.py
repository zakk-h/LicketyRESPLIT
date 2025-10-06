import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COMPARISONS = {
    "Equalized Odds (best)": "equalized_odds_difference_best",
    "Demographic Parity (best)": "demographic_parity_difference_best",
    "Equalized Odds (worst)": "equalized_odds_difference_worst",
    "Demographic Parity (worst)": "demographic_parity_difference_worst",
}

for comparison_name, comparison_val in COMPARISONS.items():

    data = pd.read_csv('results/fairness_combined.csv')

    # do a bar chart showing, on the y-axis, comparison_val
    # the x axis will show a bunch of different values; but with two results clumped together for each config: 
    # the result for lickety and the result for treefarms for that given config
    configs = data[['reg', 'depth', 'mult']].drop_duplicates().reset_index(drop=True)
    configs['config'] = configs.apply(lambda row: f"reg={row['reg']},depth={int(float(row['depth']))},mult={row['mult']}", axis=1)
    configs = configs.sort_values(by=['reg', 'depth', 'mult']).reset_index(drop=True)
    config_labels = configs['config'].tolist()
    config_indices = {config: i for i, config in enumerate(config_labels)}
    x = np.arange(len(config_labels))
    y_lickety = []
    y_treefarms = []

    for config in config_labels:
        # Filter for lickety and treefarms for this config
        mask = (
            (data['reg'] == float(config.split(',')[0].split('=')[1])) &
            (data['depth'] == int((config.split(',')[1].split('=')[1]))) &
            (data['mult'] == float(config.split(',')[2].split('=')[1]))
        )
        lickety_row = data[mask & (data['algo'] == 'lickety')]
        treefarms_row = data[mask & (data['algo'] == 'treefarms')]
        if not lickety_row.empty:
            y_lickety.append(lickety_row[comparison_val].values[0])
        else:
            y_lickety.append(np.nan)
        if not treefarms_row.empty:
            y_treefarms.append(treefarms_row[comparison_val].values[0])
        else:
            y_treefarms.append(np.nan)
    y_lickety = np.array(y_lickety)
    y_treefarms = np.array(y_treefarms)

    width = 0.35  # the width of the bars
    fig, ax = plt.subplots(figsize=(18, 6))
    bars1 = ax.bar(x - width/2, y_lickety, width, label='Lickety', color='skyblue')
    bars2 = ax.bar(x + width/2, y_treefarms, width, label='TreeFarms', color='salmon')
    ax.set_ylabel(comparison_name)
    ax.set_title(f'{comparison_name} by Algorithm and Configuration')
    ax.set_xticks(x)
    ax.set_xticklabels(config_labels, rotation=45, ha='right')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'plots/fairness_comparison_{comparison_name}.pdf')
    plt.show()
    plt.close()