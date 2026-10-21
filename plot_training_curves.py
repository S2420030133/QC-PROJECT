import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument('--run-dir', required=True)
args = p.parse_args()
run = Path(args.run_dir)
curves = pd.read_csv(run / 'training_curves.csv')
models = pd.read_csv(run / 'models.csv')
canonical = set(models.loc[models.model_id.str.endswith('_q2_d2_cnot_chain_std0_rev0'), 'model_id'])
curves = curves[curves.model_id.isin(canonical)]
labels = {'wdbc':'WDBC', 'wine_0_vs_rest':'Wine', 'synthetic_balanced':'Synthetic balanced', 'synthetic_imbalanced':'Synthetic imbalanced'}
colors = {'wdbc':'#1565C0', 'wine_0_vs_rest':'#D84315', 'synthetic_balanced':'#2E7D32', 'synthetic_imbalanced':'#6A1B9A'}
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)
for task, data in curves.groupby('task'):
    # Average multiple initializations within each fold first; plot fold-level
    # variability rather than counting those initializations as new folds.
    by_fold = data.groupby(['split','epoch']).agg(
        train_mean=('training_binary_cross_entropy','mean'), train_sd=('training_binary_cross_entropy','std'),
        dev_mean=('development_binary_cross_entropy','mean'), dev_sd=('development_binary_cross_entropy','std'),
        ba_mean=('development_balanced_accuracy_at_zero','mean')).reset_index()
    group = by_fold.groupby('epoch').agg(
        train_mean=('train_mean','mean'), dev_mean=('dev_mean','mean'), dev_sd=('dev_mean','std'),
        ba_mean=('ba_mean','mean'), ba_sd=('ba_mean','std')).reset_index()
    color=colors[task]
    axes[0].plot(group.epoch, group.train_mean, color=color, linestyle='-', label=f'{labels[task]} train')
    axes[0].plot(group.epoch, group.dev_mean, color=color, linestyle='--', label=f'{labels[task]} development')
    axes[1].plot(group.epoch, group.ba_mean, color=color, label=labels[task])
    axes[0].fill_between(group.epoch, group.dev_mean-group.dev_sd, group.dev_mean+group.dev_sd, color=color, alpha=.10)
    axes[1].fill_between(group.epoch, group.ba_mean-group.ba_sd, group.ba_mean+group.ba_sd, color=color, alpha=.10)
axes[0].set(xlabel='Epoch', ylabel='Binary cross-entropy', title='Training and development loss')
axes[0].legend(fontsize=7, ncol=2)
axes[1].set(xlabel='Epoch', ylabel='Balanced accuracy at score 0', title='Development performance', ylim=(0,1.02))
axes[1].legend(fontsize=8)
fig.savefig(run/'training_convergence.png', dpi=180)
