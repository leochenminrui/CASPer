#!/usr/bin/env python3
from pathlib import Path
import pandas as pd,numpy as np
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/minor_revision_experiments/time_forward'
c=pd.read_csv(OUT/'cutoff_statistics.csv');s=pd.read_csv(OUT/'similarity_diagnostics.csv');p=pd.read_csv(OUT/'model_performance_with_ci.csv');sp=p[p.metric=='spearman'].merge(c,on='cutoff').merge(s,on='cutoff');rows=[]
for m,g in sp.groupby('model'):
 for x in ['median_max_tanimoto','test_n','unseen_test_edit_fraction','unseen_test_monomer_fraction']:
  z=spearmanr(g.estimate,g[x]);rows.append({'model':m,'performance_metric':'Spearman','diagnostic':x,'cutoff_level_spearman':z.statistic,'p_value_descriptive':z.pvalue,'n_cutoffs':len(g)})
pd.DataFrame(rows).to_csv(OUT/'cutoff_associations.csv',index=False)
tab=sp[['cutoff','train_n','test_n','train_sequences','test_sequences','test_families','median_max_tanimoto','iqr_max_tanimoto','model','estimate','ci95_low','ci95_high']].rename(columns={'estimate':'spearman'});tab.to_csv(OUT/'supplementary_cutoff_model_table.csv',index=False)
focus=c[c.cutoff.isin([2020,2021,2023])].merge(s,on='cutoff');focus.to_csv(OUT/'focal_year_diagnostics.csv',index=False)
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(8,5))
for m,g in sp.groupby('model'):ax.plot(g.cutoff,g.estimate,marker='o',label=m)
ax.axhline(0,color='black',lw=.7);ax.set(xlabel='Cutoff year',ylabel='Spearman correlation');ax.legend(fontsize=7,ncol=2);fig.tight_layout();fig.savefig(OUT/'plots/time_forward_spearman.png',dpi=300);plt.close(fig)
print('time-forward statistics finalized')
