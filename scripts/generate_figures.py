#!/usr/bin/env python3
"""Generate all project figures exclusively from frozen result tables."""
from pathlib import Path
import hashlib
import numpy as np,pandas as pd
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

ROOT=Path(__file__).resolve().parents[1]
RES=ROOT/'results/final_experiments'
OUT=ROOT/'results/final_experiments/figures'
DATA=OUT/'source_data';OUT.mkdir(parents=True,exist_ok=True);DATA.mkdir(exist_ok=True)
ORDER=['A','B','C','A+B','A+C','B+C','A+B+C']
LABEL={'A':'Chemistry','B':'Site','C':'Context','A+B':'Chemistry + Site','A+C':'Chemistry + Context','B+C':'Site + Context','A+B+C':'Chemistry + Site + Context'}
SHORT={'A':'Chemistry','B':'Site','C':'Context','A+B':'Chem. + Site','A+C':'Chem. + Context','B+C':'Site + Context','A+B+C':'Chem. + Site + Context'}
COL={'A':'#0072B2','B':'#E69F00','C':'#56B4E9','A+B':'#009E73','A+C':'#CC79A7','B+C':'#D55E00','A+B+C':'#332288'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':11,'axes.labelsize':10,'savefig.dpi':300,'figure.dpi':150,'pdf.fonttype':42})
def save(fig,name):
 fig.tight_layout();fig.savefig(OUT/f'{name}.png',bbox_inches='tight');fig.savefig(OUT/f'{name}.pdf',bbox_inches='tight');plt.close(fig);print(name)
def err(ax,x,q,color=None):ax.errorbar(x,q['mean'],yerr=[[q['mean']-q['ci95_low']],[q['ci95_high']-q['mean']]],fmt='o',color=color or 'black',capsize=3,lw=1.1)

def ablation():
 d=pd.read_csv(RES/'primary_ablation/summary_with_ci.csv');q=d[d.metric=='r2'].copy();q['ord']=q.feature_set_id.map({x:i for i,x in enumerate(ORDER)});q=q.sort_values(['split','ord']);q.to_csv(DATA/'primary_ablation_r2.csv',index=False)
 fig,axs=plt.subplots(1,2,figsize=(11,4.3),sharey=True)
 for ax,(sp,title) in zip(axs,[('random','Random split'),('sequence_cluster','70% sequence-cluster split')]):
  z=q[q.split==sp].sort_values('ord');x=np.arange(7);ax.bar(x,z['mean'],color=[COL[a] for a in z.feature_set_id],alpha=.88);ax.errorbar(x,z['mean'],yerr=[z['mean']-z.ci95_low,z.ci95_high-z['mean']],fmt='none',ecolor='black',capsize=3,lw=.8);ax.set_xticks(x);ax.set_xticklabels([SHORT[a] for a in z.feature_set_id],rotation=38,ha='right');ax.set_title(title);ax.grid(axis='y',ls='--',alpha=.25);ax.axhline(0,color='black',lw=.6)
 axs[0].set_ylabel('Test R², mean and 95% CI (five seeds)');save(fig,'figure_primary_seven_way_ablation')

def split_slope():
 d=pd.read_csv(RES/'primary_ablation/summary_with_ci.csv');q=d[d.metric=='r2'];q.to_csv(DATA/'random_cluster_comparison.csv',index=False)
 fig,ax=plt.subplots(figsize=(7,5))
 for fs in ORDER:
  z=q[q.feature_set_id==fs].set_index('split');ys=[z.loc['random','mean'],z.loc['sequence_cluster','mean']];lo=[z.loc['random','ci95_low'],z.loc['sequence_cluster','ci95_low']];hi=[z.loc['random','ci95_high'],z.loc['sequence_cluster','ci95_high']];ax.plot([0,1],ys,'o-',color=COL[fs],label=LABEL[fs],lw=1.7);ax.errorbar([0,1],ys,yerr=[np.array(ys)-lo,hi-np.array(ys)],fmt='none',ecolor=COL[fs],capsize=2,lw=.8)
 ax.set_xticks([0,1]);ax.set_xticklabels(['Random split','Sequence-cluster split']);ax.set_ylabel('Test R², mean and 95% CI');ax.grid(axis='y',ls='--',alpha=.25);ax.legend(fontsize=7,ncol=2);save(fig,'figure_random_vs_cluster')

def estimators():
 d=pd.read_csv(RES/'estimator_matrix/summary_with_ci.csv');q=d[d.metric=='r2'].copy();ests=['Ridge','ElasticNet','RBF-SVR','Random Forest','XGBoost'];mat=q.pivot(index='estimator',columns='feature_set_id',values='mean').reindex(index=ests,columns=ORDER);mat.to_csv(DATA/'estimator_descriptor_heatmap.csv')
 fig,ax=plt.subplots(figsize=(10,4.5));sns.heatmap(mat,annot=True,fmt='.3f',cmap='viridis',vmin=0,vmax=.5,linewidths=.4,cbar_kws={'label':'Mean test R²'},ax=ax);ax.set(xlabel='Descriptor representation',ylabel='Estimator');ax.set_xticklabels([SHORT[x] for x in ORDER],rotation=35,ha='right');save(fig,'figure_estimator_descriptor_heatmap')
 z=q[q.feature_set_id=='A+B+C'].set_index('estimator').reindex(ests);z.to_csv(DATA/'complete_representation_estimators.csv');fig,ax=plt.subplots(figsize=(7,4));x=np.arange(len(z));ax.bar(x,z['mean'],color=['#999999','#BBBBBB','#56B4E9','#E69F00','#0072B2']);ax.errorbar(x,z['mean'],yerr=[z['mean']-z.ci95_low,z.ci95_high-z['mean']],fmt='none',ecolor='black',capsize=3);ax.set_xticks(x);ax.set_xticklabels(ests,rotation=25,ha='right');ax.set_ylabel('Test R², mean and 95% CI');ax.set_title('Chemistry + Site + Context across estimators');ax.grid(axis='y',ls='--',alpha=.25);save(fig,'figure_complete_representation_estimators')

def shap_figs():
 g=pd.read_csv(RES/'shap/group_summary_with_ci.csv');b=pd.read_csv(RES/'shap/subgroup_summary_with_ci.csv');g=g[g.representation_id=='A+B+C'];b=b[b.representation_id=='A+B+C'];g.to_csv(DATA/'shap_conceptual_groups.csv',index=False);b.to_csv(DATA/'shap_site_subgroups.csv',index=False)
 fig,axs=plt.subplots(1,2,figsize=(9,4));
 for ax,z,order,title in [(axs[0],g,['A','B','C'],'Conceptual groups'),(axs[1],pd.concat([g[g.group.isin(['A','C'])],b]),['A','B1','B2','B3','C'],'Descriptor subblocks')]:
  z=z.set_index('group').loc[order];x=np.arange(len(order));ax.bar(x,z['mean'],color=['#0072B2','#E69F00','#56B4E9'] if len(order)==3 else ['#0072B2','#E69F00','#009E73','#CC79A7','#56B4E9']);ax.errorbar(x,z['mean'],yerr=[z['mean']-z.ci95_low,z.ci95_high-z['mean']],fmt='none',ecolor='black',capsize=3);ax.set_xticks(x);ax.set_xticklabels(order);ax.set_title(title);ax.grid(axis='y',ls='--',alpha=.25)
 axs[0].set_ylabel('Absolute SHAP attribution proportion\nmean and 95% CI across five seeds');save(fig,'figure_shap_group_and_subblock')
 t=pd.read_csv(RES/'shap/top_features.csv');t=t[(t.representation_id=='A+B+C')&(t.scope=='overall')].sort_values('rank').head(15);t.to_csv(DATA/'shap_top15.csv',index=False);fig,ax=plt.subplots(figsize=(7.5,5.8));z=t.iloc[::-1];colors=[{'A':'#0072B2','B1':'#E69F00','B2':'#009E73','B3':'#CC79A7','C':'#56B4E9'}[x] for x in z.subblock];ax.barh(z.feature,z.proportion,color=colors,alpha=.9);ax.errorbar(z.proportion,z.feature,xerr=[z.proportion-z.ci95_low,z.ci95_high-z.proportion],fmt='none',ecolor='#222222',elinewidth=1,capsize=2.5,capthick=1,zorder=3);ax.set_xlabel('Mean absolute SHAP proportion (95% CI across five seeds)');ax.set_title('Top individual descriptors: complete representation');ax.grid(axis='x',ls='--',alpha=.25);ax.set_axisbelow(True);save(fig,'figure_shap_top15_features')

def time_forward():
 p=pd.read_csv(RES/'time_forward/model_performance_with_ci.csv');p=p[p.metric=='spearman'];s=pd.read_csv(RES/'time_forward/similarity_diagnostics.csv');c=pd.read_csv(RES/'time_forward/cutoff_statistics.csv');p.to_csv(DATA/'time_forward_spearman.csv',index=False);s.to_csv(DATA/'time_forward_similarity.csv',index=False);c.to_csv(DATA/'time_forward_cutoff_stats.csv',index=False)
 models=['Sequence','Chemistry','Chemistry + Site','Chemistry + Site + Context','RDKit','ECFP'];colors=['#999999','#0072B2','#009E73','#332288','#56B4E9','#D55E00'];fig,axs=plt.subplots(2,1,figsize=(9,7),sharex=True,gridspec_kw={'height_ratios':[2,1]})
 for m,col in zip(models,colors):
  z=p[p.model==m].sort_values('cutoff');axs[0].plot(z.cutoff,z.estimate,'o-',label=m,color=col);axs[0].fill_between(z.cutoff,z.ci95_low,z.ci95_high,color=col,alpha=.10)
 axs[0].axhline(0,color='black',lw=.6);axs[0].set_ylabel('Spearman correlation\nand grouped-bootstrap 95% CI');axs[0].legend(fontsize=7,ncol=3);axs[0].grid(ls='--',alpha=.22)
 axs[1].plot(s.cutoff,s.median_max_tanimoto,'o-',color='#D55E00',label='Median maximum Tanimoto');ax2=axs[1].twinx();ax2.bar(c.cutoff,c.test_n,color='#999999',alpha=.25,label='Test n');axs[1].set_ylabel('Median max Tanimoto');ax2.set_ylabel('Test-set size');axs[1].set_xlabel('Cutoff year');axs[1].grid(ls='--',alpha=.22);save(fig,'figure_time_forward_shift_diagnostics')

def scaffold():
 d=pd.read_csv(RES/'scaffold_ranking/paired_comparison_AB_vs_A_family_values.csv');d.to_csv(DATA/'scaffold_family_differences.csv',index=False);z=d.dropna(subset=['difference']).sort_values('difference');fig,ax=plt.subplots(figsize=(9,5));x=np.arange(len(z));cols=np.where(z.difference>0,'#009E73',np.where(z.difference<0,'#D55E00','#999999'));ax.bar(x,z.difference,color=cols);ax.axhline(0,color='black',lw=.8);ax.set(xlabel='Peptide family (ordered by difference)',ylabel='Pairwise-accuracy difference\nChemistry + Site − Chemistry');ax.set_title('Scaffold-focused ranking across peptide families');ax.grid(axis='y',ls='--',alpha=.25);save(fig,'figure_scaffold_family_differences')
 s=pd.read_csv(RES/'scaffold_ranking/summary_with_ci.csv');s.to_csv(DATA/'scaffold_summary.csv',index=False);order=['Sequence','Chemistry','Chemistry + Site','Chemistry + Site + Context','ECFP'];z=s.set_index('model').loc[order];fig,ax=plt.subplots(figsize=(7.5,4));x=np.arange(len(z));ax.bar(x,z.mean_family_pairwise_accuracy,color=['#999999','#0072B2','#009E73','#332288','#D55E00']);ax.errorbar(x,z.mean_family_pairwise_accuracy,yerr=[z.mean_family_pairwise_accuracy-z.family_bootstrap_ci95_low,z.family_bootstrap_ci95_high-z.mean_family_pairwise_accuracy],fmt='none',ecolor='black',capsize=3);ax.set_xticks(x);ax.set_xticklabels(order,rotation=25,ha='right');ax.set_ylabel('Mean family pairwise accuracy\nand family-bootstrap 95% CI');ax.grid(axis='y',ls='--',alpha=.25);save(fig,'figure_scaffold_ranking_summary')

def predicted_observed():
 rows=[]
 for sp in ['random','sequence_cluster']:
  for fs in ['A','A+B','A+B+C']:
   paths=pd.read_csv(RES/'primary_ablation/seed_level_results.csv').query('split==@sp and feature_set_id==@fs').prediction_file
   ds=[]
   for path in paths:ds.append(pd.read_csv(ROOT/path)[['sample_id','y_true','y_pred']])
   base=ds[0][['sample_id','y_true']].copy();base['y_pred']=np.mean([x.y_pred.to_numpy() for x in ds],axis=0);base['split']=sp;base['feature_set_id']=fs;rows.append(base)
 d=pd.concat(rows);d.to_csv(DATA/'predicted_vs_observed.csv',index=False);fig,axs=plt.subplots(2,3,figsize=(11,7),sharex=True,sharey=True)
 for i,sp in enumerate(['random','sequence_cluster']):
  for j,fs in enumerate(['A','A+B','A+B+C']):
   ax=axs[i,j];z=d[(d.split==sp)&(d.feature_set_id==fs)];ax.hexbin(z.y_true,z.y_pred,gridsize=35,mincnt=1,cmap='viridis');lims=[min(z.y_true.min(),z.y_pred.min()),max(z.y_true.max(),z.y_pred.max())];ax.plot(lims,lims,'--',color='white',lw=.8);ax.set_title(f"{LABEL[fs]}\n{'Random' if sp=='random' else 'Cluster'} split");ax.grid(alpha=.12)
 for ax in axs[-1]:ax.set_xlabel('Observed permeability')
 for ax in axs[:,0]:ax.set_ylabel('Mean prediction across five seeds')
 save(fig,'figure_predicted_vs_observed')

def main():
 ablation();split_slope();estimators();shap_figs();time_forward();scaffold();predicted_observed()
 rows=[]
 for path in sorted([*OUT.glob('*.png'),*OUT.glob('*.pdf')]):
  rows.append({'figure':path.stem,'format':path.suffix[1:],'path':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'source_data_dir':str(DATA.relative_to(ROOT))})
 pd.DataFrame(rows).to_csv(OUT/'figure_manifest.csv',index=False)
 print(f'Figures written to {OUT}')
if __name__=='__main__':main()
