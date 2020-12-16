import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
from xgboost import XGBClassifier
from time import time

idx = pd.IndexSlice

# COMMAND ----------

# MAGIC %md General

# COMMAND ----------

def count_unique_index(df, index_level=0):
    return df.index.get_level_values(index_level).nunique()

def describe_datetimeindex(df, index_level=1):
    return pd.Series(df.index.get_level_values(index_level)).describe(datetime_is_numeric=True)
  
def prop_table(x, dropna=False):
    tmp = (x.value_counts(sort=False, dropna=dropna).reset_index()
           .merge((100 * x.value_counts(sort=False, normalize=True, dropna=dropna)).round(2).reset_index(), on='index',
                  how='inner'))
    tmp.columns = [x.name, 'count', 'percent']
    tmp = tmp.sort_values('count', ascending=False)
    tot = x.notnull().sum() if dropna else len(x)
    return tmp.append(pd.DataFrame([['Total', tot, 100]], columns=tmp.columns), ignore_index=True)

  

# COMMAND ----------

# MAGIC %md Data generation

# COMMAND ----------

def generate_normalized_hr_sample(random_state=1729, split='train', ili_type=3):
    rnd = np.random.RandomState(random_state)
    
    participant_id = 'P'+''.join([str(rnd.choice(np.arange(0,10), 1)[0]) for i in range(12)])
    onset_date = f'2020-{rnd.choice(np.arange(2,7), 1)[0]:01d}-{rnd.choice(np.arange(1,28), 1)[0]:01d}'
    
    healthy_miss_fraction = 0.1
    illness_miss_fraction = 0.2
    healthy_dist_param = [0, 0.3]
    illness_dist_param = [0, 0.4]
    
    dict_ili = {1:{'hr_max': -0.1, ## ILI
                   'rhr':0.05,
                   'hr_stdv': -0.3,
                   'hr_50pct': -0.2,
                   'miss_fraction': {0:0.1, 1:0.15, 2:0.1},
                   'pivot': 2
                   },
                2:{'hr_max': -0.2, ## FLU
                   'rhr': 0.1,
                   'hr_stdv': -0.4,
                   'hr_50pct': -0.3,
                   'miss_fraction': {0: 0.1, 1:0.2, 2:0.1},
                   'pivot': 3
                   },
                3:{'hr_max': -0.4, ## COVID
                   'rhr': 0.3,
                   'hr_stdv': -0.6,
                   'hr_50pct': -0.5,
                   'miss_fraction': {0: 0.1, 1: 0.25, 2:0.2},
                   'pivot': 4
                   }
               }
    
    shifts = 5
    dt = pd.date_range(pd.to_datetime(onset_date) - pd.Timedelta('28d'),
              pd.to_datetime(onset_date) + pd.Timedelta('14d'),
             )
    
    col_names = {'hr_max': 'heart_rate__not_moving__max',
                 'rhr': 'heart_rate__resting_heart_rate',
                 'hr_stdv': 'heart_rate__stddev',
                 'hr_50pct': 'heart_rate__perc_50th'
                 }
    n_cols = len(col_names)
    
    def _linear_trend(peak, trough, width, days):
        step = (peak-trough)/width
        return np.pad(np.concatenate([np.arange(trough, peak, step)+step,
                                      np.arange(peak,trough, -step)-step]), 
                                      (0,days-2*width), constant_values=trough)
    
    def _sample_hr(rnd, days, label, ili_type):
        if label==1:
            return np.column_stack([rnd.normal(illness_dist_param[0], illness_dist_param[1], [days, len(col_names)]) +\
                                    np.column_stack([_linear_trend(dict_ili[ili_type][colz], illness_dist_param[0], dict_ili[ili_type]['pivot'], days)
                                                     for colz in col_names.keys()]),
                                   0+(rnd.uniform(0,1,days) < dict_ili[ili_type]['miss_fraction'][label])
                                    ])
        
        else:
            return np.column_stack([rnd.normal(healthy_dist_param[0], healthy_dist_param[1], [days, len(col_names)]),
                                    0+(rnd.uniform(0,1,days) < dict_ili[ili_type]['miss_fraction'][label])
                                    ])
      
    def _add_shifts(x, rows, cols):
        if rows==0:
            return x
        y = np.empty([rows,cols])
        y[:] = np.nan
        return np.row_stack([y, x[:-rows,:]])
    
    dat = np.row_stack([_sample_hr(rnd, 27, 0, ili_type),
                      _sample_hr(rnd, 9, 1, ili_type),
                      _sample_hr(rnd, 7, 0, ili_type)
                       ])
    dat[dat[:,-1]==1, :-1] = np.nan ### add missing values
    
    out = pd.DataFrame(np.column_stack([_add_shifts(dat[:,:-1], 1, n_cols) for i in range(shifts)]),
                       columns=pd.MultiIndex.from_product([[str(i)+'days_ago' for i in range(shifts)], col_names.values()]),
                       index=pd.MultiIndex.from_product([[participant_id], dt], names=['id_participant_external', 'dt'])
        )
        
    day_col = ('labels', 'days_since_onset')
    label_col= ('labels', 'training_labels')
    out[('labels', 'split')] = split
    out[('labels', 'ILI_type')] = ili_type
    out[day_col] = np.arange(-28,15)
    out[label_col] = -1
    out.loc[(out[day_col] > -22) & (out[day_col] < -7), label_col] = 0
    out.loc[(out[day_col] > 0) & (out[day_col] < 8), label_col] = 1
    
    return out

# COMMAND ----------

# MAGIC %md Prepare data

# COMMAND ----------

def get_dataset(df, keep_filter, days_ago, feature_cols,
               label_col = ('labels', 'training_labels')):

    y = df.loc[keep_filter ,label_col]
    X = df.loc[keep_filter, idx[days_ago, feature_cols]]
    filter_rows = (~X.isna().all(axis=1))

    print(X.shape, y.shape)
    print(f'Missing rows percent = {100 - 100*filter_rows.sum()/X.shape[0]:.2f}%')
    print(prop_table(y))
    
    return X, y, filter_rows

# COMMAND ----------

# MAGIC %md Model training

# COMMAND ----------

def run_xgb_class2(classifier, X_train, y_train, X_val, y_val, scorer=roc_auc_score):
    classifier.fit(X_train, y_train)
    
    yh_train = classifier.predict_proba(X_train)[:,1]
    yh_val = classifier.predict_proba(X_val)[:,1]
    
    print(f'Train ROC: {scorer(y_train, yh_train):.4f}')
    print(f'Val ROC: {scorer(0+(y_val > 0), yh_val):.4f}')
    
    return classifier, yh_train, yh_val
  
def run_xgb_hyperopt_2class(space, X_train, y_train, X_val, y_val, scorer=roc_auc_score):
    hypopt = []
    for params in space:
        classifier = XGBClassifier(**params)

        stime = time()
        classifier.fit(X_train, y_train)
        etime = time()

        yh_train = classifier.predict_proba(X_train)[:,1]
        yh_val = classifier.predict_proba(X_val)[:,1]

        hypopt.append(pd.Series([scorer(y_train, yh_train),
                                 scorer(y_val, yh_val),
                                (etime-stime)/60] + list(params.values()),
                                index=['train_roc', 'val_roc', 'time_mins'] + list(params.keys())
                               ))

    hypopt = pd.concat(hypopt, axis=1).T
    return hypopt.sort_values(by=['val_roc', 'train_roc'], ascending=[False, True])

# COMMAND ----------

# MAGIC %md Model predictions

# COMMAND ----------

def get_specificity_threshold(y, yh, list_specificity_fraction):
    ROC = roc_curve(y, yh)
    dict_thresh = {}
    for spec in list_specificity_fraction:
        thresh = ROC[2][np.where(ROC[0] <= 1-spec)[0]-1][-1]
        print(f'{100*spec:.0f}% Specifivity cutoff = {thresh:.4f}')
        print(classification_report(y, 0+(yh >= thresh)))
        print('-' * 50)
        dict_thresh[spec] = np.round(thresh, 4)
    return ROC, dict_thresh
  
def run_get_predictions(classifier, X, y, filter_row, df_labels, use_spec, use_spec_thresh, col_names=['I', 'C'], group_cols=['participant_id', 'event_order'], day_col='days_since_onset_v43', day_detect=-3, type_col='ILI_type', hue_col='Type', ili_type_map = {1:'any ILI', 2:'Flu', 3:'COVID'}):
    
    yh = classifier.predict_proba(X)
    
    tmp = df_labels.loc[y.index,:]
    tmp.columns = tmp.columns.droplevel(0)

    pred = (pd.DataFrame(yh, index=y.index, columns=col_names)
            .join(tmp)
           )
    
    #pred = get_predictions_v4(df_labels, y, yh, col_names)
    print('N =', count_unique_index(pred))
    
    thresh_tag = f'_spec{100*use_spec:.0f}_{"v".join(col_names)}'
    spec_thresh_col = 'pred'+thresh_tag
    filter_thresh_col = 'filter'+thresh_tag
    cumsum_thresh_col = 'cumsum'+thresh_tag
    count_thresh_col = 'count'+thresh_tag
    
    pred[spec_thresh_col] = 0+(pred[col_names[-1]] >= use_spec_thresh)
    pred.loc[~filter_row, spec_thresh_col] = np.nan 
      
    pred[filter_thresh_col] = pred[spec_thresh_col].copy()
    pred[count_thresh_col] = pred[spec_thresh_col].copy()
    pred.loc[pred[spec_thresh_col].notna(), count_thresh_col] = 1
    
    ### Set detection before Day -2 as 0
    pred.loc[pred[day_col] < day_detect, filter_thresh_col] = 0 
    pred.loc[pred[day_col] < day_detect, count_thresh_col] = 0 
    
    ### Cumsum predictions
    pred = (pred
           .join(pred
                 .groupby(group_cols, as_index=False)
                 .apply(lambda x: x[filter_thresh_col].cumsum().ffill()).rename(cumsum_thresh_col).droplevel(0).to_frame()
                )
          )

    ### Set all days after first detection as 1
    pred.loc[pred[cumsum_thresh_col] > 1, cumsum_thresh_col]  = 1
    pred[count_thresh_col] = pred[count_thresh_col].ffill()
    
    plot_df = (pred
               .groupby([day_col, type_col])
               .agg({cumsum_thresh_col: 'sum', count_thresh_col: 'sum'})
               .reset_index()
               )

    def run_expanding_max(x, colz=day_col):
        return (x
                .set_index(colz)
                .expanding()
                .max()
                .reset_index()
                )

      
    plot_df = (plot_df
               .groupby(type_col, as_index=False)
               .apply(run_expanding_max)
               .reset_index(drop=True)
              )
    
    plot_df['recall_fraction'] = plot_df[cumsum_thresh_col]/plot_df[count_thresh_col]
    print('Cumulative recall shape=', plot_df.shape)
    
    map_type = (plot_df
                 .groupby(type_col)
                 .apply(lambda x: ili_type_map[x[type_col].unique()[0]] + ', N='+ str(int(x[count_thresh_col].max())))
                 .to_dict()
                )

    pred[hue_col] = pred[type_col].map(map_type)
    plot_df[hue_col] = plot_df[type_col].map(map_type)
    print('Predictions shape=', pred.shape)

      
    return pred, plot_df
  
def get_feature_importance(classifier):
    return (pd.Series(classifier
                      .get_booster()
                      .get_score(importance_type='gain')
                     )
            .sort_values(ascending=False)
            .to_frame()
            .reset_index()
            .rename(columns={'index': 'feature_name', 0:'gain'})
           )

# COMMAND ----------

# MAGIC %md Plotting

# COMMAND ----------

sns_hue = sns.color_palette()
dict_hue = {'ILI': sns_hue[0], 
            'Covid': sns_hue[1],
            'Healthy': sns_hue[2], 
            'Flu': sns_hue[3]}

ili_type_map = {0:'Healthy', 1: 'ILI', 2: 'Flu', 3: 'COVID-19'}

def plot_trend_lines(df, plot_cols, use_palette, use_hue_order, type_col=('labels', 'Type'),
                     ts_col = ('labels', "days_since_onset"), ci=67, ts_cut=30, ts_step=4, line_color='coral',
                     per_row=2, thick=2, plot_width=5, plot_height=4, sharex=False, sharey=True, grid=False):
  
    plotz = len(plot_cols)
    rowz = plotz // per_row + 0+(plotz % per_row > 0)
    
    fig, axes = plt.subplots(nrows=rowz, ncols=per_row, figsize=(plot_width*per_row, plot_height*rowz), sharey=sharey, sharex=sharex)
    keep_rows = (df[ts_col] >= -ts_cut) & (df[ts_col] <= ts_cut)
    (ts_min, ts_max) = df[keep_rows].agg({ts_col: ['min', 'max']}).unstack().values
    print(ts_min, ts_max)
    
    for ft_col, ax in zip(plot_cols, axes.flatten() if type(axes) == np.ndarray else [axes]):
        
        if type(ft_col) == tuple:
          colr = ft_col[-1]
        else:
          colr = ft_col
      
        if type(ts_col) == tuple:
          xlabel = ts_col[-1]
        else:
          xlabel = ts_col

        sns.lineplot(x=ts_col, y=ft_col, hue=type_col, data=df.loc[keep_rows,:],
                     palette = use_palette,
                     hue_order = use_hue_order,
                     ax=ax, ci=ci, color=line_color, linewidth=thick)

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles[1:], labels=labels[1:], fontsize=14)
        ax.axvline(x=0, c='k', ls='--')
        ax.axhline(y=0, c='k', ls='--')
        ax.set_xticks(np.arange(ts_min, ts_max+1, ts_step))
        ax.set_title(colr.replace('__',' : ').replace('_',' ').capitalize(), fontsize=16)
        ax.set_xlabel(xlabel.replace('_',' ').capitalize(), fontsize=14)
        ax.set_ylabel('')
        
        if grid:
            ax.grid()

    plt.tight_layout()
    plt.close() 
    return fig

def single_plot_missing_performance(y_val, yh_val, X_val, title='Healthy v. ILI', min_N=10):
    out_val = pd.DataFrame(np.column_stack([y_val.values, yh_val, X_val.isna().sum(axis=1)/X_val.shape[1]]), index=y_val.index, columns=['gt', 'prob', 'frac'])
    fig, ax = plt.subplots(1, 1, figsize=(7,5))
       
    ## Add cumulative missing-fraction's AUROC score
    plot_roc = pd.DataFrame([{'frac': q, 
                              'Data Fraction': (out_val['frac'] <= q).sum()/out_val.shape[0],
                              'Class-0 mean': out_val.loc[(out_val['frac'] <= q) & (out_val['gt']==0), 'prob'].mean(),
                              'Class-1 mean': out_val.loc[(out_val['frac'] <= q) & (out_val['gt']==1), 'prob'].mean(),
                              'AUROC': roc_auc_score(out_val.loc[out_val.frac <= q, 'gt'], out_val.loc[out_val.frac <= q, 'prob'])} 
                             for q in np.arange(0, 1.05, 0.05) if (out_val.frac <= q).sum() >= min_N])
    
    plot_roc.plot(x='frac', y=['Class-0 mean', 'Class-1 mean', 'AUROC', 'Data Fraction'], ax=ax, 
                  color=['C0', 'orange', 'k', 'coral'], style=['-', '-', '--', '-.'], linewidth=2)
    
    #sns.lineplot(x='frac', y='prob', hue='True label', data=out_val.rename(columns={'gt': 'True label'}), ax=ax[1], lw=2)
    #plot_roc.plot(x='frac', y='AUROC', ax=ax[1], color='k', style='--', linewidth=2)
    
    ax.set_ylabel('Score', fontsize=14)
    ax.set_xlabel('Missing data Less Than fraction', fontsize=14)
    ax.set_title(title, fontsize=15)
    ax.legend(fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.close()  
    return fig
   
def plot_roc(fpr, tpr):
    fig = plt.figure(figsize=(5,4))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.grid()
    plt.close() 
    return fig

def plot_spec_recall_since_onset(pred, plot_df, use_spec, spec_thresh_col, use_palette, use_hue_order, set_tag, max_missing_frac, cumsum_col, count_col, y_tag='COVID-19', dataset_tag='LSFS', hue_col='Type', x_col='days_since_onset_v43', ci=67, xticks_range=np.arange(-28,15,4)):

    fig, ax = plt.subplots(1,1,figsize=(9,6))
   
    threshold_line = 1-use_spec

    sns.lineplot(x = x_col,
                 y = spec_thresh_col,
                 hue = hue_col,
                 palette = use_palette,
                 hue_order = use_hue_order,
                 data = pred,
                 ci = ci
                 )
    
    if cumsum_col is not None:
        for i,q in enumerate(use_hue_order):
            labz = 'Cumulative '+q.split(',')[0] 
            (plot_df[plot_df[hue_col]==q]
             .rename(columns={'recall_fraction':labz})
             .plot(x=x_col, y=labz, c=use_palette[i], lw=2,
                   style='--', ax=ax, legend=False)
            )
    
    ax.axvline(x=0, c='k', ls=':', alpha=0.5)
    ax.axhline(y=threshold_line, c='k', ls=':', alpha=0.7)
    ax.set_xticks(xticks_range)
    plt.xticks(fontsize=12);
    ax.legend(fontsize=14) ##loc='upper left', 
    plt.yticks(fontsize=12);
    ax.set_ylabel(f'Fraction positive predictions\n for {y_tag}', fontsize=16)
    ax.set_xlabel('Days since onset', fontsize=16)
    #ax.set_title(f'{dataset_tag}: {set_tag}-set predictions\n {100*use_spec:.0f}% specificity, max-{100*max_missing_frac:.0f}% missing', fontsize=18)
    ax.set_title(f'{dataset_tag}: {set_tag} \n {100*use_spec:.0f}% specificity threshold, max-{100*max_missing_frac:.0f}% missing data', fontsize=18)
    plt.close() 
    return fig, ax
