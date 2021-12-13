import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import uniform
from fairlearn.metrics import *
from sklearn.metrics import *
from fairmlhealth import report, measure
from functools import partial


def plotOccurence(data,colname,label):
    plot=pd.crosstab(index=data[colname],columns=data[label]).plot(kind='bar',stacked=True,figsize=(16,5))
    plt.xlabel(colname)
    plt.ylabel('Count')
    plt.grid(axis='y',linestyle='-')
    plt.title(colname+" vs "+label+" count")

def plotProportion(data,colname,label):
    plot=pd.crosstab(index=data[colname],columns=data[label],normalize='index').plot(kind='bar',stacked=True,figsize=(16,5))
    plt.xlabel(colname)
    plt.ylabel('Proportion')
    plt.grid(axis='y',linestyle='-')
    plt.title(colname+" vs "+label+" proportion")

def draw_fairlearn_figure(y_test, y_pred, data, output_dir, bias_feature, text='wihtout'):
    metrics = {
        #'count': count,
        # 'auc': partial(roc_auc_score,  average='micro'),
        # 'f1':  partial(f1_score,  average='micro'),
        # 'precision': partial(precision_score,  average='micro') ,
        # 'recall': partial(recall_score,  average='micro'),
        'auc': partial(roc_auc_score,  average='micro'),
        'selection rate': selection_rate,
        'true positive rate': true_positive_rate,
        'false positive rate':false_positive_rate
        }
        #,
        # 'demographic parity difference': demographic_parity_difference_fn,
        # 'demographic parity ratio': demographic_parity_ratio,
        # 'equalized odds difference': equalized_odds_difference,
        # 'equalized odds ratio': equalized_odds_ratio}
    metric_frame = MetricFrame(metrics=metrics,
                            y_true=y_test,
                            y_pred=y_pred,
                            sensitive_features=data)
    
    ax = metric_frame.by_group.plot.bar(
        subplots=True,
        layout=[4, 2],
        legend=False,
        figsize=[12, 8],
        title=f"The evaluation metrics for feature '{bias_feature}' {text} fairness mitigation",
    )
    fig = plt.gcf()
    fig.savefig(f'{output_dir}/{bias_feature}_figure.png')
    

    results_df = pd.DataFrame({'difference': metric_frame.difference(),
                   'difference_to_overall': metric_frame.difference(method='to_overall'),
                   'ratio': metric_frame.ratio(),
                   'ratio_to_overall': metric_frame.ratio(method='to_overall'),
                   'group_min': metric_frame.group_min(),
                  'group_max': metric_frame.group_max(),
                  'overall': metric_frame.overall}).T

    results_df.to_csv(f'{output_dir}/{bias_feature}_results.csv')

def fairmlhealth_metrics(model, X_test, y_test, y_pred, BIAS_COLUMNS):
    for bias_feature in BIAS_COLUMNS:
        report.compare(test_data=X_test, targets=y_test, predictions=y_pred, protected_attr=X_test[bias_feature], models=model)
    measure.performance(X=X_test, y_true=y_test, y_pred=y_pred, features=BIAS_COLUMNS)
    measure.bias(X_test[BIAS_COLUMNS], y_test, y_pred)