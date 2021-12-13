import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from xgboost import XGBClassifier
import numpy as np
from scipy.stats import uniform
from fairlearn.metrics import *
from sklearn.metrics import *
import pickle
from fairlearn.reductions import *
import scipy.stats.distributions as dists
from functools import partial

def performance_metrics(output_dir, classifier_id, y_test, y_predictions):
    classifier_names = ['SVC', 'RandomForestClassifier', 'XGBClassifier']
    with open(f'{output_dir}/{classifier_names[classifier_id]}_results.txt', 'w') as f:
        print("----- PERFORMANCE METRICS -----", file=f)
        print("--- ACCURACY SCORE ---", file=f)
        print(accuracy_score(y_test, y_predictions), file=f)
        print("--- PRECISION SCORE ---", file=f)
        print(precision_score(y_test, y_predictions, average='micro'), file=f)
        print("--- RECALL SCORE ---", file=f)
        print(recall_score(y_test, y_predictions, average='micro'), file=f)
        print("--- F1 SCORE ---", file=f)
        print(f1_score(y_test, y_predictions, average='micro'), file=f)
        print("--- CLASSIFICATION REPORT ---", file=f)
        print(classification_report(y_test, y_predictions), file=f)
        print("--- CONFUSION MATRIX ---", file=f)
        print(confusion_matrix(y_test, y_predictions), file=f)
        false_pos_rate, true_pos_rate, thresholds = roc_curve(y_test, y_predictions, pos_label=1)
        print("--- FALSE POSITIVE RATE ---", file=f)
        print(false_pos_rate, file=f)
        print("--- TRUE POSITIVE RATE ---", file=f)
        print(true_pos_rate, file=f)
        print("--- AREA UNDER CURVE ---", file=f)
        auc_score = auc(false_pos_rate, true_pos_rate)
        print(auc_score, file=f)
        print("--- ROC AUC ---", file=f)
        print(roc_auc_score(y_test, y_predictions, average='micro'), file=f)
    return auc_score

def predict_all_classifiers(output_dir, X, y, X_test, y_test, negative_positive_ratio, eval_metric=partial(roc_auc_score, average='micro'), seed= 1234):

    from sklearn.metrics import make_scorer
    scorer = make_scorer(roc_auc_score, average='micro')

    # Define the classifiers
    classifiers = [
        SVC(max_iter=1000, C=1e9, class_weight="balanced"),#SGDClassifier(loss='log', alpha=0.01, max_iter=2000, tol=0, class_weight='balanced'),s
        RandomForestClassifier(class_weight="balanced"),
        XGBClassifier(objective='binary:logistic', scale_pos_weight=negative_positive_ratio, use_label_encoder=False, seed=seed, eval_metric=eval_metric)
        ]

    # Define hyperparameters for the classifiers. 
    # Note: I did not use hyperparameters since performance is not a focus of the excersise as mentioned by the professor
    hyperparameters = [
        dict(C= uniform(0.1, 100), gamma=['scale', 'auto'], kernel=['linear', 'rbf']),
        dict(bootstrap= [True, False], max_features= ['auto', 'sqrt'], max_depth=dists.randint(5, 50), n_estimators= dists.randint(50, 200), min_samples_leaf= [1, 2, 4], min_samples_split= [2, 5, 10]),
        dict(eta=[0.1, 0.3], min_child_weight=[1, 3, 5], max_depth=dists.randint(4, 20), n_estimators=  dists.randint(50, 200), subsample= [0.6, 0.8, 1.0])
    ]

    # Define the crosss validation strategy
    # cv = KFold(n_splits=10)
    best_auc, best_y_pred, best_model_so_far = 0, None, None

    # Iterate all classifiers
    for i in range(len(classifiers)):
        # Select the classifier and its hyperparameters for the experimentation
        classifier = classifiers[i]
        hp = hyperparameters[i]

        # Define the hyperparameter search strategy and find the best model accordingly
        clf = RandomizedSearchCV(classifier, hp, n_iter=10, scoring =scorer, n_jobs=-1, verbose=0)
        clf.fit(X, y)
        best_model = clf.best_estimator_

        # Predict the labels given the test set's features
        y_pred = best_model.predict(X_test)

        with open(f'{output_dir}/{best_model.__class__.__name__}.npy', 'wb') as f:
            np.save(f, y_pred)
        
        # Evaluate the performance of the model based on the test set
        auc_score = performance_metrics(output_dir, i, y_test, y_pred)

        #fairmlhealth_metrics(best_model, X_test, y_test, y_pred)

        if best_auc < auc_score:
            best_y_pred = y_pred
            best_auc = auc_score
            best_model_so_far = best_model

    with open(f'{output_dir}/best_model.pckl', 'wb') as f:
        pickle.dump(best_model_so_far, f)
    
    return best_y_pred, best_auc, best_model_so_far
            