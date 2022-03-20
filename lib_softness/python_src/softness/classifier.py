import sys, os
import numpy as np
import numpy.ma as ma
import scipy as sp
import pandas as pd
import sklearn as skl
import sklearn.svm as svm
import sklearn.preprocessing as prep
import sklearn.pipeline as pipeline
import sklearn.model_selection as modsel
import sklearn.metrics as metrics
import sklearn.feature_selection as featsel
import sklearn.ensemble as ensemble



def get_classifier(df, x_col, y_col, random_state=777, clf_type='RF', kwargs=dict()):

    X = df[x_col].values
    Y = df[y_col].replace({True:1, False:-1}).values.ravel()


    if clf_type == 'SV':
        clf = pipeline.make_pipeline(prep.RobustScaler(),
                                 svm.LinearSVC(class_weight='balanced', random_state=random_state, **kwargs))
    elif clf_type == 'RF':
        clf = ensemble.RandomForestClassifier(class_weight='balanced', random_state=random_state, **kwargs)

    clf.fit(X, Y)

    print("score:", clf.score(X, Y))

    return clf




def calc_cv_score(df, x_col, y_col, random_state=777, clf_type='RF', clf_kwargs=dict(), cv_kwargs=dict(),
                  verbose=False, scoring={'balanced'}):

    X = df[x_col].values
    Y = df[y_col].replace({True:1, False:-1}).values.ravel()

    if clf_type == 'SV':
        clf = pipeline.make_pipeline(prep.RobustScaler(),
                                 svm.LinearSVC(class_weight='balanced', random_state=random_state, **clf_kwargs))
    elif clf_type == 'RF':
        clf = ensemble.RandomForestClassifier(class_weight='balanced', random_state=random_state, **clf_kwargs)

#     cv = modsel.StratifiedKFold(shuffle=True, random_state=random_state, **cv_kwargs)
    cv = modsel.RepeatedStratifiedKFold(random_state=random_state, **cv_kwargs)

    scores = {}

    if 'balanced' in scoring:
        scores['bal_acc'] = []

    if 'recall' in scoring:
        scores['recall_neg'] = []
        scores['recall_pos'] = []

    if 'auc' in scoring:
        scores['auc'] = []

    for train, test in cv.split(X, Y):

        X_train = X[train]
        Y_train = Y[train]

        X_test = X[test]
        Y_test = Y[test]


        clf.fit(X_train, Y_train)

        Y_pred = clf.predict(X_test)

        if 'balanced' in scoring:
            scores['bal_acc'].append(metrics.balanced_accuracy_score(Y_test, Y_pred))

        if 'recall' in scoring:

            recall_scores = metrics.recall_score(Y_test, Y_pred, average=None)
            scores['recall_neg'].append(recall_scores[0])
            scores['recall_pos'].append(recall_scores[1])

        if 'auc' in scoring:

            Y_score = clf.predict_proba(X_test)

            scores['auc'].append(metrics.roc_auc_score(Y_test, Y_score[:,1]))

    return scores


def calc_cv_score_with_subset(df, x_col, y_col, s_col, random_state=777, clf_type='RF', clf_kwargs=dict(), cv_kwargs=dict(), verbose=False, scoring={'balanced'}):

    # Use stratified K fold to split into three classes
    # First score is negative y_col vs. (positive y_col and positive s_col)
    # Second score is negative y_col vs. positive s_col

    X = df[x_col].values
    Y = df[y_col].replace({True:1, False:0}).values.ravel()+ df[s_col].replace({True:1, False:0}).values.ravel()


    if clf_type == 'SV':
        clf = pipeline.make_pipeline(prep.RobustScaler(),
                                 svm.LinearSVC(class_weight='balanced', random_state=random_state, **clf_kwargs))
    elif clf_type == 'RF':
        clf = ensemble.RandomForestClassifier(class_weight='balanced', random_state=random_state, **clf_kwargs)


#     cv = modsel.StratifiedKFold(shuffle=True, random_state=random_state, **cv_kwargs)
    cv = modsel.RepeatedStratifiedKFold(random_state=random_state, **cv_kwargs)

    scores = {}

    if 'balanced' in scoring:
        # balanced accuracy between negative and positive classes (average recall of two classes)
        scores['bal_acc'] = []
        # balanced accuracy between negative class and subset
        scores['bal_acc_sub'] = []

    if 'recall' in scoring:
        # recall of negative class
        scores['recall_neg'] = []
        # recall of postive class
        scores['recall_pos'] = []
        # recall of subset
        scores['recall_sub'] = []

    if 'auc' in scoring:
        # area under roc curve for negative vs positive class
        scores['auc'] = []
        # area under roc curve for negative vs subset
        scores['auc_sub'] = []
        # are under roc curve for positive vs subset
        scores['auc_pos_sub'] = []

    for train, test in cv.split(X, Y):

        X_train = X[train]
        Y_train = Y[train]

        X_test = X[test]
        Y_test = Y[test]


        clf.fit(X_train, np.where(Y_train==0, 0, 1))

        Y_true = np.where(Y_test==0, 0, 1)
        Y_pred = clf.predict(X_test)

        Y_sub_true = np.where(Y_test[np.nonzero(Y_test!=1)]==0, 0, 1)
        Y_sub_pred = clf.predict(X_test[np.nonzero(Y_test!=1)])

        if 'balanced' in scoring:
            scores['bal_acc'].append(metrics.balanced_accuracy_score(Y_true, Y_pred))
            scores['bal_acc_sub'].append(metrics.balanced_accuracy_score(Y_sub_true, Y_sub_pred))

        if 'recall' in scoring:

            recall_scores = metrics.recall_score(Y_true, Y_pred, average=None)
            scores['recall_neg'].append(recall_scores[0])
            scores['recall_pos'].append(recall_scores[1])
            scores['recall_sub'].append(metrics.recall_score(Y_sub_true, Y_sub_pred, pos_label=1))

        if 'auc' in scoring:

            Y_score = clf.predict_proba(X_test)
            Y_sub_score = clf.predict_proba(X_test[np.nonzero(Y_test!=1)])

            Y_pos_sub_true = np.where(Y_test[np.nonzero(Y_test!=0)]==1, 0, 1)
            Y_pos_sub_score = clf.predict_proba(X_test[np.nonzero(Y_test!=0)])

            scores['auc'].append(metrics.roc_auc_score(Y_true, Y_score[:,1]))
            scores['auc_sub'].append(metrics.roc_auc_score(Y_sub_true, Y_sub_score[:,1]))
            scores['auc_pos_sub'].append(metrics.roc_auc_score(Y_pos_sub_true, Y_pos_sub_score[:,1]))

    return scores


def calc_brute_rfe_scores(df, x_col, y_col, random_state=777, clf_type='RF', clf_kwargs=dict(), cv_kwargs=dict(),
                          verbose=False, scoring={'balanced'}):

    scores = calc_cv_score(df, x_col, y_col, random_state=random_state,
                           clf_type=clf_type, clf_kwargs=clf_kwargs, cv_kwargs=cv_kwargs, verbose=verbose, scoring=scoring)

    order = []
    rfe_scores = {score_type:[np.mean(scores[score_type])] for score_type in scores}
    rfe_scores_err = {score_type:[np.std(scores[score_type])] for score_type in scores}


    print("Accuracy", rfe_scores['bal_acc'][-1], "+\-", rfe_scores_err['bal_acc'][-1])


    features = x_col.copy()
    for i in range(len(x_col)-1):

        print("Feature", i+1, "/", len(x_col))

        test_scores = {score_type:[] for score_type in rfe_scores}
        test_scores_err = {score_type:[] for score_type in rfe_scores}

        for fi in range(len(features)):
            test_feats = np.delete(features, fi)


            scores = calc_cv_score(df, test_feats, y_col, random_state=random_state,
                           clf_type=clf_type, clf_kwargs=clf_kwargs, cv_kwargs=cv_kwargs, verbose=verbose, scoring=scoring)

            for score_type in scores:
                test_scores[score_type].append(np.mean(scores[score_type]))
                test_scores_err[score_type].append(np.std(scores[score_type]))


        imax = np.argmax(test_scores['bal_acc'])

        print("Removed:", features[imax])

        order.append(features[imax])

        features = np.delete(features, imax)

        for score_type in test_scores:
            rfe_scores[score_type].append(test_scores[score_type][imax])
            rfe_scores_err[score_type].append(test_scores_err[score_type][imax])


        print("Accuracy", rfe_scores['bal_acc'][-1], "+\-", rfe_scores_err['bal_acc'][-1])


    order.append(features[0])

    return (order, rfe_scores, rfe_scores_err)



def calc_brute_rfe_scores_with_subset(df, x_col, y_col, s_col, random_state=777, clf_type='RF',
                                      clf_kwargs=dict(), cv_kwargs=dict(), verbose=False, scoring={'balanced'}):


    scores = calc_cv_score_with_subset(df, x_col, y_col, s_col, random_state=random_state,
                           clf_type=clf_type, clf_kwargs=clf_kwargs, cv_kwargs=cv_kwargs, verbose=verbose, scoring=scoring)

    order = []
    rfe_scores = {score_type:[np.mean(scores[score_type])] for score_type in scores}
    rfe_scores_err = {score_type:[np.std(scores[score_type])] for score_type in scores}


    print("Accuracy", rfe_scores['bal_acc'][-1], "+\-", rfe_scores_err['bal_acc'][-1])
    print("Subset Accuracy", rfe_scores['bal_acc_sub'][-1], "+\-", rfe_scores_err['bal_acc_sub'][-1])


    features = x_col.copy()
    for i in range(len(x_col)-1):

        print("Feature", i+1, "/", len(x_col))

        test_scores = {score_type:[] for score_type in rfe_scores}
        test_scores_err = {score_type:[] for score_type in rfe_scores}

        for fi in range(len(features)):
            test_feats = np.delete(features, fi)


            scores = calc_cv_score_with_subset(df, test_feats, y_col, s_col, random_state=random_state,
                           clf_type=clf_type, clf_kwargs=clf_kwargs, cv_kwargs=cv_kwargs, verbose=verbose, scoring=scoring)

            for score_type in scores:
                test_scores[score_type].append(np.mean(scores[score_type]))
                test_scores_err[score_type].append(np.std(scores[score_type]))


        imax = np.argmax(test_scores['bal_acc'])

        print("Removed:", features[imax])

        order.append(features[imax])

        features = np.delete(features, imax)

        for score_type in test_scores:
            rfe_scores[score_type].append(test_scores[score_type][imax])
            rfe_scores_err[score_type].append(test_scores_err[score_type][imax])


        print("Accuracy", rfe_scores['bal_acc'][-1], "+\-", rfe_scores_err['bal_acc'][-1])
        print("Subset Accuracy", rfe_scores['bal_acc_sub'][-1], "+\-", rfe_scores_err['bal_acc_sub'][-1])


    order.append(features[0])

    return (order, rfe_scores, rfe_scores_err)



# def calc_MI_rfe_scores(df, x_col, y_col, random_state=777, n_splits=8, clf_type='RF', kwargs=dict()):



#     if clf_type == 'SV':
#         clf = pipeline.make_pipeline(prep.RobustScaler(),
#                                  svm.LinearSVC(class_weight='balanced', random_state=random_state, **kwargs))
#     elif clf_type == 'RF':
#         clf = ensemble.RandomForestClassifier(class_weight='balanced', random_state=random_state, **kwargs)


#     cv = modsel.StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

#     rfe_score = []
#     rfe_score_err = []
#     order = []

#     X = df[x_col].values
#     Y = df[y_col].replace({True:1, False:-1}).values.ravel()


#     x_mi = np.full([len(x_col), len(x_col)], np.nan, float)
#     for i in range(len(x_col)):
#         for j in range(i+1, len(x_col)):
#             x_mi[i, j] = metrics.adjusted_mutual_info_score(X[:, i], X[:, j], average_method='geometric')

#     x_mi = ma.masked_invalid(x_mi)

# #     print(x_mi)

#     y_mi = np.zeros(len(x_col), float)
#     for i in range(len(x_col)):
#         y_mi[i] = metrics.adjusted_mutual_info_score(X[:, i], Y, average_method='geometric')

# #     print(y_mi)

#     scores = modsel.cross_validate(clf, X, Y, scoring={'accuracy': 'accuracy'},
#                                    cv=cv, return_train_score=False)

#     rfe_score.append(scores['test_accuracy'].mean())
#     rfe_score_err.append(scores['test_accuracy'].std() / np.sqrt(len(scores['test_accuracy'])))

#     print("Accuracy", rfe_score[-1], "+\-", rfe_score_err[-1])

#     features = x_col.copy()
#     for i in range(len(x_col)-1):

#         (f1, f2) = np.unravel_index(x_mi.argmax(), x_mi.shape)

#         if y_mi[f1] > y_mi[f2]:
#             fmax = f1
#         else:
#             fmax = f2

#         fmin = y_mi.argmin()

#         test_feats = np.delete(features, fmax)

#         X = df[test_feats].values

#         scores = modsel.cross_validate(clf, X, Y, scoring={'accuracy': 'accuracy'},
#                                cv=cv, return_train_score=False)

#         max_score = scores['test_accuracy'].mean()
#         max_score_err = scores['test_accuracy'].std() / np.sqrt(len(scores['test_accuracy']))

#         print("Max Corr:", features[fmax], "Score", max_score, "+\-", max_score_err)


#         test_feats = np.delete(features, fmin)

#         X = df[test_feats].values

#         scores = modsel.cross_validate(clf, X, Y, scoring={'accuracy': 'accuracy'},
#                                cv=cv, return_train_score=False)

#         min_score = scores['test_accuracy'].mean()
#         min_score_err = scores['test_accuracy'].std() / np.sqrt(len(scores['test_accuracy']))

#         print("Min Info:", features[fmin], "Score", min_score, "+\-", min_score_err)

#         if min_score > max_score:
#             imax = fmin

#             rfe_score.append(min_score)
#             rfe_score_err.append(min_score_err)

#         else:
#             imax = fmax

#             rfe_score.append(max_score)
#             rfe_score_err.append(max_score_err)

#         print("Removed:", features[imax])

#         order.append(features[imax])

#         features = np.delete(features, imax)

#         x_mi = np.delete(x_mi, imax, axis=0)
#         x_mi = np.delete(x_mi, imax, axis=1)

#         x_mi = ma.masked_invalid(x_mi)

# #         print(x_mi)

#         y_mi = np.delete(y_mi, imax)



#         print("Accuracy", rfe_score[-1], "+\-", rfe_score_err[-1])


#     order.append(features[0])

#     return (order, rfe_score, rfe_score_err)


# def calc_MI_rfe_scores_with_subset(df, df_subset, x_col, y_col, random_state=777, n_splits=8, clf_type='RF', kwargs=dict()):

#     Y_subset = df_subset[y_col].replace({True:1, False:-1}).values.ravel()

#     subset_cv = modsel.StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


#     def subset_score(estimator, X, y):

#         train, test = next(split)

#         X_test = X_subset[test]
#         Y_test = Y_subset[test]

#         return estimator.score(X_test, Y_test)


#     if clf_type == 'SV':
#         clf = pipeline.make_pipeline(prep.RobustScaler(),
#                                  svm.LinearSVC(class_weight='balanced', random_state=random_state, **kwargs))
#     elif clf_type == 'RF':
#         clf = ensemble.RandomForestClassifier(class_weight='balanced', random_state=random_state, **kwargs)
#     elif clf_type == "ET":
#         clf = ensemble.ExtraTreesClassifier(class_weight='balanced', random_state=random_state, **kwargs)


#     cv = modsel.StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

#     rfe_score = []
#     rfe_score_err = []
#     rfe_subset_score = []
#     rfe_subset_score_err = []
#     order = []

#     X = df[x_col].values
#     Y = df[y_col].replace({True:1, False:-1}).values.ravel()


#     x_mi = np.full([len(x_col), len(x_col)], np.nan, float)
#     for i in range(len(x_col)):
#         for j in range(i+1, len(x_col)):
#             x_mi[i, j] = metrics.adjusted_mutual_info_score(X[:, i], X[:, j], average_method='geometric')

#     x_mi = ma.masked_invalid(x_mi)

# #     print(x_mi)

#     y_mi = np.zeros(len(x_col), float)
#     for i in range(len(x_col)):
#         y_mi[i] = metrics.adjusted_mutual_info_score(X[:, i], Y, average_method='geometric')

# #     print(y_mi)

#     X_subset = df_subset[x_col].values
#     split = subset_cv.split(X_subset, Y_subset, random_state=random_state)

#     scores = modsel.cross_validate(clf, X, Y, scoring={'accuracy': 'accuracy', 'subset': subset_score},
#                                    cv=cv, return_train_score=False)

#     rfe_score.append(scores['test_accuracy'].mean())
#     rfe_score_err.append(scores['test_accuracy'].std() / np.sqrt(len(scores['test_accuracy'])))

#     rfe_subset_score.append(scores['test_subset'].mean())
#     rfe_subset_score_err.append(scores['test_subset'].std() / np.sqrt(len(scores['test_subset'])))

#     print("Accuracy", rfe_score[-1], "+\-", rfe_score_err[-1])
#     print("Subset Accuracy", rfe_subset_score[-1], "+\-", rfe_subset_score_err[-1])

#     features = x_col.copy()
#     for i in range(len(x_col)-1):

#         (f1, f2) = np.unravel_index(x_mi.argmax(), x_mi.shape)

#         if y_mi[f1] > y_mi[f2]:
#             fmax = f1
#         else:
#             fmax = f2

#         fmin = y_mi.argmin()

#         test_feats = np.delete(features, fmax)

#         X = df[test_feats].values

#         X_subset = df_subset[test_feats].values
#         split = subset_cv.split(X_subset, Y_subset, random_state=random_state)

#         scores = modsel.cross_validate(clf, X, Y, scoring={'accuracy': 'accuracy', 'subset': subset_score},
#                                    cv=cv, return_train_score=False)

#         max_score = scores['test_accuracy'].mean()
#         max_score_err = scores['test_accuracy'].std() / np.sqrt(len(scores['test_accuracy']))

#         max_subset_score = scores['test_subset'].mean()
#         max_subset_score_err = scores['test_subset'].std() / np.sqrt(len(scores['test_subset']))

#         print("Max Corr:", features[fmax], "Score", max_score, "+\-", max_score_err)


#         test_feats = np.delete(features, fmin)

#         X = df[test_feats].values

#         X_subset = df_subset[test_feats].values
#         split = subset_cv.split(X_subset, Y_subset, random_state=random_state)

#         scores = modsel.cross_validate(clf, X, Y, scoring={'accuracy': 'accuracy', 'subset': subset_score},
#                                    cv=cv, return_train_score=False)

#         min_score = scores['test_accuracy'].mean()
#         min_score_err = scores['test_accuracy'].std() / np.sqrt(len(scores['test_accuracy']))

#         min_subset_score = scores['test_subset'].mean()
#         min_subset_score_err = scores['test_subset'].std() / np.sqrt(len(scores['test_subset']))

#         print("Min Info:", features[fmin], "Score", min_score, "+\-", min_score_err)

#         if min_score > max_score:
#             imax = fmin

#             rfe_score.append(min_score)
#             rfe_score_err.append(min_score_err)
#             rfe_subset_score.append(min_subset_score)
#             rfe_subset_score_err.append(min_subset_score_err)

#         else:
#             imax = fmax

#             rfe_score.append(max_score)
#             rfe_score_err.append(max_score_err)
#             rfe_subset_score.append(max_subset_score)
#             rfe_subset_score_err.append(max_subset_score_err)

#         print("Removed:", features[imax])

#         order.append(features[imax])

#         features = np.delete(features, imax)

#         x_mi = np.delete(x_mi, imax, axis=0)
#         x_mi = np.delete(x_mi, imax, axis=1)

#         x_mi = ma.masked_invalid(x_mi)

# #         print(x_mi)

#         y_mi = np.delete(y_mi, imax)



#         print("Accuracy", rfe_score[-1], "+\-", rfe_score_err[-1])
#         print("Subset Accuracy", rfe_subset_score[-1], "+\-", rfe_subset_score_err[-1])


#     order.append(features[0])

#     return (order, rfe_score, rfe_score_err, rfe_subset_score, rfe_subset_score_err)


def calc_rfe_scores(df, x_col, y_col, random_state=777, cv=False, n_splits=8, loss='squared_hinge'):

     X = df[x_col].values
     Y = df[y_col].replace({True:1, False:-1}).values.ravel()

     scaler = prep.RobustScaler()
     x = scaler.fit_transform(X)

     clf = svm.LinearSVC(class_weight='balanced', random_state=random_state, C=1e0, loss=loss)

     features = x_col.copy()

     scores = []
     min_coeffs = []
     for i in range(X.shape[1]):

         print(i, "/", X.shape[1], "Features:", features)

         clf.fit(x, Y)

         print("Weights:", clf.coef_)

         feature_importances = clf.coef_[0]
         imin = np.argmin(np.abs(feature_importances))

         min_coeffs.append(feature_importances[imin])

         if cv:

             cvscores = modsel.cross_validate(clf, x, Y, scoring={'accuracy': 'accuracy'},
                                    cv=modsel.StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state),
                                            return_train_score=False)

             score = np.mean(cvscores['test_accuracy'])
         else:
             score = clf.score(x, Y)

         scores.append(score)



         print("Score:", score, "Min Feature:", features[imin], "Coeff:", feature_importances[imin])

         x = np.delete(x, imin, axis=1)
         features = np.delete(features, imin)

     return (scores, min_coeffs)



# def tune_hyper_params(df, x_col, y_col, random_state=777, n_splits=8):

#     X = df[x_col].values
#     Y = df[y_col].replace({True:1, False:-1}).values.ravel()

#     pipe = pipeline.make_pipeline(prep.RobustScaler(),
#                                  svm.LinearSVC(class_weight='balanced', random_state=random_state))

#     print(pipe.named_steps)

#     cv = modsel.StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

#     parameters = {
#         'linearsvc__loss': ['hinge', 'squared_hinge'],
#         'linearsvc__C': [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]
#     }

#     grid_search = modsel.GridSearchCV(pipe, parameters, cv=n_splits, verbose=2)

#     grid_search.fit(X, Y)

#     df = pd.DataFrame(grid_search.cv_results_)

#     return df

def calc_dist_score(df, y_col, random_state=777,max_dist=8, clf_type='RF', clf_kwargs=dict(), cv_kwargs=dict(),
                  verbose=False, scoring={'balanced'}):

    scores = []
    names = ['particle_type']
    for l in range(max_dist):
        names += [(l+1,'g')]
        names += [(l+1,'o')]
        scores += [calc_cv_score(df, names,y_col, random_state=777, clf_type='RF', clf_kwargs=dict(), cv_kwargs=dict(), 
                          verbose=False, scoring={'balanced'})]


    return scores


def get_scaler_and_classifier(df, x_col, y_col, random_state=777,  loss='squared_hinge'):

    X = df[x_col].values
    Y = df[y_col].replace({True:1, False:-1}).values.ravel()

    scaler = prep.RobustScaler()
    x = scaler.fit_transform(X)

    clf = svm.LinearSVC(class_weight='balanced', random_state=random_state, C=1e0, loss=loss,max_iter=10000)

    features = x_col.copy()


    clf.fit(x, Y)
    return scaler, clf
