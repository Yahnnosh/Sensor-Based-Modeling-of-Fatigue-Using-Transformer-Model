import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, \
    f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, LeaveOneGroupOut, train_test_split
from scipy.stats import ttest_ind
from tqdm import tqdm
import os
import json
import warnings

SEED = 2022


def evaluator(y_pred, y_true, verbose=False):
    """Returns evaluation metric scores"""
    accuracy = accuracy_score(y_pred=y_pred, y_true=y_true)
    balanced_accuracy = balanced_accuracy_score(y_pred=y_pred, y_true=y_true)
    f1 = f1_score(y_pred=y_pred, y_true=y_true, average='weighted')
    recall = recall_score(y_pred=y_pred, y_true=y_true, average='weighted')
    precision = precision_score(y_pred=y_pred, y_true=y_true, average='weighted')
    confusion = confusion_matrix(y_pred=y_pred, y_true=y_true)

    # display scores
    if verbose:
        ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=[False, True]).plot(cmap=plt.cm.Blues)
        plt.title('Physical fatigue')

        print(f'accuracy: {accuracy}\n'
              f'balanced accuracy: {balanced_accuracy}\n'
              f'f1 (weighted): {f1}\n'
              f'recall (weighted): {recall}\n'
              f'precision (weighted): {precision}')

    return {'accuracy': accuracy,
            'balanced_accuracy': balanced_accuracy,
            'f1': f1,
            'recall': recall,
            'precision': precision}


def aggregate_by_day(values, indices, metadata):
    """
    Aggregates values along individual days
    :param values: e.g. labels or label predictions
    :param indices: indices of values w.r.t dataset
    :param metadata: metadata for FULL(!) dataset
    :return:
    """
    # day information for each segment
    metadata_selection = np.array(metadata)[indices]
    days_selection = [(meta['date'], meta['subjectID']) for meta in
                      metadata_selection]  # note that same date can be used by different subjects

    # assign values to corresponding day
    values_daily = {key: [] for key in set(days_selection)}
    for key, pred in zip(days_selection, values):
        values_daily[key].append(pred)

    return values_daily


# TODO: check for multiclass predictions
def daily_majority_vote(y_pred_segments, indices, metadata):
    """Predicts majority class from segement predictions"""
    # aggregate by day
    y_pred_daily = aggregate_by_day(y_pred_segments, indices, metadata)

    # majority vote
    for day, day_preds in y_pred_daily.items():
        if len(day_preds) == 1:
            y_pred_daily[day] = day_preds[0]
        else:
            county_by_label = np.bincount(day_preds)
            y_pred_daily[day] = np.argmax(county_by_label)

    return y_pred_daily


def agreements(y_pred_segments, indices, metadata):
    """Calculates percentage of agreements between predictions of same-day segments"""
    # aggregate by day
    y_pred_daily = aggregate_by_day(y_pred_segments, indices, metadata)

    # agreements by day
    agreements = {key: 0 for key in y_pred_daily.keys()}
    for day, day_preds in y_pred_daily.items():
        if len(day_preds) == 1:
            agreements[day] = 1  # only one value -> agrees by default
        else:
            agreements[day] = int(np.sum(day_preds) == len(day_preds) or np.sum(day_preds) == 0)

    return np.mean(list(agreements.values()))


# TODO: check if assumption correct, that we do not need X data to create the splits
# TODO: currently only working for binary classification -> make for multiclass
def stratified_group_k_fold(path, groups, model, images, folds=5, verbose=True, variable=0) -> dict:
    """
    Calculates stratified group k-fold for a specified model
    :param path: path to folder where data is stored
    :param groups: subjectID for all datapoints in dataset
    :param model: model implementing train, fit, predict
    :param folds: k folds
    :param verbose: whether to print mean test set metrics
    :param images: whether to use images or statistical features
    :param variable: which variable to use (0 -> phF or 1 -> MF)
    :return: dictionary with test set metrics for each fold
    """
    assert variable in (0, 1)

    # calculate data size
    if images:
        N = sum([1 for p in os.listdir(path) if (p[:14] == 'feature_vector' and p[:19] != 'feature_vector_stat')])
    else:
        N = sum([1 for p in os.listdir(path) if p[:19] == 'feature_vector_stat'])

    # load labels (we need them for stratification)
    y = np.empty(N, dtype=int)
    for i in range(N):
        if images:
            y[i] = np.load(path + f'/labels{i}.npy', allow_pickle=True)[variable]  # TODO: multiclass
        else:
            y[i] = np.load(path + f'/labels_stat{i}.npy', allow_pickle=True)[variable]  # TODO: multiclass

    # CV
    cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=SEED)
    scores_cv = []
    data_indices = np.arange(N)

    print(f'Starting stratified group {folds}-fold for {["physical fatigue", "mental fatigue"][variable]}')
    with tqdm(total=folds) as pbar:
        for i, (train_indices, test_indices) in enumerate(cv.split(X=data_indices, y=y, groups=groups)):
            # test labels
            y_test = y[test_indices]

            # training
            try:
                model.reset()
            except AttributeError:
                pass  # model has no trainable parameters
            model.fit(train_indices)

            # predict
            y_pred = model.predict(test_indices)

            # evaluate
            scores = evaluator(y_pred, y_test, verbose=False)
            scores_cv.append(scores)

            # for progress bar
            pbar.update(1)
            pbar.set_description(f' Fold {i + 1} F1: {scores["f1"]}')

    # print (if verbose==True)
    if verbose:
        print('Performance model:')
        metrics = scores_cv[0].keys()
        for metric in metrics:
            mean = np.mean([scores_cv_i[metric] for scores_cv_i in scores_cv])
            std = np.std([scores_cv_i[metric] for scores_cv_i in scores_cv])
            print(f' {metric}: {round(mean, 3)} +- {round(std, 3)} \n')

    return scores_cv


# TODO: check if assumption correct, that we do not need X data to create the splits
# TODO: currently only working for binary classification -> make for multiclass
def stratified_k_fold(path, model, images, folds=5, verbose=True, variable=0) -> dict:
    """
    Calculates stratified k-fold for a specified model
    :param path: path to folder where data is stored
    :param model: model implementing train, fit, predict
    :param folds: k folds
    :param verbose: whether to print mean test set metrics
    :param images: whether to use images or statistical features
    :param variable: which variable to use (0 -> phF or 1 -> MF)
    :return: dictionary with test set metrics for each fold
    """
    assert variable in (0, 1)

    # calculate data size
    if images:
        N = sum([1 for p in os.listdir(path) if (p[:14] == 'feature_vector' and p[:19] != 'feature_vector_stat')])
    else:
        N = sum([1 for p in os.listdir(path) if p[:19] == 'feature_vector_stat'])

    # load labels (we need them for stratification)
    y = np.empty(N, dtype=int)
    for i in range(N):
        if images:
            y[i] = np.load(path + f'/labels{i}.npy', allow_pickle=True)[variable]  # TODO: multiclass
        else:
            y[i] = np.load(path + f'/labels_stat{i}.npy', allow_pickle=True)[variable]  # TODO: multiclass

    # CV
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    scores_cv = []
    data_indices = np.arange(N)

    print(f'Starting stratified {folds}-fold for {["physical fatigue", "mental fatigue"][variable]}')
    with tqdm(total=folds) as pbar:
        for i, (train_indices, test_indices) in enumerate(cv.split(X=data_indices, y=y)):
            # test labels
            y_test = y[test_indices]

            # training
            try:
                model.reset()
            except AttributeError:
                pass  # model has no trainable parameters
            model.fit(train_indices)

            # predict
            y_pred = model.predict(test_indices)

            # evaluate
            scores = evaluator(y_pred, y_test, verbose=False)
            scores_cv.append(scores)

            # for progress bar
            pbar.update(1)
            pbar.set_description(f' Fold {i + 1} F1: {scores["f1"]}')

    # print (if verbose==True)
    if verbose:
        print('Performance model:')
        metrics = scores_cv[0].keys()
        for metric in metrics:
            mean = np.mean([scores_cv_i[metric] for scores_cv_i in scores_cv])
            std = np.std([scores_cv_i[metric] for scores_cv_i in scores_cv])
            print(f' {metric}: {round(mean, 3)} +- {round(std, 3)} \n')

    return scores_cv


# TODO: check if assumption correct, that we do not need X data to create the splits
# TODO: currently only working for binary classification -> make for multiclass
def leave_one_subject_out(path, groups, model, images, verbose=True, variable=0) -> dict:
    """
    Calculates LOSO for a specified model
    :param path: path to folder where data is stored
    :param groups: subjectID for all datapoints in dataset
    :param model: model implementing train, fit, predict
    :param verbose: whether to print mean test set metrics
    :param images: whether to use images or statistical features
    :param variable: which variable to use (0 -> phF or 1 -> MF)
    :return: dictionary with test set metrics for each fold
    """
    assert variable in (0, 1)

    # calculate data size
    if images:
        N = sum([1 for p in os.listdir(path) if (p[:14] == 'feature_vector' and p[:19] != 'feature_vector_stat')])
    else:
        N = sum([1 for p in os.listdir(path) if p[:19] == 'feature_vector_stat'])

    # load labels (we need them for stratification)
    y = np.empty(N, dtype=int)
    for i in range(N):
        if images:
            y[i] = np.load(path + f'/labels{i}.npy', allow_pickle=True)[variable]  # TODO: multiclass
        else:
            y[i] = np.load(path + f'/labels_stat{i}.npy', allow_pickle=True)[variable]  # TODO: multiclass

    # CV
    cv = LeaveOneGroupOut()
    scores_cv = []
    data_indices = np.arange(N)

    print(f'Starting leave-one-subject-out for {["physical fatigue", "mental fatigue"][variable]}')
    with tqdm(total=len(np.unique(groups))) as pbar:
        for i, (train_indices, test_indices) in enumerate(cv.split(X=data_indices, y=y, groups=groups)):
            # test labels
            y_test = y[test_indices]

            # training
            try:
                model.reset()
            except AttributeError:
                pass  # model has no trainable parameters
            model.fit(train_indices)

            # predict
            y_pred = model.predict(test_indices)

            # evaluate
            scores = evaluator(y_pred, y_test, verbose=False)
            scores_cv.append(scores)

            # for progress bar
            pbar.update(1)
            pbar.set_description(f' Fold {i + 1} F1: {scores["f1"]}')

    # print (if verbose==True)
    if verbose:
        print('Performance model:')
        metrics = scores_cv[0].keys()
        for metric in metrics:
            mean = np.mean([scores_cv_i[metric] for scores_cv_i in scores_cv])
            std = np.std([scores_cv_i[metric] for scores_cv_i in scores_cv])
            print(f' {metric}: {round(mean, 3)} +- {round(std, 3)} \n')

    return scores_cv


# TODO: check if assumption correct, that we do not need X data to create the splits
# TODO: currently only working for binary classification -> make for multiclass
def stratified_train_test(path, model, test_size, images, verbose=True, variable=0) -> dict:
    """
    Calculates performance of specified model under one stratified train/test split
    :param path: path to folder where data is stored
    :param model: model implementing train, fit, predict
    :param test_size: size of test set (between 0 to 1)
    :param verbose: whether to print mean test set metrics
    :param images: whether to use images or statistical features
    :param variable: which variable to use (0 -> phF or 1 -> MF)
    :return: dictionary with test set metrics for each fold
    """
    assert variable in (0, 1)

    # calculate data size
    if images:
        N = sum([1 for p in os.listdir(path) if (p[:14] == 'feature_vector' and p[:19] != 'feature_vector_stat')])
    else:
        N = sum([1 for p in os.listdir(path) if p[:19] == 'feature_vector_stat'])

    # load labels (we need them for stratification)
    y = np.empty(N, dtype=int)
    for i in range(N):
        if images:
            y[i] = np.load(path + f'/labels{i}.npy', allow_pickle=True)[variable]  # TODO: multiclass
        else:
            y[i] = np.load(path + f'/labels_stat{i}.npy', allow_pickle=True)[variable]  # TODO: multiclass

    # train/test split
    data_indices = np.arange(N)
    train_indices, test_indices, y_train, y_test = train_test_split(data_indices, y, test_size=test_size, shuffle=True,
                                                                    random_state=SEED, stratify=y)

    print(f'Starting stratified train/test for {["physical fatigue", "mental fatigue"][variable]}')

    # training
    model.fit(train_indices)

    # predict
    y_pred = model.predict(test_indices)

    # evaluate
    scores = evaluator(y_pred, y_test, verbose=False)

    # print (if verbose==True)
    if verbose:
        print('Performance model:')
        for metric, score in scores.items():
            print(f' {metric}: {round(score, 3)} +- {round(score, 3)} \n')

    return scores


def scores_tables():
    """Prints out all scores in LaTeX tables"""
    phF = []
    MF = []

    # not very pretty but works
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')

        score_path = './Scores'

        for path in os.listdir(score_path):
            # separate tables for phF/MF
            table_phF = {'accuracy': [],
                         'balanced_accuracy': [],
                         'f1': [],
                         'recall': [],
                         'precision': []}
            table_MF = {'accuracy': [],
                        'balanced_accuracy': [],
                        'f1': [],
                        'recall': [],
                        'precision': []}
            for model in os.listdir(score_path + '/' + path):
                full_path = score_path + '/' + path + '/' + model

                # load scores for model
                with open(full_path) as f:
                    scores = f.read()
                    scores = scores.replace('\'', '\"')
                    #scores = scores.replace(', None', '')
                scores = json.loads(scores)

                # phF/MF
                for variable in (0, 1):
                    # aggregate all metrics of different folds for specific model
                    aggregation = {'accuracy': [],
                                   'balanced_accuracy': [],
                                   'f1': [],
                                   'recall': [],
                                   'precision': []}
                    for fold in scores[variable]:
                        for metric in aggregation.keys():
                            aggregation[metric].append(fold[metric])

                    for metric in aggregation.keys():
                        mean = np.round(np.mean(aggregation[metric]), 3)
                        std = np.round(np.std(aggregation[metric]), 3)
                        entry = f'{mean} ± {std}'

                        # put mean +- std of scores into table for each model
                        if variable == 0:
                            table_phF[metric].append(entry)
                        else:
                            table_MF[metric].append(entry)

            # prettier metric names
            table_phF = {'Accuracy': table_phF['accuracy'],
                         'Balanced accuracy': table_phF['balanced_accuracy'],
                         'F1-score': table_phF['f1'],
                         'Recall': table_phF['recall'],
                         'Precision': table_phF['precision']}
            table_MF = {'Accuracy': table_MF['accuracy'],
                        'Balanced accuracy': table_MF['balanced_accuracy'],
                        'F1-score': table_MF['f1'],
                        'Recall': table_MF['recall'],
                        'Precision': table_MF['precision']}
            caption_phF = {'loso': 'LOSO (Physical fatigue)',
                           'strat_5_fold': 'Stratified 5-fold (Physical fatigue)',
                           'strat_group_5_fold': 'Stratified group 5-fold (Physical fatigue)'}[path]
            caption_MF = {'loso': 'LOSO (Mental fatigue)',
                          'strat_5_fold': 'Stratified 5-fold (Mental fatigue)',
                          'strat_group_5_fold': 'Stratified group 5-fold (Mental fatigue)'}[path]

            # to latex
            """models = [{'cnn.txt': 'CNN',
                       'cnn2.txt': 'CNN (2nd run)',
                       'majority_voting.txt': 'Majority Voting',
                       'random_guess.txt': 'Random Guess',
                       'xgboost.txt': 'XGBoost',
                       'random_forest.txt': 'Random Forest'}[model] for model in os.listdir(score_path + '/' + path)]"""
            models = [model.capitalize().replace('Cnn', 'CNN').replace('Xgboost', 'XGBoost').
                          replace('_', ' ').replace('.txt', '').replace('forest', 'Forest').replace('guess', 'Guess')
                      for model in os.listdir(score_path + '/' + path)]
            models = pd.Series(models)
            # pHF
            df = pd.DataFrame(table_phF)
            df = df.set_index([models])
            # make largest value bold
            for column_name in list(df.columns):
                index_max = np.argmax([float(entry.split('±')[0]) for entry in df[column_name].to_numpy()])
                entry = df[column_name].iloc[index_max]
                entry = 'BOLD{' + entry + '}'
                df[column_name].iloc[index_max] = entry
            phF.append(df.to_latex(index=True,
                                   bold_rows=True,
                                   caption=caption_phF,
                                   column_format='llllll',
                                   position='H'))
            # MF
            df = pd.DataFrame(table_MF)
            df = df.set_index([models])
            # make largest value bold
            for column_name in list(df.columns):
                index_max = np.argmax([float(entry.split('±')[0]) for entry in df[column_name].to_numpy()])
                entry = df[column_name].iloc[index_max]
                entry = 'BOLD{' + entry + '}'
                df[column_name].iloc[index_max] = entry
            MF.append(df.to_latex(index=True,
                                  bold_rows=True,
                                  caption=caption_MF,
                                  column_format='llllll',
                                  position='H'))

    # print
    for entry in phF:
        print(entry.replace('BOLD\\', r'\textbf').replace('\}', '}'))
    for entry in MF:
        print(entry.replace('BOLD\\', r'\textbf').replace('\}', '}'))


def p_value_tables():
    """Calculates statistical performance significance compared to baselines"""
    phF = []
    MF = []

    # not very pretty but works
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')

        score_path = './Scores'

        for path in os.listdir(score_path):

            # store scores for each model
            models = {}
            for model in os.listdir(score_path + '/' + path):
                full_path = score_path + '/' + path + '/' + model

                # load scores for model
                with open(full_path) as f:
                    scores = f.read()
                scores = json.loads(scores.replace('\'', '\"'))

                # phF/MF
                for variable in (0, 1):
                    # aggregate all metrics of different folds for specific model
                    aggregation = {'accuracy': [],
                                   'balanced_accuracy': [],
                                   'f1': [],
                                   'recall': [],
                                   'precision': []}
                    for fold in scores[variable]:
                        for metric in aggregation.keys():
                            aggregation[metric].append(fold[metric])

                    # store scores in models
                    if variable == 0:
                        models[model] = {variable: aggregation}
                    else:
                        models[model][variable] = aggregation

            # build tables
            caption_phF = {'loso': 'LOSO - p-values (Physical fatigue)',
                           'strat_5_fold': 'Stratified 5-fold - p-values (Physical fatigue)',
                           'strat_group_5_fold': 'Stratified group 5-fold - p-values (Physical fatigue)'}[path]
            caption_MF = {'loso': 'LOSO - p-values (Mental fatigue)',
                          'strat_5_fold': 'Stratified 5-fold - p-values (Mental fatigue)',
                          'strat_group_5_fold': 'Stratified group 5-fold - p-values (Mental fatigue)'}[path]
            # phF
            models_name = [model for model in os.listdir(score_path + '/' + path)]
            baselines_name = ['majority_voting.txt', 'random_guess.txt']
            non_baselines_name = [model_name for model_name in models_name if
                                  model_name not in baselines_name]
            n_models = len(non_baselines_name) * 2
            df = pd.DataFrame({'accuracy': [pd.NA]*n_models,
                               'balanced_accuracy': [pd.NA]*n_models,
                               'f1': [pd.NA]*n_models,
                               'recall': [pd.NA]*n_models,
                               'precision': [pd.NA]*n_models})
            df = df.set_index(pd.Series([non_baseline.replace('.txt', '') +
                                         '/' + baseline.replace('.txt', '')
                                         for non_baseline in non_baselines_name
                                         for baseline in baselines_name]))
            # calculate p-value
            for metric in list(aggregation.keys()):
                i = 0
                for non_baseline in non_baselines_name:
                    for baseline in baselines_name:
                        baseline_scores = models[baseline][0][metric]
                        model_scores = models[non_baseline][0][metric]
                        _, p_value = ttest_ind(baseline_scores, model_scores, equal_var=False)
                        df[metric].iloc[i] = round(p_value, 3)
                        i += 1
            # make value bold if p < 0.05
            for column_name in list(df.columns):
                for row in range(df.shape[0]):
                    entry = df[column_name].iloc[row]
                    if float(entry) < 0.05:
                        entry = 'BOLD{' + str(entry) + '}'
                        df[column_name].iloc[row] = entry
            # prettier names
            df = df.rename(columns={"accuracy": "Accuracy", "balanced_accuracy": "Balanced accuracy",
                                    "f1": "F1-score", "recall": "Recall", "precision": 'Precision'})
            df = df.set_index(
                pd.Series(
                    [str(n).capitalize().replace('Cnn', 'CNN').replace('Xgboost', 'XGBoost').
                         replace('_', ' ').replace('.txt', '').replace('forest', 'Forest').
                         replace('guess', 'Guess').replace('majority', 'Majority').replace('random', 'Random').
                         replace('voting', 'Voting')
                     for n in list(df.index)]
                )
            )
            # to latex
            phF.append(df.to_latex(index=True,
                                   bold_rows=True,
                                   caption=caption_phF,
                                   column_format='llllll',
                                   position='H'))
            # MF
            models_name = [model for model in os.listdir(score_path + '/' + path)]
            baselines_name = ['majority_voting.txt', 'random_guess.txt']
            non_baselines_name = [model_name for model_name in models_name if
                                  model_name not in baselines_name]
            n_models = len(non_baselines_name) * 2
            df = pd.DataFrame({'accuracy': [pd.NA]*n_models,
                               'balanced_accuracy': [pd.NA]*n_models,
                               'f1': [pd.NA]*n_models,
                               'recall': [pd.NA]*n_models,
                               'precision': [pd.NA]*n_models})
            df = df.set_index(pd.Series([non_baseline.replace('.txt', '') +
                                         '/' + baseline.replace('.txt', '')
                                         for non_baseline in non_baselines_name
                                         for baseline in baselines_name]))
            # calculate p-value
            for metric in list(aggregation.keys()):
                i = 0
                for non_baseline in non_baselines_name:
                    for baseline in baselines_name:
                        baseline_scores = models[baseline][1][metric]
                        model_scores = models[non_baseline][1][metric]
                        _, p_value = ttest_ind(baseline_scores, model_scores, equal_var=False)
                        df[metric].iloc[i] = round(p_value, 3)
                        i += 1
            # make value bold if p < 0.05
            for column_name in list(df.columns):
                for row in range(df.shape[0]):
                    entry = df[column_name].iloc[row]
                    if float(entry) < 0.05:
                        entry = 'BOLD{' + str(entry) + '}'
                        df[column_name].iloc[row] = entry
            # prettier names
            df = df.rename(columns={"accuracy": "Accuracy", "balanced_accuracy": "Balanced accuracy",
                                    "f1": "F1-score", "recall": "Recall", "precision": 'Precision'})
            df = df.set_index(
                pd.Series(
                    [str(n).capitalize().replace('Cnn', 'CNN').replace('Xgboost', 'XGBoost').
                         replace('_', ' ').replace('.txt', '').replace('forest', 'Forest').
                         replace('guess', 'Guess').replace('majority', 'Majority').replace('random', 'Random').
                         replace('voting', 'Voting')
                     for n in list(df.index)]
                )
            )
            # to latex
            MF.append(df.to_latex(index=True,
                                   bold_rows=True,
                                   caption=caption_MF,
                                   column_format='llllll',
                                   position='H'))
    # print
    for entry in phF:
        print(entry.replace('BOLD\\', r'\textbf').replace('\}', '}'))
    for entry in MF:
        print(entry.replace('BOLD\\', r'\textbf').replace('\}', '}'))


if __name__ == '__main__':
    scores_tables()
    print(r'\newpage')
    p_value_tables()

#%%
