import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, \
    f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, LeaveOneGroupOut, train_test_split
from tqdm import tqdm
import os


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


# TODO: check for multiclass predictions (axis=?)
def daily_majority_vote(y_pred_segments, days):
    """Predicts majority class from segement predictions"""
    y_pred_days = {day: [] for day in days}

    # aggregate
    for day, y_pred_segment in zip(days, y_pred_segments):
        y_pred_days[day].append(y_pred_segment)

    # majority vote
    for day in days:
        predictions = y_pred_days[day]
        counts_per_prediction = np.bincount(predictions)
        y_pred_days[day] = np.argmax(counts_per_prediction)

    return y_pred_days


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
            model.fit(train_indices)

            # predict
            y_pred = model.predict(test_indices)

            # evaluate
            scores = evaluator(y_pred, y_test, verbose=False)
            scores_cv.append(scores)

            # for progress bar
            pbar.update(1)
            pbar.set_description(f' Fold {i+1} F1: {scores["f1"]}')

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
            model.fit(train_indices)

            # predict
            y_pred = model.predict(test_indices)

            # evaluate
            scores = evaluator(y_pred, y_test, verbose=False)
            scores_cv.append(scores)

            # for progress bar
            pbar.update(1)
            pbar.set_description(f' Fold {i+1} F1: {scores["f1"]}')

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
            model.fit(train_indices)

            # predict
            y_pred = model.predict(test_indices)

            # evaluate
            scores = evaluator(y_pred, y_test, verbose=False)
            scores_cv.append(scores)

            # for progress bar
            pbar.update(1)
            pbar.set_description(f' Fold {i+1} F1: {scores["f1"]}')

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
    train_indices, test_indices, y_train, y_test = train_test_split(X=data_indices, y=y, test_size=test_size, random_state=SEED)

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
        for score, metric in scores.items():
            print(f' {metric}: {round(score, 3)} +- {round(score, 3)} \n')

    return scores
