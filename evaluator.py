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
    days_selection = [(meta['date'], meta['subjectID']) for meta in metadata_selection] # note that same date can be used by different subjects

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
