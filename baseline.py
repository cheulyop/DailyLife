import os
import math
import json
import logging
import argparse
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb

from tqdm import tqdm
from collections import OrderedDict
from numpy.random import default_rng
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

from pyteap.signals.gsr import acquire_gsr, get_gsr_features
from pyteap.signals.ecg import get_ecg_features
from utils.logging import init_logger


def load_dataset(paths):
    # load esm data
    esms = pd.read_csv(filepath_or_buffer=paths['esms'], header=0)

    # for each user
    uid_to_segments = {}
    for uid in os.listdir(paths['root']):
        uid_to_segments.setdefault(int(uid), [])
        segs_dir = os.path.join(paths['root'], uid)
        
        # for each segment file in segs_dir
        for fname in os.listdir(segs_dir):
            # get index and labels
            idx = int(fname.split('.')[0])
            esm = esms.loc[idx]
            labels = (esm.arousal, esm.valence)

            # load segment saved as json file and save to dict
            with open(os.path.join(segs_dir, fname)) as f:
                seg = json.load(f)
                uid_to_segments[int(uid)].append((idx, seg, labels))

    # return dict ordered by uid
    return OrderedDict(sorted(uid_to_segments.items(), key=lambda x: x[0]))


def prepare_dataset(paths):
    # load segments
    uid_to_segments = load_dataset(paths)

    # prepare features and labels
    X, y = {}, {}

    def rmssd(rri):
        rri = rri[~np.isnan(rri)]
        diff = [(rri[i] - rri[i+1]) ** 2 for i in range(len(rri) - 1)]

        return math.sqrt(sum(diff) / len(diff))

    # for each user
    for uid, segs in uid_to_segments.items():
        # sort segments by index
        segs = sorted(segs, key=lambda x: x[0])

        curr_X, curr_y = [], []
        # with each segment
        for (_, seg, labels) in tqdm(segs, desc=f'User {uid}', ascii=True, dynamic_ncols=True):

            # get features
            features = []
            for sigtype in ['gsr', 'bpm', 'rri', 'temp']:
                sig = seg[sigtype]
                if sigtype == 'gsr':
                    # divide by 1e3 as raw gsr is in kOhms for msband 2
                    features.extend(get_gsr_features(acquire_gsr(np.array(sig) / 1e3, 5), 5))
                elif sigtype == 'bpm':
                    features.extend(get_ecg_features(sig))
                elif sigtype == 'rri':
                    features.extend([np.mean(sig), np.std(sig, ddof=1), rmssd(np.array(sig))])
                elif sigtype == 'temp':
                    features.extend([np.mean(sig), np.std(sig, ddof=1)])

            # skip if one or more feature is NaN
            if np.isnan(features).any():
                logging.getLogger('default').warning('One or more feature is NaN, skipped.')
                continue

            curr_X.append(features)
            curr_y.append([labels[0] >= 0, labels[1] >= 0])

        X[uid] = StandardScaler().fit_transform(np.stack(curr_X))
        y[uid] = np.stack(curr_y)

    return X, y


def get_results(y_test, preds, probs):
    return {
        'acc.': accuracy_score(y_test, preds),
        'bacc.': balanced_accuracy_score(y_test, preds, adjusted=False),
        'f1': f1_score(y_test, preds),
        'auroc': roc_auc_score(y_test, probs),
    }


def pred_majority(majority, y_test):
    preds = np.repeat(majority, y_test.size)
    probs = np.repeat(majority, y_test.size)
    return get_results(y_test, preds, probs)


def pred_random(y_classes, y_test, rng, ratios=None):
    preds = rng.choice(y_classes, y_test.size, replace=True, p=ratios)
    if ratios is not None:
        probs = np.where(preds == 1, ratios[1], ratios[0])
    else:
        probs = np.repeat(0.5, y_test.size)
    return get_results(y_test, preds, probs)


def pred_gnb(X_train, y_train, X_test, y_test):
    clf = GaussianNB().fit(X_train, y_train)
    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:, 1]
    return get_results(y_test, preds, probs)


def pred_xgb(X_train, y_train, X_test, y_test, seed):
    # load data into DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # set parameters
    params = {
        'booster': 'gbtree',
        'verbosity': 1,
        'max_depth': 6,
        'eta': 0.3,
        'objective': 'binary:logistic',
        # 'num_class': 2,
        'eval_metric': 'auc',
        'seed': seed,
    }

    # train model and predict
    num_round = 100
    bst = xgb.train(params, dtrain, num_round)
    probs = bst.predict(dtest)
    preds = probs > 0.5
    
    # return results
    return get_results(y_test, preds, probs)


def get_baseline_kfold(X, y, seed, target, n_splits, shuffle):
    # initialize random number generator and fold generator
    rng = default_rng(seed)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)

    # aggregated features and labels
    X = np.concatenate(list(X.values()))
    y = np.concatenate(list(y.values()))
    logging.getLogger('default').info(f'Dataset size: {X.shape}')

    # get labels corresponding to target class
    if target == 'arousal':
        y = y[:, 0]
    elif target == 'valence':
        y = y[:, 1]

    results = {}
    # for each fold, split train & test and get classification results
    for i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        y_classes, y_counts = np.unique(y_train, return_counts=True)
        majority = y_classes[np.argmax(y_counts)]
        class_ratios = y_counts / y_train.size

        results[i+1] = {
            'Random': pred_random(y_classes, y_test, rng),
            'Majority': pred_majority(majority, y_test),
            'Class ratio': pred_random(y_classes, y_test, rng, ratios=class_ratios),
            'Gaussian NB': pred_gnb(X_train, y_train, X_test, y_test),
            'XGBoost': pred_xgb(X_train, y_train, X_test, y_test, seed),
        }

    # return results as table
    results = {(fold, classifier): values for (fold, _results) in results.items() for (classifier, values) in _results.items()}
    results_table = pd.DataFrame.from_dict(results, orient='index').stack().unstack(level=1).rename_axis(['Fold', 'Metric'])
    return results_table[['Random', 'Majority', 'Class ratio', 'Gaussian NB', 'XGBoost']]


def get_baseline_loso(X, y, seed, target, n_splits, shuffle):
    # initialize random number generator
    rng = default_rng(seed)

    results = {}
    # for each participant split train & test
    for uid in X.keys():
        X_train, X_test = np.concatenate([v for k, v in X.items() if k != uid]), X[uid]
        y_train, y_test = np.concatenate([v for k, v in y.items() if k != uid]), y[uid]

        # get labels corresponding to target class
        if target == 'arousal':
            y_train, y_test = y_train[:, 0], y_test[:, 0]
        elif target == 'valence':
            y_train, y_test = y_train[:, 1], y_test[:, 1]

        # get majority label and class ratios
        y_classes, y_counts = np.unique(y_train, return_counts=True)
        majority = y_classes[np.argmax(y_counts)]
        class_ratios = y_counts / y_train.size

        # get classification results
        results[uid] = {
            'Random': pred_random(y_classes, y_test, rng),
            'Majority': pred_majority(majority, y_test),
            'Class ratio': pred_random(y_classes, y_test, rng, ratios=class_ratios),
            'Gaussian NB': pred_gnb(X_train, y_train, X_test, y_test),
            'XGBoost': pred_xgb(X_train, y_train, X_test, y_test, seed),
        }

    results = {(uid, classifier): value for (uid, _results) in results.items() for (classifier, value) in _results.items()}
    results_table = pd.DataFrame.from_dict(results, orient='index').stack().unstack(level=1)
    return results_table[['Random', 'Majority', 'Class ratio', 'Gaussian NB', 'XGBoost']]


def get_baseline(X, y, configs):
    seed = configs['seed']
    target = configs['target']
    cv = configs['cv']
    n_splits = configs['splits']
    shuffle = configs['shuffle']

    if cv == 'kfold':
        results = get_baseline_kfold(X, y, seed, target, n_splits, shuffle)
    elif cv == 'loso':
        results = get_baseline_loso(X, y, seed, target, n_splits, shuffle)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', '-r', type=str, required=True)
    parser.add_argument('--esms', '-e', type=str, required=True)
    parser.add_argument('--timezone', '-tz', type=str, default='UTC', help='a pytz timezone string for logger, default is UTC')
    parser.add_argument('--seed', '-s', type=int, default=0, help='seed for random number generation')
    parser.add_argument('--target', '-t', type=str, default='valence', help='target label for classification, must be either "valence" or "arousal"')
    parser.add_argument('--cv', type=str, default='kfold', help='type of cross-validation to perform, must be either "kfold" or "loso" (leave-one-subject-out)')
    parser.add_argument('--splits', type=int, default=5, help='number of folds for k-fold stratified classification')
    parser.add_argument('--shuffle', default=False, action='store_true', help='shuffle data before splitting to folds, default is no shuffle')
    args = parser.parse_args()

    # initialize default logger and path variables
    logger = init_logger(tz=args.timezone)
    PATHS = {
        'root': os.path.expanduser(args.root),
        'esms': os.path.expanduser(args.esms)
    }

    # filter these RuntimeWarning messages
    warnings.filterwarnings('ignore')
    # warnings.filterwarnings(action='ignore', message='Mean of empty slice')
    # warnings.filterwarnings(action='ignore', message='invalid value encountered in double_scalars')
    # warnings.filterwarnings(action='ignore', message='divide by zero encountered in true_divide')
    # warnings.filterwarnings(action='ignore', message='invalid value encountered in subtract')

    # check commandline arguments
    assert args.target in ['valence', 'arousal'], f'--target must be either "valence" or "arousal", but given {args.target}'
    assert args.cv in ['kfold', 'loso'], f'--cv must be either "kfold" or "loso", but given {args.cv}'
    assert args.splits > 1, f'--splits must be greater than 1, but given {args.splits}'

    logger.info('Preprocessing data with...')
    logger.info(f"Dataset: {PATHS['root']}")
    logger.info(f"ESM: {PATHS['esms']}")
    X, y = prepare_dataset(PATHS)
    logger.info('Preprocessing complete.')

    CONFIGS = {
        'seed': args.seed,
        'target': args.target,
        'cv': args.cv,
        'splits': args.splits,
        'shuffle': args.shuffle,
    }

    logger.info(f'Config: {CONFIGS}')
    results = get_baseline(X, y, CONFIGS)
    # print summary of classification results
    if args.cv == 'kfold':
        print(results.groupby(level='Metric').mean())
    else:
        print(results)
