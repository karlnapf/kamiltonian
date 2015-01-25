import numpy as np
from sklearn.cross_validation import KFold

def xvalidate_objective(Z, n_folds, estimator, objective, num_repetitions=1):
    kf = KFold(len(Z), n_folds=n_folds)
    
    Js = np.zeros((num_repetitions, n_folds))
    for i in range(num_repetitions):
        for j, (train, test) in enumerate(kf):
            a = estimator(Z[train])
            Js[i, j] = objective(Z[test], a)
    
    return np.mean(Js)
