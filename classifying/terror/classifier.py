import utils.array_utils as au
import utils.file_iterator as fi
import utils.timer_utils as tmu

import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegressionCV
from sklearn.externals import joblib
from sklearn import metrics


np.random.seed(233)
model_dict = dict()
lr_model_file = '/home/nfs/cdong/tw/seeding/Terrorist/model/lr_model'
svm_model_file = '/home/nfs/cdong/tw/seeding/Terrorist/model/svm_model'


def get_model(model_key=lr_model_file):
    if model_key not in model_dict:
        model_dict[model_key] = joblib.load(model_key)
    return model_dict[model_key]


def predict_proba(x, model=None):
    """ returns a list of probas for every vector in x as being classified as event """
    if not type(x) is np.ndarray:
        x = np.array(x)
    if len(x.shape) == 1:
        raise ValueError('dimension of x should be 2')
    model = get_model() if model is None else model
    probs = model.predict_proba(x)
    return [item[1] for item in probs]


def random_combine_vectors(matrix):
    if len(matrix) <= 1:
        return matrix
    len_vecarr = len(matrix)
    idx_range = np.array([i for i in range(len_vecarr)])
    len_range = idx_range[3:]
    idx_cmb_arr = list()
    for i in range(len_vecarr):
        vec_num = np.random.choice(a=len_range, p=len_range / np.sum(len_range))
        idx_cmb = np.random.choice(idx_range, vec_num)
        idx_cmb_arr.append(idx_cmb)
    combine = np.array([np.mean(matrix[idx_cmb], axis=0) for idx_cmb in idx_cmb_arr])
    print(matrix.shape, combine.shape, '\n')
    return np.concatenate([matrix, combine], axis=0)


def load_train_test():
    """ load matrices from disk """
    base_pattern = '/home/nfs/cdong/tw/seeding/Terrorist/data/{}'
    pos_files = fi.listchildren(base_pattern.format(''), fi.TYPE_FILE, 'pos')
    pos_matrices = [np.load(base_pattern.format(file)) for file in pos_files]
    neg_files = fi.listchildren(base_pattern.format(''), fi.TYPE_FILE, 'neg')[:4]
    neg_matrices = [np.load(base_pattern.format(file)) for file in neg_files]
    neg_matrix = np.concatenate(neg_matrices * 2, axis=0)
    
    """ create the matrices and their labels """
    pos_split = 40
    # pos_train_x = np.concatenate([random_combine_vectors(matrix) for matrix in pos_matrices[:pos_split]] * 10, axis=0)
    pos_train_x = np.concatenate(pos_matrices[:pos_split] * 20, axis=0)
    pos_test_x = np.concatenate(pos_matrices[pos_split:], axis=0)
    neg_split = int(0.7 * len(neg_matrix))
    neg_train_x = neg_matrix[:neg_split]
    neg_test_x = neg_matrix[neg_split:neg_split + 150000]
    pos_train_y, pos_test_y = np.array([1 for _ in pos_train_x]), np.array([1 for _ in pos_test_x])
    neg_train_y, neg_test_y = np.array([0 for _ in neg_train_x]), np.array([0 for _ in neg_test_x])
    
    print('train  p: {:<6}, n: {}\ntest   p: {}, n: {:<6}'
          .format(len(pos_train_x), len(neg_train_x), len(pos_test_x), len(neg_test_x)))
    
    train_x, train_y = np.concatenate([pos_train_x, neg_train_x]), np.concatenate([pos_train_y, neg_train_y])
    test_x, test_y = np.concatenate([pos_test_x, neg_test_x]), np.concatenate([pos_test_y, neg_test_y])
    return train_x, train_y, test_x, test_y


def train_model(x, y, model):
    print('using model {}'.format(type(model)))
    tmu.check_time()
    model.fit(x, y)
    tmu.check_time()
    return model


def test_model(x, y, model):
    preds_0or1 = model.predict_proba(x)
    preds_score = [entry[1] for entry in model.predict_proba(x)]
    auc = au.score(y, preds_0or1, 'auc')
    print(auc)
    auc = au.score(y, preds_score, 'auc')
    print(auc)
    
    precision, recall, thresholds = metrics.precision_recall_curve(y, preds_score)
    last_idx = 0
    for ref in [i / 10 for i in range(1, 10)]:
        for idx in range(last_idx, len(thresholds)):
            if thresholds[idx] >= ref:
                print('threshold', round(thresholds[idx], 2), '\tprecision', round(precision[idx], 5),
                      '\trecall', round(recall[idx], 5))
                last_idx = idx
                break


if __name__ == '__main__':
    train_x, train_y, test_x, test_y = load_train_test()
    """ choose model """
    model, model_file = LogisticRegressionCV(), lr_model_file
    # model, model_file = SVC(kernel='linear', C=0.5, probability=True), svm_model_file
    """ train & test """
    model = train_model(train_x, train_y, model)
    joblib.dump(model, model_file)
    print('model saved to {}'.format(model_file))
    model = joblib.load(model_file)
    print('model loaded from {}'.format(model_file))
    test_model(test_x, test_y, model)
