import classifying.fast_text_utils as ftu
from collections import Counter
import utils.array_utils as au


label2value = ftu.binary_label2value
terror_model_file = '/home/nfs/cdong/tw/seeding/Terrorist/model/fasttext_model'


def predict(target, threshold=0.5):
    """ returns value/value array given input text/text array; value(s) are dependent on the threshold """
    model = ftu.get_model(terror_model_file)
    pred_value_arr, score_arr = ftu.binary_predict(target, model, threshold)
    return pred_value_arr, score_arr


def train(train_file, model_file):
    model = ftu.FastText()
    model.train_supervised(input=train_file, epoch=50, lr=2, wordNgrams=2, verbose=2, minCount=10)
    ftu.save_model(model_file, model)
    return model


def test(test_file, model_file):
    textarr, labelarr = list(), list()
    with open(test_file) as testfp:
        lines = testfp.readlines()
    for line in lines:
        label, text = line.strip().split(' ', 1)
        textarr.append(text)
        labelarr.append(label)
    preds, scores = predict(textarr, threshold=0.2)
    assert len(preds) == len(textarr)
    
    for thres in [i/10 for i in range(2, 11)]:
        print(thres, Counter([1 if s > thres else 0 for s in scores]))
    
    label = [label2value[label] for label in labelarr]
    print(au.score(label, preds, 'auc'))
    for idx in range(1000, 1100):
        pred, lb, text = preds[idx], label[idx], textarr[idx]
        if not pred == lb:
            print(pred, lb, text)


if __name__ == '__main__':
    train_file = '/home/nfs/cdong/tw/seeding/Terrorist/data/fasttext/train'
    test_file = '/home/nfs/cdong/tw/seeding/Terrorist/data/fasttext/test'
    
    train(train_file, terror_model_file)
    test(test_file, terror_model_file)
