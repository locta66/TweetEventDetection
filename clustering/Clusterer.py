import numpy as np
from WordFreqCounter import WordFreqCounter
from EventClassifier import EventClassifier
import TweetKeys


class Clusterer:
    def __init__(self, dict_file=None, model_file=None):
        self.freqcounter = self.classifier = None
        if dict_file:
            self.load_worddict(dict_file)
            if model_file:
                self.load_classifier_model(model_file)
                self.validate_model_dimension()
    
    def load_worddict(self, dict_file):
        self.freqcounter = WordFreqCounter()
        self.freqcounter.load_worddict(dict_file)
    
    def load_classifier_model(self, model_file):
        self.classifier = EventClassifier(vocab_size=self.freqcounter.s(), learning_rate=0)
        self.classifier.save_params(model_file)
    
    def validate_model_dimension(self):
        try:
            self.classifier.predict(np.random.random([1, self.freqcounter.s()]))
        except:
            print('Model dimension does not match that of the dict.')
    
    def make_classification(self, twarr):
        idfmtx = [self.freqcounter.idf_vector_of_wordlabels(tw[TweetKeys.key_wordlabels]) for tw in twarr]
        predictions = self.classifier.predict(idfmtx=idfmtx)
        return [twarr[idx] for idx, prediction in enumerate(predictions) if prediction > 0.5]
