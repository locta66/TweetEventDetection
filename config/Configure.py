import os
from configparser import ConfigParser, ExtendedInterpolation


class Configure:
    def __init__(self):
        path_module = 'path'
        file_module = 'file'
        model_module = 'filter model'
        # file_dir = os.path.abspath(os.path.dirname(__file__))
        file_dir = os.path.split(os.path.realpath(__file__))[0]
        config_file_name = os.path.join(file_dir, 'conf.ini')
        configreader = ConfigParser(interpolation=ExtendedInterpolation())
        configreader.read(config_file_name)
        
        # path
        self.data_path = configreader.get(path_module, 'data_path')
        self.seed_path = configreader.get(path_module, 'seed_path')
        self.summary_path = configreader.get(path_module, 'summary_path')
        self.origin_path = configreader.get(path_module, 'origin_path')
        
        self.ner_service_command = configreader.get(path_module, 'ner_service_command')
        self.ark_service_command = configreader.get(path_module, 'ark_service_command')
        self.ark_service_command = '/home/nfs/cdong/tw/src/tools/ark-tweet-nlp-0.3.2.jar'
        
        # files
        self.pre_dict_file = configreader.get(file_module, 'pre_dict_file')
        self.post_dict_file = configreader.get(file_module, 'post_dict_file')
        self.ft_terror_model_file = configreader.get(file_module, 'ft_terror_model_file')
        
        # files
        self.afinn_file = configreader.get(model_module, 'afinn_file')
        self.black_list_file = configreader.get(model_module, 'black_list_file')
        
        self.clf_model_file = configreader.get(model_module, 'clf_model_file')
        
        self.class_dist_file = configreader.get(model_module, 'class_dist_file')
        self.chat_filter_file = configreader.get(model_module, 'chat_filter_file')
        self.is_noise_dict_file = configreader.get(model_module, 'is_noise_dict_file')
        self.orgn_predict_label_file = configreader.get(model_module, 'orgn_predict_label_file')
        
        self.ft_add_model_file = configreader.get(model_module, 'ft_add_model_file')
        self.lr_add_model_file = configreader.get(model_module, 'lr_add_model_file')


_conf = Configure()


def getcfg():
    return _conf


if __name__ == "__main__":
    print(getcfg().__dict__)
