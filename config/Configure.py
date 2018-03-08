import os
from configparser import ConfigParser, ExtendedInterpolation


class Configure:
    def __init__(self):
        path_module_name = 'path'
        file_module_name = 'file'
        model_module_name = 'filter model'
        file_dir = os.path.split(os.path.realpath(__file__))[0] + os.path.sep
        config_file_name = file_dir + 'conf.ini'
        configreader = ConfigParser(interpolation=ExtendedInterpolation())
        configreader.read(config_file_name)
        
        # path
        self.data_path = configreader.get(path_module_name, 'data_path')
        self.seed_path = configreader.get(path_module_name, 'seed_path')
        self.summary_path = configreader.get(path_module_name, 'summary_path')
        self.origin_path = configreader.get(path_module_name, 'origin_path')
        
        self.ner_service_command = configreader.get(path_module_name, 'ner_service_command')
        # self.ark_service_command = configreader.get(path_module_name, 'ark_service_command')
        self.ark_service_command = '/home/nfs/cdong/tw/src/tools/ark-tweet-nlp-0.3.2.jar'
        
        # files
        # self.pos_data_file = configreader.get(file_module_name, 'pos_data_file')
        # self.non_pos_data_file = configreader.get(file_module_name, 'non_pos_data_file')
        
        self.pre_dict_file = configreader.get(file_module_name, 'pre_dict_file')
        self.post_dict_file = configreader.get(file_module_name, 'post_dict_file')
        
        # self.pre_prop_file = configreader.get(file_module_name, 'pre_prop_file')
        # self.pre_comm_file = configreader.get(file_module_name, 'pre_comm_file')
        # self.pre_verb_file = configreader.get(file_module_name, 'pre_verb_file')
        # self.pre_hstg_file = configreader.get(file_module_name, 'pre_hstg_file')

        # self.post_prop_file = configreader.get(file_module_name, 'post_prop_file')
        # self.post_comm_file = configreader.get(file_module_name, 'post_comm_file')
        # self.post_verb_file = configreader.get(file_module_name, 'post_verb_file')
        # self.post_hstg_file = configreader.get(file_module_name, 'post_hstg_file')
        
        self.ft_terror_model_file = configreader.get(file_module_name, 'ft_terror_model_file')
        
        # filter model files
        self.afinn_file = configreader.get(model_module_name, 'afinn_file')
        self.black_list_file = configreader.get(model_module_name, 'black_list_file')
        
        self.clf_model_file = configreader.get(model_module_name, 'clf_model_file')
        
        self.class_dist_file = configreader.get(model_module_name, 'class_dist_file')
        self.chat_filter_file = configreader.get(model_module_name, 'chat_filter_file')
        self.is_noise_dict_file = configreader.get(model_module_name, 'is_noise_dict_file')
        self.orgn_predict_label_file = configreader.get(model_module_name, 'orgn_predict_label_file')


_conf = Configure()


def getcfg():
    return _conf


if __name__ == "__main__":
    print(getcfg().__dict__)
