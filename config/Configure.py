import os
from configparser import ConfigParser, ExtendedInterpolation


class Configure:
    def __init__(self):
        path_module_name = 'path'
        file_module_name = 'file'
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
        self.ark_service_command = configreader.get(path_module_name, 'ark_service_command')
        
        self.dc_test = configreader.get(path_module_name, 'dc_test')
        self.test_data_path = configreader.get(path_module_name, 'test_data_path')
        
        # file
        self.pos_data_file = configreader.get(file_module_name, 'pos_data_file')
        self.non_pos_data_file = configreader.get(file_module_name, 'non_pos_data_file')
        
        self.pre_dict_file = configreader.get(file_module_name, 'pre_dict_file')
        self.post_dict_file = configreader.get(file_module_name, 'post_dict_file')
        
        self.pre_prop_file = configreader.get(file_module_name, 'pre_prop_file')
        self.pre_comm_file = configreader.get(file_module_name, 'pre_comm_file')
        self.pre_verb_file = configreader.get(file_module_name, 'pre_verb_file')
        self.pre_hstg_file = configreader.get(file_module_name, 'pre_hstg_file')
        
        self.post_prop_file = configreader.get(file_module_name, 'post_prop_file')
        self.post_comm_file = configreader.get(file_module_name, 'post_comm_file')
        self.post_verb_file = configreader.get(file_module_name, 'post_verb_file')
        self.post_hstg_file = configreader.get(file_module_name, 'post_hstg_file')


_conf = Configure()


def getcfg():
    return _conf


if __name__ == "__main__":
    print(getcfg().__dict__)
