import os
from configparser import ConfigParser, ExtendedInterpolation


class MyConfig:
    def __init__(self):
        module_name = 'config'
        file_module_name = 'file'
        file_dir = os.path.split(os.path.realpath(__file__))[0] + os.path.sep
        config_file_name = file_dir + 'conf.ini'
        configreader = ConfigParser(interpolation=ExtendedInterpolation())
        configreader.read(config_file_name)
        
        self.data_path = configreader.get(module_name, 'data_path')
        self.seed_path = configreader.get(module_name, 'seed_path')
        self.summary_path = configreader.get(module_name, 'summary_path')
        self.ner_service_command = configreader.get(module_name, 'ner_service_command')
        
        self.test_data_path = configreader.get(module_name, 'test_data_path')
        self.pos_data_file = configreader.get(file_module_name, 'pos_data_file')
        self.non_pos_data_file = configreader.get(file_module_name, 'non_pos_data_file')


conf = MyConfig()


def getconfig():
    return conf


if __name__ == "__main__":
    print(getconfig().__dict__)
