import os
import subprocess
def auto_phrase(autophrase_path,my_data_path):
    my_data_file_name = os.path.split(my_data_path)[1]
    model_name = os.path.splitext(my_data_file_name)[0]
    my_data_dir = os.path.join(autophrase_path,'data')
    model_dir = os.path.join(autophrase_path,'models')
    os.system('cp ' + my_data_path + ' ' + my_data_dir)
    working_dir = os.path.abspath('.')
    os.chdir(autophrase_path)
    subprocess.call('./auto_phrase.sh ' + my_data_file_name + ' ' + model_name,shell=True)
    get_phrase(model_dir,model_name) # 可以注释掉这块
    os.chdir(working_dir)

def get_phrase(model_dir,model_name):
    try:
        phrase = os.path.join(model_dir,model_name)
        os.chdir(phrase)
        with open('AutoPhrase.txt') as f:
            for line in f:
                if float(line.strip().split()[0]) > 0.5:
                    print (line)
        return phrase
    except:
        print ('no such model!')


if __name__ =='__main__':
    '''输入一个AutoPhrase 安装路径和文件路径,输出阈值大于0.5的key phrase，并且返回提取关键词后的文件路径'''
    autophrase_path = '/home/nfs/yangl/event_detection/src/AutoPhrase/AutoPhrase'
    my_data_path = r'/home/yangl/2016-1-10_explode_Baghdad.txt'
    auto_phrase(autophrase_path,my_data_path)

