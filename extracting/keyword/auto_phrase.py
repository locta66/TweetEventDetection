import os
import errno
from subprocess import Popen, PIPE

import utils.function_utils as fu
import utils.tweet_keys as tk
import utils.timer_utils as tmu


autophrase_path = "/home/nfs/cdong/tw/src/tools/AutoPhrase"
model_base = "/home/nfs/cdong/tw/src/models/autophrase"
command = os.path.join(autophrase_path, "auto_phrase.sh")


def autophrase(raw, model):
    try:
        p = Popen(command, shell=True, bufsize=1, stdin=PIPE, stdout=PIPE, cwd=autophrase_path)
        p.stdin.flush()
        p.stdin.write((raw + '\n').encode("utf8"))
        p.stdin.write((model + '\n').encode("utf8"))
        p.communicate()
    except Exception as e:
        print('AutoPhrase Error', e)


def get_quality_phrase(twarr, raw_train="raw_train.txt", model_name="default"):
    quality_list = list()
    if len(twarr) == 0:
        return quality_list
    
    raw_train = os.path.join(model_base, raw_train)
    model_path = os.path.join(model_base, model_name)
    fu.write_lines(raw_train, [tw[tk.key_text] for tw in twarr])
    autophrase(raw_train, model_path)
    try:
        output_file = os.path.join(model_path, "AutoPhrase.txt")
        with open(output_file) as fp:
            lines = fp.readlines()
        for line in lines:
            confidence, phrase = line.strip().split(maxsplit=1)
            if float(confidence) > 0.5:
                quality_list.append(phrase)
        return quality_list
    except Exception as e:
        print("get_quality_phrase Error", e)
        return quality_list


if __name__ == "__main__":
    # file = "/home/nfs/cdong/tw/seeding/Terrorist/queried/event_corpus/2016-01-01_shoot_Aviv.txt"
    file = "/home/nfs/cdong/tw/seeding/Terrorist/queried/event_corpus/2016-07-25_kill_Mogadishu.txt"
    twarr = fu.load_array(file)
    tmu.check_time()
    print(get_quality_phrase(twarr))
    tmu.check_time()

