import sys
import json
import calling.back_filter as bflt

import utils.array_utils as au
import utils.file_iterator as fi
import utils.function_utils as fu
import utils.tweet_keys as tk


def input_tweet_json_array(twarr):
    tw_batches = au.array_partition(twarr, [1] * bflt.pool_size, random=False)
    bflt.input_batches(tw_batches)


def convert_lines_to_tweet_json_array(lines):
    twarr = list()
    for line in lines:
        try:
            tw = json.loads(line, encoding='utf8')
            twarr.append(tw)
        except:
            continue
    return twarr


if __name__ == '__main__':
    # lines = sys.stdin.getlines()
    # twarr = convert_lines_to_tweet_json_array(lines)
    # input_tweet_json_array(twarr)
    base_pattern = '/home/nfs/cdong/tw/origin/{}'
    sub_files = fi.listchildren(base_pattern.format(''), fi.TYPE_FILE)[:4]
    twarr = list()
    for file in sub_files:
        twarr.extend(fu.load_array(base_pattern.format(file)))
    input_tweet_json_array(twarr)
    batches = bflt.output_batches()
    twarr = au.merge_array(batches)
    textarr = [tw.get(tk.key_text) for tw in twarr]
    for text in textarr:
        print(text)
    print(len(textarr))
