import utils.array_utils as au
import utils.file_iterator as fi
import utils.function_utils as fu
import utils.multiprocess_utils as mu
import utils.pattern_utils as pu
import utils.tweet_keys as tk
import preprocess.tweet_filter as tflt
import classifying.fast_text_utils as ftu


def prefix_textarr_with_label(label, textarr):
    label_text_arr = list()
    for text in textarr:
        if pu.is_empty_string(text):
            continue
        label_text_arr.append('{} {}'.format(label, text))
    return label_text_arr


def dump_textarr(file, textarr, mode='w'):
    textarr = [text + '\n' for text in textarr]
    fu.write_lines(file, textarr, mode)
