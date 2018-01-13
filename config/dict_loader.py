import utils.pattern_utils as pu
from utils.id_freq_dict import IdFreqDict
from config.configure import getcfg


pre_list = [getcfg().pre_prop_file, getcfg().pre_comm_file, getcfg().pre_verb_file, getcfg().pre_hstg_file]
post_list = [getcfg().post_prop_file, getcfg().post_comm_file, getcfg().post_verb_file, getcfg().post_hstg_file]
prop_dict, comm_dict, verb_dict, hstg_dict = dict_list = [IdFreqDict() for _ in post_list]


if __name__ == '__main__':
    def word_remove(word, freq):
        if pu.search_pattern(r'!?<>.,&\'`\^*', word) is not None or freq < 10:
            return True
        return False
    
    pre2post = dict(zip(pre_list, post_list))
    for pre, post in pre2post.items():
        ifd = IdFreqDict()
        ifd.load_dict(pre)
        pre_vocab = ifd.vocabulary_size()
        print('{} loaded, {} words'.format(pre, pre_vocab))
        ifd.drop_words_by_condition(word_remove)
        print('{} words dropped, remain {} words'.format(pre_vocab - ifd.vocabulary_size(), ifd.vocabulary_size()))
        ifd.dump_dict(post)
        print('dump over')


for idx, ifd in enumerate(dict_list):
    ifd.load_dict(post_list[idx])
    print(idx)
