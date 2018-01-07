# -*- coding: utf-8 -*-
"""Simple Python wrapper for runTagger.sh script for CMU's Tweet Tokeniser and Part of Speech tagger:
http://www.ark.cs.cmu.edu/TweetNLP/
Usage:
results=runtagger_parse(['example tweet 1', 'example tweet 2'])
results will contain a list of lists (one per tweet) of triples, each triple represents (term, type, confidence)
"""
import __init__
import re
import shlex
import subprocess

import PatternUtils as pu
import TweetKeys as tk
import TweetUtils as tu
from Configure import getconfig


# The only relavent source I've found is here:
# http://m1ked.com/post/12304626776/pos-tagger-for-twitter-successfully-implemented-in
# which is a very simple implementation, my implementation is a bit more useful (but not much).

# NOTE this command is directly lifted from runTagger.sh
# RUN_TAGGER_CMD = "java -XX:ParallelGCThreads=2 -Xmx500m -jar {}".format(getconfig().ark_service_command)
RUN_TAGGER_CMD = getconfig().ark_service_command


def _split_results(wordtags):
    """ Parse the tab-delimited returned lines, modified from:
        https://github.com/brendano/ark-tweet-nlp/blob/master/scripts/show.py
        :param wordtags: corresponds to a set of word and their tags (String)in a tweet. """
    word_tag_arr = list()
    for wordtag in wordtags:
        wordtag = wordtag.strip()  # remove '\n'
        if len(wordtag) > 0:
            parts = wordtag.split('\t')
            tokens, tags, confidence = parts[0], parts[1], float(parts[2])
            word_tag_arr.append((tokens, tags, confidence))   # yield generates a result on getting a request
    return word_tag_arr


def _call_runtagger(textarr, run_tagger_cmd=RUN_TAGGER_CMD):
    """Call runTagger.sh using a named input file"""
    # remove carriage returns as they are tweet separators for the stdin interface
    tweets_cleaned = [text.replace('\n', ' ') for text in textarr]
    message = '\n'.join(tweets_cleaned)

    # force UTF-8 encoding (from internal unicode type) to avoid .communicate encoding error as per:
    # http://stackoverflow.com/questions/3040101/python-encoding-for-pipe-communicate
    message = message.encode('utf-8')

    # build a list of args
    args = shlex.split(run_tagger_cmd)
    args.extend(['--output-format', 'conll', ])
    po = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # old call - made a direct call to runTagger.sh (which is not Windows friendly)
    # po = subprocess.Popen([run_tagger_cmd, '--output-format', 'conll'],
    # stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    result = po.communicate(message)
    # expect a tuple of 2 items like:
    # ('hello\t!\t0.9858\nthere\tR\t0.4168\n\n',
    # 'Listening on stdin for input.  (-h for help)\nDetected text input format\nTokenized and tagged 1 tweets (2 tokens) in 7.5 seconds: 0.1 tweets/sec, 0.3 tokens/sec\n')
    
    """ The first line of the result contains all the sentences, and can be separated by '\n\n' from other
        execution information; consistently, the pos result is joined by '\n\n' either. """
    pos_result = result[0].decode('utf8').strip('\n\n')  # get first line, remove final double carriage return
    pos_result = pos_result.replace('\n' * 3, '\n' * 4)
    pos_result = pos_result.split('\n\n')                # split messages by double carriage returns
    pos_results = [pr.split('\n') for pr in pos_result]  # split parts of message by each carriage return
    """ [['word1\ttag1\tconf1', 'word2\ttag2\tconf2', ...],     # sentence 1
        [...],       # sentence 2
        ...]
    """
    return pos_results


def runtagger_parse(textarr, run_tagger_cmd=RUN_TAGGER_CMD):
    """Call runTagger.sh on a list of tweets, parse the result, return lists of tuples of (term, type, confidence)"""
    pos_raw_results = _call_runtagger(textarr, run_tagger_cmd)
    posarr = list()
    for sentence_pos in pos_raw_results:
        posarr.append(_split_results(sentence_pos))
    return posarr


def check_script_is_present(run_tagger_cmd=RUN_TAGGER_CMD):
    """Simple test to make sure we can see the script"""
    success = False
    try:
        args = shlex.split(run_tagger_cmd)
        args.append("--help")
        po = subprocess.Popen(args, stdout=subprocess.PIPE)
        # old call - made a direct call to runTagger.sh (not Windows friendly)
        # po = subprocess.Popen([run_tagger_cmd, '--help'], stdout=subprocess.PIPE)
        lines = list()
        while not po.poll():
            lines.extend([l.decode('utf8') for l in po.stdout])
        # we expected the first line of --help to look like the following:
        assert "RunTagger [options]" in lines[0]
        success = True
    except OSError as err:
        print("Caught an OSError, have you specified the correct path to runTagger.sh? "
              "We are using \"%s\". Exception: %r" % (run_tagger_cmd, repr(err)))
    return success


def twarr_ark(twarr, from_field=tk.key_text, to_field=tk.key_ark):
    textarr = [tw[from_field].strip() for tw in twarr]
    empty_idxes = set([idx for idx in range(len(textarr)) if pu.is_empty_string(textarr[idx])])
    posarr = runtagger_parse([textarr[idx] for idx in range(len(textarr)) if idx not in empty_idxes])
    for idx in range(len(twarr)):
        twarr[idx][to_field] = [] if idx in empty_idxes else posarr[idx]
    if len(textarr) != len(posarr) + len(empty_idxes):
        raise ValueError('len(textarr):{},len(posarr):{},len(empty_idxes):{}. Error occur during pos'.
                         format(len(textarr), len(posarr), len(empty_idxes)))
    return twarr


tags_proper_noun = {'^', 'M', 'Z', }
tags_common_noun = {'N', }
tags_verb = {'V', 'T', }
tags_hashtag = {'#', }

proper_noun_label, common_noun_label, verb_label, hashtag_label = 'proper', 'common', 'verb', 'hashtag'
label_dict = dict([(t, proper_noun_label) for t in tags_proper_noun] +
                  [(t, common_noun_label) for t in tags_common_noun] +
                  [(t, verb_label) for t in tags_verb] +
                  [(t, hashtag_label) for t in tags_hashtag])


# pos_token resembles ('word', 'pos tag', 0.91)
def is_proper_noun(pos_token): return pos_token[1] in tags_proper_noun and not is_hashtag(pos_token)
def is_common_noun(pos_token): return pos_token[1] in tags_common_noun and not is_hashtag(pos_token)
def is_verb(pos_token): return pos_token[1] in tags_verb and not is_hashtag(pos_token)
def is_hashtag(pos_token): return pos_token[0].startswith('#') and pu.has_azAZ(pos_token[0])

def pos_token2label(pos_token):
    if is_proper_noun(pos_token): return proper_noun_label
    elif is_common_noun(pos_token): return common_noun_label
    elif is_verb(pos_token): return verb_label
    elif is_hashtag(pos_token): return hashtag_label
    else: return None


if __name__ == "__main__":
    print("Checking that we can see \"{}\", this will crash if we can't." .format(RUN_TAGGER_CMD))
    success = check_script_is_present()
    if success:
        print("Success.")
        print("Now pass in two messages, get a list of tuples back:")
        tweets = ['this is a message', 'and a second message']
        print(runtagger_parse(tweets))
    # runtagger_parse(['I am DC', 'Who are you.', ''])
