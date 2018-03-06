import spacy
import numpy as np


"""
# doc = su.text_nlp(get_text(tw), nlp)
# [(token.text, token.ent_type_, token.tag_, token.pos_, ) for token in doc]
# [(ent.text, ent.root.tag_, ent.label_) for ent in doc.ents]
"""


default_model_name = 'en_core_web_lg'
nlp_dict = {}
glovec_dim = 300


def new_nlp(model):
    try:
        return spacy.load(model)
    except:
        print('load model error')
        return None


def get_nlp(model_key=default_model_name):
    if model_key not in nlp_dict:
        nlp_dict.setdefault(model_key, new_nlp(model_key))
    return nlp_dict.get(model_key)


def text_nlp(text, nlp=None):
    nlp = get_nlp() if nlp is None else nlp
    return nlp(text)


def textarr_nlp(textarr, nlp=None, n_threads=4):
    nlp = get_nlp() if nlp is None else nlp
    return [doc for doc in nlp.pipe(textarr, n_threads=n_threads)]


pos_prop = 'PROPN'
pos_comm = 'NOUN'
pos_verb = 'VERB'
pos_hstg = 'HSTG'
# key_ent = 'ENT'
# ent_glf = {'FAC', 'GPE', 'LOC'}
target_pos_types = [pos_hstg, pos_prop, pos_comm, pos_hstg]
target_pos_sets = set(target_pos_types)


def get_doc_pos_vectors(doc):
    """ returns embedding vector featuring the doc using above types of pos tags """
    postype2itemarr = dict([(pos_type, list()) for pos_type in target_pos_types])
    for token in doc:
        if not (token.has_vector and token.pos_ in postype2itemarr):
            continue
        postype2itemarr[token.pos_].append([token.tag_, token.vector])
    vecarr = list()
    for pos_type in target_pos_types:
        itemarr = postype2itemarr[pos_type]
        if len(itemarr) == 0:
            pos_vector = np.zeros([glovec_dim, ])
        else:
            pos_vector = np.mean([item[1] for item in itemarr], axis=0)
        assert len(pos_vector) == glovec_dim
        vecarr.append(pos_vector)
    return vecarr


if __name__ == '__main__':
    s = 'You shall not pass.'
    doc = text_nlp(s)
    print([t.pos_ for t in doc])
