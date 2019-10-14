# coding:utf-8
import numpy as np
import pickle
import gzip
import logging
from collections import Counter
import random
import math
import json
from tqdm import tqdm
MAX_NUM = 1e09

def gen_examples(data, batch_size):
    """
        Divide examples into batches of size `batch_size`.
    """
    minibatches = get_minibatches(len(data['ids']), batch_size)
    for minibatch in minibatches:
        mb_x0 = [data['ids'][t] for t in minibatch]
        mb_x1 = [data['doc'][t] for t in minibatch]
        mb_x2 = [data['ques'][t] for t in minibatch]
        mb_y = [data['answer'][t] for t in minibatch]
        mb_c = [data['candidates'][t] for t in minibatch]
        mb_x1, mb_mask1 = prepare_data(mb_x1)
        mb_x2, mb_mask2 = prepare_data(mb_x2)
        yield {'ids':mb_x0, 'x':mb_x1, 'x_mask':mb_mask1, 'q':mb_x2, 'q_mask':mb_mask2, 'candidates':mb_c, 'y':mb_y}
#        all_ex.append((mb_x0, mb_x1, mb_mask1, mb_x2, mb_mask2, mb_c, mb_y))
#    return all_ex

def load_data(in_file, start_ratio=0, end_ratio=1.0):
    """
        load CNN / Daily Mail data from {train | dev | test}.txt
        relabeling: relabel the entities by their first occurence if it is True.
        """
    with open(in_file,'r') as fr:
        inf = json.load(fr)
    return inf


def build_dict(sentences, max_words=50000):
    """
        Build a dictionary for the words in `sentences`.
        Only the max_words ones are kept and the remaining will be mapped to <UNK>.
        """
    word_count = Counter()
    for sent in sentences:
        for w in sent.split(' '):
            word_count[w] += 1
    max_words = int(len(word_count.keys()) * 0.95)
    word_count['unk'] = MAX_NUM
    ls = word_count.most_common(max_words)
    logging.info('#Words: %d -> %d' % (len(word_count), len(ls)))
    for key in ls[:5]:
        logging.info(key)
    logging.info('...')
    for key in ls[-5:]:
        logging.info(key)
    # leave 0 to UNK
    # leave 1 to delimiter |||
    return {w[0]: index for (index, w) in enumerate(ls)}, {index:w[0] for (index, w) in enumerate(ls)}


def get_word_id(w,word_dict):
    if w in word_dict:
        return word_dict[w]
    else:
        return 0


def vectorize(examples, word_dict, args,
              sort_by_len=True, verbose=True):
    """
        Vectorize `examples`.
        in_x1, in_x2: sequences for document and question respecitvely.
        in_y: label
        """
    in_x0 = []
    in_x1 = []
    in_x2 = []
    in_y = np.zeros((len(examples[0]), 1)).astype(float)
    in_c = []
    n = 0
    for idx, (m_id, d, q, a, c) in tqdm(enumerate(zip(examples[0], examples[1], examples[2], examples[3], examples[4]))):
        d_words = d.split(' ')
        q_words = q.split(' ')
        if args.cut_q:
            q_pos = q_words.index('XXXX')
            q_words = q_words[q_pos - args.boundary:q_pos + args.boundary]
        if args.no_passages:
            seq1 = np.zeros(10)
        else:
            seq1 = [ get_word_id(w,word_dict) for w in d_words]
        seq2 = [ get_word_id(w,word_dict) for w in q_words]
        assert a != 'unk'
        if (len(seq1) > 0) and (len(seq2) > 0) and (a in word_dict) and len(c['from_pass']) > args.candidate_nums:
            in_x0.append(m_id)
            in_x1.append(seq1)
            in_x2.append(seq2)
            tmp = [word_dict[a]]
            for j in range(args.candidate_nums):
                w_id = get_word_id(c['from_pass'][j],word_dict)
                tmp.append(w_id)
            random.shuffle(tmp)
            in_y[n, 0] = tmp.index(word_dict[a])
            in_c.append(tmp)
            n += 1
        else:
            print(a)
    if verbose:
        logging.info('Vectorization: processed %d / %d' % (n, len(examples[0])))
    in_y = in_y[:n]
    assert len(in_y) == len(in_x1)

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    if sort_by_len:
        # sort by the document length
        sorted_index = len_argsort(in_x1)
        in_x0 = [in_x0[i] for i in sorted_index]
        in_x1 = [in_x1[i] for i in sorted_index]
        in_x2 = [in_x2[i] for i in sorted_index]
        in_y = [in_y[i] for i in sorted_index]
        in_c = [in_c[i] for i in sorted_index]
    return in_x0, in_x1, in_x2, in_c, in_y


def prepare_data(seqs):
    lengths = [len(seq) for seq in seqs]
    n_samples = len(seqs)
    max_len = np.max(lengths)
    x = np.zeros((n_samples, max_len)).astype('int32')
    x_mask = np.zeros((n_samples, max_len)).astype(float)
    for idx, seq in enumerate(seqs):
        x[idx, :lengths[idx]] = seq
        x_mask[idx, :lengths[idx]] = 1.0
    return x, x_mask


def get_minibatches(n, minibatch_size, shuffle=False):
    idx_list = np.arange(0, n, minibatch_size)
    if shuffle:
        np.random.shuffle(idx_list)
    minibatches = []
    import pdb
    for idx in idx_list:
        tmp = np.arange(idx,min(idx+minibatch_size,n))
        if len(tmp)< minibatch_size:
            tmp1 = list(tmp)
            tmp2 = list(np.arange(0,minibatch_size-len(tmp)))
            tmp = np.array(tmp1 + tmp2)
        minibatches.append(tmp)
    return minibatches


def get_dim(in_file):
    line = open(in_file).readline()
    return len(line.split()) - 1


def gen_embeddings(word_dict, dim, in_file=None):
    """
        Generate an initial embedding matrix for `word_dict`.
        If an embedding file is not given or a word is not in the embedding file,
        a randomly initialized vector will be used.
        """
    #import pdb
    #pdb.set_trace()
    num_words = len(word_dict.keys())
    embeddings = np.random.rand(num_words, dim)
    logging.info('Embeddings: %d x %d' % (num_words, dim))

    if in_file is not None:
        logging.info('Loading embedding file: %s' % in_file)
        pre_trained = 0
        for line in open(in_file).readlines():
            sp = line.split()
            if len(sp) == 2 and int(sp[1]) == dim:
                continue
            assert len(sp) == dim + 1
            if sp[0] in word_dict:
                pre_trained += 1
                assert word_dict[sp[0]] >= 0
                embeddings[word_dict[sp[0]]] = [float(x) for x in sp[1:]]
        logging.info('Pre-trained: %d (%.2f%%)' %
                     (pre_trained, pre_trained * 100.0 / num_words))
    return embeddings