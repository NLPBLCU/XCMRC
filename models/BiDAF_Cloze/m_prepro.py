import argparse
import json
import os
# data: q, cq, (dq), (pq), y, *x, *cx
# shared: x, cx, (dx), (px), word_counter, char_counter, word2vec
# no metadata
from collections import Counter
import pdb
from tqdm import tqdm
import re
import logging
import numpy as np
import random
MAX_NUM=1e09

def main():
    import pdb
    #pdb.set_trace()
    args = get_args()
    source_train_f = os.path.join(args.source_dir,args.version,'raw','train.json')
    source_dev_f = os.path.join(args.source_dir,args.version,'raw','dev.json')
    train_data = prepro_each(source_train_f)
    dev_data = prepro_each(source_dev_f)
    word_2_id_dict, id_2_word_dict, pretrained_emb = build_dict(args, train_data[1] + train_data[2] + dev_data[1] + dev_data[2],emb_f=args.embedding_f)
    import pdb
    #pdb.set_trace()
    train_data = vectorize(examples=train_data,word_dict=word_2_id_dict,args=args)
    dev_data = vectorize(examples=dev_data,word_dict=word_2_id_dict,args=args)
    target_train_f = os.path.join(args.source_dir,args.version,'featurized','train.json')
    import pdb
    #pdb.set_trace()
    train_data['pretrained_emb'] = pretrained_emb
    with open(target_train_f,'w') as fw:
        fw.write(json.dumps(train_data,ensure_ascii=False))
    target_dev_f = os.path.join(args.source_dir,args.version,'featurized','dev.json')
    with open(target_dev_f, 'w') as fw:
        fw.write(json.dumps(dev_data,ensure_ascii=False))
    with open(os.path.join(args.source_dir, args.version,'featurized','vocab.file'),'w') as fw:
        fw.write('\n'.join(word_2_id_dict.keys()))

def get_args():
    parser = argparse.ArgumentParser()
    source_dir = '/home/yndeng/MRC/data/CrossTest'
    version = 'pure_ch'
    glove_dir = os.path.join("data", "glove")
    en_glove_path = os.path.join(glove_dir, 'en_vectors.txt')
    ch_glove_path = os.path.join(glove_dir, 'ch_vectors.txt')
    parser.add_argument('-s', "--source_dir", default=source_dir)
    parser.add_argument("--version", default=version)
    parser.add_argument('-d', "--debug", action='store_true')
    parser.add_argument("--train_ratio", default=1.0, type=int)
    parser.add_argument("--ch_glove_corpus", default=en_glove_path, type=str)
    parser.add_argument("--en_glove_corpus", default=ch_glove_path, type=str)
    parser.add_argument("--embedding_f", default='../data/dim_300/mix_vectors.txt', type=str)
    parser.add_argument("--glove_vec_size", default=300, type=int)
    parser.add_argument("--mode", default="full", type=str)
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--cut_q", default=False, type=bool)
    parser.add_argument("--boundary", default=10, type=int)
    parser.add_argument("--candidate_nums", default=10, type=int)
    parser.add_argument("--no_passages", default=False, type=bool)

    # TODO : put more args here
    return parser.parse_args()


def save(args, data, shared, data_type):
    data_path = os.path.join(args.target_dir, "data_{}.json".format(data_type))
    shared_path = os.path.join(args.target_dir, "shared_{}.json".format(data_type))
    json.dump(data, open(data_path, 'w'))
    json.dump(shared, open(shared_path, 'w'))


def get_word2vec(args, word_counter):
    word2vec_dict = {}
    with open(args.en_glove_corpus, 'r', encoding='utf-8') as fh:
        for line in tqdm(fh):
            array = line.lstrip().rstrip().split(" ")
            word = array[0]
            vector = list(map(float, array[1:]))
            if word in word_counter:
                word2vec_dict[word] = vector
            elif word.capitalize() in word_counter:
                word2vec_dict[word.capitalize()] = vector
            elif word.lower() in word_counter:
                word2vec_dict[word.lower()] = vector
            elif word.upper() in word_counter:
                word2vec_dict[word.upper()] = vector
    with open(args.ch_glove_corpus, 'r', encoding='utf-8') as fh:
        for line in tqdm(fh):
            array = line.lstrip().rstrip().split(" ")
            word = array[0]
            vector = list(map(float, array[1:]))
            if word in word_counter:
                word2vec_dict[word] = vector
            elif word.capitalize() in word_counter:
                word2vec_dict[word.capitalize()] = vector
            elif word.lower() in word_counter:
                word2vec_dict[word.lower()] = vector
            elif word.upper() in word_counter:
                word2vec_dict[word.upper()] = vector

    print("{}/{} of word vocab have corresponding vectors in {} and {}".format(len(word2vec_dict), len(word_counter),\
                        args.ch_glove_corpus, args.en_glove_corpus))
    return word2vec_dict


def padding_seq(seq):
    results = []
    max_len = 0
    for s in seq:
        if max_len < len(s):
            max_len = len(s)
    for i in range(0, len(seq)):
        l = max_len - len(seq[i])
        results.append(seq[i] + [0 for j in range(l)])
    return results


def get_minibatches(n, minibatch_size, shuffle=False):
    idx_list = np.arange(0, n, minibatch_size).astype(np.int32)
    if shuffle:
        np.random.shuffle(idx_list)
    minibatches = []
    import pdb
    for idx in idx_list:
        tmp = np.arange(idx, min(idx+minibatch_size, n))
        if len(tmp) < minibatch_size:
            tmp1 = list(tmp)
            tmp2 = list(np.arange(0, minibatch_size-len(tmp)))
            tmp = np.array(tmp1 + tmp2)
        minibatches.append(tmp)
    return minibatches


def build_dict(args, sentences, emb_f, max_words=50000):
    """
        Build a dictionary for the words in `sentences`.
        Only the max_words ones are kept and the remaining will be mapped to <UNK>.
        """

    word_count = Counter()
    for sent in sentences:
        for w in sent.split(' '):
            word_count[w] += 1
    max_words = int(len(word_count.keys()) * 0.95)
    ls = word_count.most_common(max_words)
    logging.info('#Words: %d -> %d' % (len(word_count), len(ls)))
    for key in ls[:5]:
        logging.info(key)
    logging.info('...')
    for key in ls[-5:]:
        logging.info(key)
    ls.append(("<s>", 10000))
    ls.append(("</s>", 10000))
    word_dict = {w[0]: index for (index, w) in enumerate(ls)}
    pre_trained_emb = []
    pre_trained_words = []
    #embeddings = np.random.randn(len(word_dict.keys()),args.glove_vec_size)
    j = 0
    for line in open(emb_f).readlines():
        sp = line.split()
        if len(sp) == 2 and int(sp[1]) == args.word_emb_size:
            continue
        assert len(sp) == args.word_emb_size + 1
        if sp[0] in word_dict and sp[0] not in pre_trained_words:
            assert word_dict[sp[0]] >= 0
            pre_trained_emb.append([float(x) for x in sp[1:]])
            pre_trained_words.append(sp[0])
            j += 1
    not_in_w2v = []
    for w in word_dict.keys():
        if w not in pre_trained_words:
            not_in_w2v.append(w)
    assert 'unk' not in pre_trained_words
    new_word_dict = ['unk'] + not_in_w2v + pre_trained_words
    assert len(new_word_dict) == len(word_dict.keys()) + 1
    return {w: index for (index, w) in enumerate(new_word_dict)}, {index: w for (index, w) in enumerate(new_word_dict)},\
           np.array(pre_trained_emb, dtype=np.float32)


def gen_examples(data, batch_size):
    """
        Divide examples into batches of size `batch_size`.
    """
    minibatches = get_minibatches(len(data['ids']), batch_size)
    import pdb
    #pdb.set_trace()
    for minibatch in minibatches:
        mb_x0 = [data['ids'][t] for t in minibatch]
        mb_x1 = [data['doc'][t] for t in minibatch]
        mb_x2 = [data['que'][t] for t in minibatch]

        mb_y = [data['answer'][t] for t in minibatch]
        mb_c = [data['candidates'][t] for t in minibatch]

        mb_x1, mb_mask1 = prepare_data(mb_x1)
        mb_x1_len = mb_mask1.sum(axis=1)
        mb_x2, mb_mask2 = prepare_data(mb_x2)
        mb_x2_len = mb_mask2.sum(axis=1)
        yield {'ids': mb_x0, 'doc': mb_x1, 'que': mb_x2,
               'doc_len': mb_x1_len, 'que_len': mb_x2_len,
               'doc_mask': mb_mask1, "que_mask": mb_mask2,
               'candidates': mb_c, 'y': mb_y}
#        all_ex.append((mb_x0, mb_x1, mb_mask1, mb_x2, mb_mask2, mb_c, mb_y))
#    return all_ex


def prepro_each(in_file, start_ratio=0, end_ratio=1):

    """
        load CNN / Daily Mail data from {train | dev | test}.txt
        relabeling: relabel the entities by their first occurence if it is True.
    """
    ids = []
    documents = []
    source_questions = []
    target_questions = []
    answers = []
    candidates = []
    f = open(in_file, 'r')
    i = 0
    lines = f.readlines()
    l = len(lines)
    from_id = int(start_ratio*l)
    to_id = int(end_ratio*l)
    for idx, line in enumerate(lines[from_id:to_id]):
        #print(idx)
        sample = json.loads(line.strip('\n'))
        ids.append(sample['id'])
        target_question = (' '.join([w[0] for w in sample['question']])).replace('\n', '')
        assert 'XXXX' in target_question
        target_questions.append(target_question)
        candidates.append([c[0] for c in sample['candidates']])
        answers.append(sample['answer'][0])
        doc = []
        for j in range(len(sample['passage'])):
            doc.append((' '.join([w[0] for w in sample['passage'][j]])).replace('\n', ''))
        doc = ' '.join(doc)
        documents.append(doc)
        i += 1
    f.close()
    logging.info('#Examples: %d' % len(documents))
    return ids, documents, target_questions, answers, candidates


def vectorize(examples, word_dict, args,
              sort_by_len=True, verbose=True):
    """
        Vectorize `examples`.
        in_x1, in_x2: sequences for document and question respecitvely.
        in_y: label
        """

    def get_word_id(w, word_dict):
        if w in word_dict:
            return word_dict[w]
        else:
            return word_dict['unk']

    in_x0 = []  # ids
    in_x1 = []  # doc
    in_x2 = []  # que
    in_y = np.zeros((len(examples[0]), 1)).astype(np.float32)
    in_c = []
    n = 0
    #import pdb
    #pdb.set_trace()
    for idx, (m_id, d, t_q, a, c) in enumerate(zip(examples[0], examples[1], examples[2], examples[3], examples[4])):
        d_words = d.split(' ')
        t_q_words = t_q.split(' ')
        if args.cut_q:
            q_pos = q_words.index('XXXX')
            q_words = q_words[q_pos - args.boundary:q_pos + args.boundary]
        if args.no_passages:
            seq1 = np.zeros(10)
        else:
            seq1 = [get_word_id(w, word_dict) for w in d_words]

        seq2 = [get_word_id(w, word_dict) for w in t_q_words]

        if a not in word_dict:
            logging.info("answer word %s not in word_dict " % a)
            continue
        tmp = []
        c_not_in_candidates = False
        for j in range(args.candidate_nums):
            w_id = get_word_id(c[j], word_dict)
            if w_id == get_word_id('unk', word_dict):
                c_not_in_candidates = True
                break
            tmp.append(w_id)
        if not c_not_in_candidates:
            continue
        random.shuffle(tmp)
        if (len(seq1) > 0) and (len(seq2) > 0):
            if args.monolingual:
                if word_dict[a] not in seq1:
                    continue
            in_x0.append(m_id)
            in_x1.append(seq1)
            in_x2.append(seq2)
            tmp = []

            in_y[n, 0] = tmp.index(word_dict[a])
            assert word_dict[a] in tmp
            in_c.append(tmp)
            n += 1
        else:
            print(a)
    print(sum(map(len, in_x1)) / len(in_c))
    print(sum(map(len, in_x2)) / len(in_c))
    print('Vectorization: processed %d / %d' % (n, len(examples[0])))
    logging.info('Vectorization: processed %d / %d' % (n, len(examples[0])))
    in_y = in_y[:n]
    assert len(in_y) == len(in_x1)
    assert len(in_x0) == len(in_x1) == len(in_x2)

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    if sort_by_len:
        # sort by the document length
        sorted_index = len_argsort(in_x1)
        in_x0 = [in_x0[i] for i in sorted_index]
        in_x1 = [in_x1[i] for i in sorted_index]
        in_x2 = [in_x2[i] for i in sorted_index]
        in_y = [in_y[i].tolist() for i in sorted_index]
        in_c = [in_c[i] for i in sorted_index]

    return {'ids': in_x0,
            'doc': in_x1,
            'que': in_x2,
            'candidates': in_c,
            'answer': in_y}


def prepare_data(seqs):
    lengths = [len(seq) for seq in seqs]
    n_samples = len(seqs)
    max_len = np.max(lengths)
    x = np.zeros((n_samples, max_len)).astype('int32')
    x_mask = np.zeros((n_samples, max_len), np.float32)
    for idx, seq in enumerate(seqs):
        x[idx, :lengths[idx]] = seq
        x_mask[idx, :lengths[idx]] = 1.0
    return x, x_mask


if __name__ == "__main__":
    import pdb
    #pdb.set_trace()
    main()
    load_data()
