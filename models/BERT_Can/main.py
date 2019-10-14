import argparse
import json
import math
import os
import shutil
from pprint import pprint

import tensorflow as tf
from tqdm import tqdm
import numpy as np

from model_doc_bert.evaluator import Evaluator
from model_doc_bert.graph_handler import GraphHandler
from model_doc_bert.trainer import Trainer
from model_doc_bert.en_prepro import build_dict, prepro_each, vectorize, gen_examples, load_vocab
from model_doc_bert.model import BaselineModel
from my.tensorflow import get_num_params
from model_doc_bert.bert_model import *
import pdb
import copy
import logging

def main(config):
    set_dirs(config)
    config.data_dir = os.path.join('data', config.version)
    logging.basicConfig(level=logging.INFO,
                        filename=os.path.join(config.log_dir, 'train.log'),
                        filemode='a',
                        format=
                        '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                        # 日志格式
                        )
    #with tf.device(config.device):
    with tf.device("/gpu:0"):
        if config.mode == 'train':
            #pdb.set_trace()
            _train(config)
        elif config.mode == 'test':
            _test(config)
        elif config.mode == 'forward':
            _forward(config)
        else:
            raise ValueError("invalid value for 'mode': {}".format(config.mode))


def set_dirs(config):
    # create directories
    assert config.load or config.mode == 'train', "config.load must be True if not training"
    out_dir = os.path.join(config.out_base_dir, config.model_name, str(config.run_id).zfill(2))

    config.save_dir = os.path.join(out_dir, "save")
    config.log_dir = os.path.join(out_dir, "log")
    config.eval_dir = os.path.join(out_dir, "eval")
    config.answer_dir = os.path.join(out_dir, "answer")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if not os.path.exists(config.save_dir):
        os.mkdir(config.save_dir)
    if not os.path.exists(config.log_dir):
        os.mkdir(config.log_dir)
    if not os.path.exists(config.answer_dir):
        os.mkdir(config.answer_dir)
    if not os.path.exists(config.eval_dir):
        os.mkdir(config.eval_dir)


def _config_debug(config):
    if config.debug:
        config.num_steps = 2
        config.eval_period = 1
        config.log_period = 1
        config.save_period = 1
        config.val_num_batches = 2
        config.test_num_batches = 2


def _train(config):
    import pdb
    #pdb.set_trace()
    bert_config = BertConfig.from_json_file(config.bert_config_file)

    processed_train_f = os.path.join('/data/private/dyn/MRC/data/CrossTest', "processed", "doc_bert", config.version, 'train.json')
    processed_dev_f = os.path.join('/data/private/dyn/MRC/data/CrossTest', "processed", "doc_bert", config.version, 'dev.json')
    processed_test_f = os.path.join('/data/private/dyn/MRC/data/CrossTest', "processed", "doc_bert", config.version, 'test.json')
    processed_vocab_f = os.path.join('/data/private/dyn/MRC/data/CrossTest', "processed", "doc_bert", config.version, 'vocab.data')

    with open(processed_train_f, "r") as fr:
        train_data = json.load(fr)

    with open(processed_dev_f, "r") as fr:
        dev_data = json.load(fr)

    with open(processed_test_f, "r") as fr:
        test_data = json.load(fr)

    with open(processed_vocab_f, "r") as fr:
        vocab_data = json.load(fr)

    emb = vocab_data["pretrained_emb"]
    word_2_id_dict = vocab_data["word2id"]
    config.word_emb_size = np.shape(emb)[1]
    with tf.device("/gpu:0"):
        model = BaselineModel(config, bert_config, len(word_2_id_dict.keys()), emb)

    # construct model graph and variables (using default graph)
    #with open(os.path.join(config.out_dir, 'flags'),'w') as fw:
    #    inf = copy.deepcopy(config.__flags)
    #    fw.write(json.dumps(inf, indent=2))
    pprint(config.__flags, indent=2)
    trainer = Trainer(config, model, word_2_id_dict)
    evaluator = Evaluator(config, model, word_2_id_dict)
    graph_handler = GraphHandler(config, model)  # controls all tensors and variables in the graph, including loading /saving

    # Variables
    #cpu_num = int(os.environ.get('CPU_NUM', 1))
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    # 这一行设置 gpu 随使用增长，我一般都会加上
    tfconfig.gpu_options.allow_growth = True
    sess = tf.Session(config=tfconfig)

    graph_handler.initialize(sess)

    # Begin training
    train_num_steps = int(math.ceil(len(train_data['ids']) / (config.batch_size )))
    global_step = 0
    best_res = 0
    for epoch in range(config.epochs):
        losses = []
        for batches in gen_examples(train_data, config.batch_size):
            global_step = sess.run(model.global_step) + 1  # +1 because all calculations are done after step
            loss = trainer.step(sess, batches)
            losses.append(loss)
        import pdb
        logging.info('epoch {} loss {}:'.format(epoch, np.mean(losses)))
        dev_num_steps = int(math.ceil(len(dev_data['ids']) / (config.batch_size)))
        e_dev = evaluator.get_evaluation_from_batches(
            sess, gen_examples(dev_data, config.batch_size))
        logging.info('epoch {} dev acc is {}:'.format(epoch, e_dev['acc']))

        test_num_steps = int(math.ceil(len(test_data['ids']) / (config.batch_size)))
        e_test = evaluator.get_evaluation_from_batches(
            sess, gen_examples(test_data, config.batch_size))
        logging.info('epoch {} test acc is {}:'.format(epoch, e_test['acc']))

        if float(e_test['acc']) > best_res:
            best_res = float(e_test['acc'])
            graph_handler.save(sess, global_step=epoch)
            graph_handler.dump_answer(e_test, os.path.join(config.answer_dir,'epoch_{}_answer.json').format(epoch))

def _test(config):
    source_train_f = os.path.join('/data/private/dyn/MRC/data/CrossTest',config.version,'train.json')
    source_dev_f = os.path.join('/data/private/dyn/MRC/data/CrossTest',config.version,'dev.json')
    source_test_f = os.path.join('/data/private/dyn/MRC/data/CrossTest', config.version, 'test.json')

    train_data = prepro_each(source_train_f)
    dev_data = prepro_each(source_dev_f)
    test_data = prepro_each(source_test_f)
    word_2_id_dict, id_2_word_dict, pretrained_emb = build_dict(config, train_data[1] + train_data[2] + dev_data[1] + dev_data[2],emb_f='/data/private/dyn/MRC/data/dim_300/mix_vectors.txt')

    test_data = vectorize(examples=test_data, word_dict=word_2_id_dict, args=config)

    config.word_emb_size = np.shape(pretrained_emb)[1]

    model = BaselineModel(config, pretrained_emb)
    pprint(config.__flags, indent=2)
    trainer = Trainer(config, model)
    evaluator = Evaluator(config, model, word_2_id_dict)
    graph_handler = GraphHandler(config, model)  # controls all tensors and variables in the graph, including loading /saving

    # Variables
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    graph_handler.initialize(sess)
    import pdb
    #pdb.set_trace()
    # Begin training
    test_num_steps = int(math.ceil(len(test_data['ids']) / (config.batch_size)))
    e_test = evaluator.get_evaluation_from_batches(
        sess, gen_examples(test_data, config.batch_size))
    print('----------------------test------------------------\n')
    print('echpo {} dev acc is {}'.format(config.load_step, e_test['acc']))

def _forward(config):
    assert config.load
    test_data = read_data(config, config.forward_name, True)
    update_config(config, [test_data])

    _config_debug(config)

    if config.use_glove_for_unk:
        word2vec_dict = test_data.shared['lower_word2vec'] if config.lower_word else test_data.shared['word2vec']
        new_word2idx_dict = test_data.shared['new_word2idx']
        idx2vec_dict = {idx: word2vec_dict[word] for word, idx in new_word2idx_dict.items()}
        new_emb_mat = np.array([idx2vec_dict[idx] for idx in range(len(idx2vec_dict))], dtype='float32')
        config.new_emb_mat = new_emb_mat

    pprint(config.__flags, indent=2)
    models = get_multi_gpu_models(config)
    model = models[0]
    print("num params: {}".format(get_num_params()))
    evaluator = Evaluator(config, model)
    graph_handler = GraphHandler(config, model)  # controls all tensors and variables in the graph, including loading /saving

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    graph_handler.initialize(sess)

    num_batches = math.ceil(test_data.num_examples / config.batch_size)
    if 0 < config.test_num_batches < num_batches:
        num_batches = config.test_num_batches
    e = evaluator.get_evaluation_from_batches(sess, tqdm(test_data.get_batches(config.batch_size, num_batches=num_batches), total=num_batches))
    print(e)
    if config.dump_answer:
        print("dumping answer ...")
        graph_handler.dump_answer(e, path=config.answer_path)
    if config.dump_eval:
        print("dumping eval ...")
        graph_handler.dump_eval(e, path=config.eval_path)


def _get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path")
    return parser.parse_args()


class Config(object):
    def __init__(self, **entries):
        self.__dict__.update(entries)


def _run():
    args = _get_args()
    with open(args.config_path, 'r') as fh:
        config = Config(**json.load(fh))
        import pdb
        #pdb.set_trace()
        main(config)


if __name__ == "__main__":
    import logging
    logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                        level=logging.INFO)
    _run()
