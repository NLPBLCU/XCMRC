import argparse
import json
import math
import os
import shutil
from pprint import pprint

import tensorflow as tf
from tqdm import tqdm
import numpy as np

from QO_Can.evaluator import Evaluator
from QO_Can.graph_handler import GraphHandler
from QO_Can.trainer import Trainer
from QO_Can.m_prepro import build_dict, prepro_each,vectorize,gen_examples
from QO_Can.model import BaselineModel
from my.tensorflow import get_num_params
import pdb
import logging
import copy

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
    with tf.device("/gpu:0"):
        if config.mode == 'train':
            _train(config)
        elif config.mode == 'test':
            _test(config)
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
    processed_train_f = os.path.join('/data/private/dyn/MRC/data/CrossTest', "processed", "can", config.version, 'train.json')
    processed_dev_f = os.path.join('/data/private/dyn/MRC/data/CrossTest', "processed", "can", config.version, 'dev.json')
    processed_test_f = os.path.join('/data/private/dyn/MRC/data/CrossTest', "processed", "can", config.version, 'test.json')
    processed_vocab_f = os.path.join('/data/private/dyn/MRC/data/CrossTest', "processed", "can", config.version, 'vocab.data')

    import pdb
    # pdb.set_trace()
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
        model = BaselineModel(config, len(word_2_id_dict.keys()), emb)

    # construct model graph and variables (using default graph)
    #with open(os.path.join(config.out_dir, 'flags'),'w') as fw:
    #    inf = copy.deepcopy(config.__flags)
    #    fw.write(json.dumps(inf, indent=2))
    pprint(config.__flags, indent=2)
    trainer = Trainer(config, model, word_2_id_dict)
    evaluator = Evaluator(config, model, word_2_id_dict)
    graph_handler = GraphHandler(config, model)  # controls all tensors and variables in the graph, including loading /saving

    # Variables
    tf_config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=tf_config)
    graph_handler.initialize(sess)

    # Begin training
    train_num_steps = int(math.ceil(len(train_data['ids']) / (config.batch_size )))
    global_step = 0
    best_res = 0
    log_every_n_batch = 10
    n_batch_loss = 0
    for epoch in range(config.epochs):
        losses = []
        for bitx, batches in enumerate(gen_examples(train_data, config.batch_size)):
            global_step = sess.run(model.global_step) + 1  # +1 because all calculations are done after step
            loss = trainer.step(sess, batches)
            losses.append(loss)
            n_batch_loss += loss
            if log_every_n_batch > 0 and bitx % log_every_n_batch == 0:
                logging.info('Average loss from batch {} to {} is {}'.format(
                    bitx - log_every_n_batch + 1, bitx, n_batch_loss / log_every_n_batch))
                n_batch_loss = 0
        e_dev = evaluator.get_evaluation_from_batches(
            sess, gen_examples(dev_data, config.batch_size))
        logging.info('epoch {} dev acc is {}:'.format(epoch, e_dev['acc']))

        e_test = evaluator.get_evaluation_from_batches(
            sess, gen_examples(test_data, config.batch_size))
        logging.info('epoch {} test acc is {}:'.format(epoch, e_test['acc']))

        if float(e_test['acc']) > best_res:
            best_res = float(e_test['acc'])
            graph_handler.save(sess, global_step=epoch)
            graph_handler.dump_answer(e_test, os.path.join(config.answer_dir, 'epoch_{}_answer.json').format(epoch))


def _test(config):
    source_train_f = os.path.join('/data/private/dyn/MRC/data/CrossTest', config.version, 'train.json')
    source_dev_f = os.path.join('/data/private/dyn/MRC/data/CrossTest', config.version, 'train.json')
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

    tf_config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=tf_config)
    graph_handler.initialize(sess)
    import pdb
    #pdb.set_trace()
    # Begin training
    test_num_steps = int(math.ceil(len(test_data['ids']) / (config.batch_size)))
    e_test = evaluator.get_evaluation_from_batches(
        sess, gen_examples(test_data, config.batch_size))
    print('----------------------test------------------------\n')
    print('echpo {} dev acc is {}'.format(config.load_step, e_test['acc']))


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
