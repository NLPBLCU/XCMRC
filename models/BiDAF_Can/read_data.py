import os
from bidaf_can.m_prepro import prepro_each, build_dict, vectorize
import json
import tensorflow as tf


flags = tf.app.flags

# Names and directories
flags.DEFINE_string("version", "pure_ch", "Data dir [data/squad]")
flags.DEFINE_string("raw_type", "raw", "Data dir [data/squad]")
flags.DEFINE_string("pro_type", "processed", "Data dir [data/squad]")
flags.DEFINE_string("type", "can", "Data dir [data/squad]")
flags.DEFINE_string("emb_f", "0", "Run ID [0]")
flags.DEFINE_integer("word_emb_size", "300", "Run ID [0]")
flags.DEFINE_integer("candidate_nums", "10", "Run ID [0]")

flags.DEFINE_string("log_dir", "/data/private/dyn/MRC/data/CrossTest/processed/mix_en/", "Run ID [0]")
flags.DEFINE_bool("cut_q", False, "Data dir [data/squad]")
flags.DEFINE_bool("no_passages", False, "Data dir [data/squad]")
flags.DEFINE_bool("monolingual", False, "Data dir [data/squad]")

def generate_data(config):
    source_train_f = os.path.join('/data/private/dyn/MRC/data/CrossTest', config.raw_type,  config.version, 'train.json')
    source_dev_f = os.path.join('/data/private/dyn/MRC/data/CrossTest', config.raw_type, config.version, 'dev.json')
    source_test_f = os.path.join('/data/private/dyn/MRC/data/CrossTest', config.raw_type,  config.version, 'test.json')

    processed_train_f = os.path.join('/data/private/dyn/MRC/data/CrossTest', config.pro_type,  config.type, config.version, 'train.json')
    processed_dev_f = os.path.join('/data/private/dyn/MRC/data/CrossTest', config.pro_type, config.type, config.version, 'test.json')
    processed_test_f = os.path.join('/data/private/dyn/MRC/data/CrossTest', config.pro_type, config.type, config.version, 'dev.json')
    processed_vocab_f = os.path.join('/data/private/dyn/MRC/data/CrossTest', config.pro_type, config.type, config.version, 'vocab.data')

    train_data = prepro_each(source_train_f)
    dev_data = prepro_each(source_dev_f)
    test_data = prepro_each(source_test_f)
    word_2_id_dict, id_2_word_dict, pretrained_emb = build_dict(config, train_data[1] + train_data[2] + train_data[3] +
                                                                [" ".join(train_data[4][i]) for i in range(len(train_data[4]))] +
                                                                dev_data[1] + dev_data[2] + dev_data[3] + [" ".join(dev_data[4][i]) for i in range(len(dev_data[4]))] +
                                                                test_data[1] + test_data[2] + test_data[3] + [" ".join(test_data[4][i]) for i in range(len(test_data[4]))],
                                                                emb_f=config.emb_f)

    vocab_data = {"word2id": word_2_id_dict, "id2word": id_2_word_dict, "pretrained_emb": pretrained_emb.tolist()}
    train_data = vectorize(examples=train_data, word_dict=word_2_id_dict, args=config)
    dev_data = vectorize(examples=dev_data, word_dict=word_2_id_dict, args=config)
    test_data = vectorize(examples=test_data, word_dict=word_2_id_dict, args=config)

    with open(processed_train_f, "w") as fw:
        fw.write(json.dumps(train_data, ensure_ascii=False))

    with open(processed_dev_f, "w") as fw:
        fw.write(json.dumps(dev_data, ensure_ascii=False))

    with open(processed_test_f, "w") as fw:
        fw.write(json.dumps(test_data, ensure_ascii=False))

    with open(processed_vocab_f, "w") as fw:
        fw.write(json.dumps(vocab_data, ensure_ascii=False))


if __name__ == "__main__":
    config = flags.FLAGS
    generate_data(config)
