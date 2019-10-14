import os
from model_doc_bert.en_prepro import prepro_each, build_dict, vectorize, load_vocab
import json
import tensorflow as tf


flags = tf.app.flags

# Names and directories
flags.DEFINE_string("version", "pure_ch", "Data dir [data/squad]")
flags.DEFINE_string("emb_f", "0", "Run ID [0]")
flags.DEFINE_integer("word_emb_size", "300", "Run ID [0]")
flags.DEFINE_integer("max_p_num", 6, "Run ID [0]")
flags.DEFINE_integer("candidate_nums", 10, "Run ID [0]")
flags.DEFINE_string("log_dir", "/data/private/dyn/MRC/data/CrossTest/processed/mix_ch/", "Run ID [0]")
flags.DEFINE_bool("cut_q", False, "Data dir [data/squad]")
flags.DEFINE_bool("no_passages", False, "Data dir [data/squad]")
flags.DEFINE_bool("monolingual", False, "Data dir [data/squad]")
flags.DEFINE_string("bert_vocab_file", "0", "Run ID [0]")


def generate_data(config):
    import pdb
    pdb.set_trace()
    source_train_f = os.path.join('/data/private/dyn/MRC/data/CrossTest', "raw", config.version, 'train.json')
    source_dev_f = os.path.join('/data/private/dyn/MRC/data/CrossTest', "raw", config.version, 'dev.json')
    source_test_f = os.path.join('/data/private/dyn/MRC/data/CrossTest', "raw", config.version, 'test.json')

    processed_train_f = os.path.join('/data/private/dyn/MRC/data/CrossTest', "processed",  "doc_bert", config.version, 'train.json')
    processed_dev_f = os.path.join('/data/private/dyn/MRC/data/CrossTest', "processed",  "doc_bert", config.version, 'test.json')
    processed_test_f = os.path.join('/data/private/dyn/MRC/data/CrossTest', "processed",  "doc_bert", config.version, 'dev.json')
    processed_vocab_f = os.path.join('/data/private/dyn/MRC/data/CrossTest', "processed", "doc_bert", config.version,
                                     'vocab.data')

    train_data = prepro_each(source_train_f, config)
    dev_data = prepro_each(source_dev_f, config)
    test_data = prepro_each(source_test_f, config)

    word_2_id_dict, id_2_word_dict, pretrained_emb = build_dict(config, train_data[2] + train_data[3] +
                                                                [" ".join(train_data[4][i]) for i in
                                                                 range(len(train_data[4]))] +
                                                                 dev_data[2] + dev_data[3] + [
                                                                    " ".join(dev_data[4][i]) for i in
                                                                    range(len(dev_data[4]))],
                                                                emb_f=config.emb_f)

    vocab_data = {"word2id": word_2_id_dict, "id2word": id_2_word_dict, "pretrained_emb": pretrained_emb.tolist()}

    bert_word_2_id_dict = load_vocab(config.bert_vocab_file)
    train_data = vectorize(examples=train_data, bert_word_dict=bert_word_2_id_dict, normal_word_dict=word_2_id_dict, args=config)
    dev_data = vectorize(examples=dev_data, bert_word_dict=bert_word_2_id_dict, normal_word_dict=word_2_id_dict, args=config)
    test_data = vectorize(examples=test_data, bert_word_dict=bert_word_2_id_dict, normal_word_dict=word_2_id_dict, args=config)

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
