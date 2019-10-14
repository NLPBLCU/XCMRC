import os

import tensorflow as tf

from BiDAF_Cloze.main import main as m
import pdb

flags = tf.app.flags

# Names and directories
flags.DEFINE_string("model_name", "basic", "Model name [basic]")
flags.DEFINE_string("version", "pure_ch", "Data dir [data/squad]")
flags.DEFINE_string("vocab_file", "ch_vocab.txt", "vocab path")
flags.DEFINE_string("run_id", "0", "Run ID [0]")
flags.DEFINE_string("out_dir", "", "Run ID [0]")
flags.DEFINE_string("answer_dir", "0", "Run ID [0]")
flags.DEFINE_string("eval_dir", "0", "Run ID [0]")
flags.DEFINE_string("data_dir", "0", "Run ID [0]")
flags.DEFINE_string("save_dir", "0", "Run ID [0]")
flags.DEFINE_string("log_dir", "0", "Run ID [0]")
flags.DEFINE_string("out_base_dir", "out", "out base dir [out]")
flags.DEFINE_string("forward_name", "single", "Forward name [single]")
flags.DEFINE_string("answer_path", "", "Answer path []")
flags.DEFINE_string("eval_path", "", "Eval path []")
flags.DEFINE_string("load_path", "", "Load path []")
flags.DEFINE_string("shared_path", "", "Shared path []")

# Device placement
flags.DEFINE_string("device", "/cpu:0", "default device for summing gradients. [/cpu:0]")
flags.DEFINE_string("device_type", "gpu", "device for computing gradients (parallelization). cpu | gpu [gpu]")
flags.DEFINE_integer("num_gpus", 1, "num of gpus or cpus for computing gradients [1]")

# Essential training and test options
flags.DEFINE_string("mode", "test", "trains | test | forward [test]")
flags.DEFINE_boolean("load", True, "load saved data? [True]")
flags.DEFINE_bool("single", False, "supervise only the answer sentence? [False]")
flags.DEFINE_boolean("debug", False, "Debugging mode? [False]")
flags.DEFINE_bool('load_ema', True, "load exponential average of variables when testing?  [True]")
flags.DEFINE_bool("eval", True, "eval? [True]")
flags.DEFINE_bool("wy", False, "Use wy for loss / eval? [False]")
flags.DEFINE_bool("use_gru", True, "Use wy for loss / eval? [False]")
flags.DEFINE_bool("add_q", False, "Use wy for loss / eval? [False]")
flags.DEFINE_bool("use_cnn_4_encode", False, "Use wy for loss / eval? [False]")
flags.DEFINE_bool("use_lstm_4_encode", False, "Use wy for loss / eval? [False]")
flags.DEFINE_bool("na", False, "Enable no answer strategy and learn bias? [False]")
flags.DEFINE_float("th", 0.5, "Threshold [0.5]")

# Training / test parameters
flags.DEFINE_integer("batch_size", 60, "Batch size [60]")
flags.DEFINE_integer("VW_1", 60, "Batch size [60]")
flags.DEFINE_integer("VW_2", 60, "Batch size [60]")
flags.DEFINE_integer("word_emb_size", 300, "Batch size [60]")
flags.DEFINE_integer("val_num_batches", 100, "validation num batches [100]")
flags.DEFINE_integer("test_num_batches", 0, "test num batches [0]")
flags.DEFINE_integer("num_epochs", 12, "Total number of epochs for training [12]")
flags.DEFINE_integer("num_steps", 20000, "Number of steps [20000]")
flags.DEFINE_integer("load_step", 0, "load step [0]")
flags.DEFINE_float("init_lr", 0.001, "Initial learning rate [0.001]")
flags.DEFINE_float("input_keep_prob", 0.8, "Input keep prob for the dropout of LSTM weights [0.8]")
flags.DEFINE_float("keep_prob", 0.8, "Keep prob for the dropout of Char-CNN weights [0.8]")
flags.DEFINE_float("wd", 0.0, "L2 weight decay for regularization [0.0]")
flags.DEFINE_integer("hidden_size", 100, "Hidden size [100]")
flags.DEFINE_integer("char_out_size", 100, "char-level word embedding size [100]")
flags.DEFINE_integer("glove_vec_size", 300, "Hidden size [100]")
flags.DEFINE_integer("char_emb_size", 8, "Char emb size [8]")
flags.DEFINE_string("out_channel_dims", "100", "Out channel dims of Char-CNN, separated by commas [100]")
flags.DEFINE_string("filter_heights", "5", "Filter heights of Char-CNN, separated by commas [5]")
flags.DEFINE_bool("finetune", False, "Finetune word embeddings? [False]")
flags.DEFINE_bool("highway", True, "Use highway? [True]")
flags.DEFINE_integer("highway_layers", 2, "highway num layers [2]")
flags.DEFINE_bool("share_cnn_weights", True, "Share Char-CNN weights [True]")
flags.DEFINE_bool("share_lstm_weights", True, "Share pre-processing (phrase-level) LSTM weights [True]")
flags.DEFINE_float("var_decay", 0.999, "Exponential moving average decay for variables [0.999]")
flags.DEFINE_bool("point_words", False, "Use wy for loss / eval? [False]")

flags.DEFINE_float("l2_regularizer", 0.01, "Exponential moving average decay for variables [0.999]")
# Optimizations
flags.DEFINE_bool("cluster", False, "Cluster data for faster training [False]")
flags.DEFINE_bool("len_opt", False, "Length optimization? [False]")
flags.DEFINE_bool("add_l2_loss", False, "Length optimization? [False]")
flags.DEFINE_bool("cpu_opt", False, "CPU optimization? GPU computation can be slower [False]")

# Logging and saving options
flags.DEFINE_boolean("progress", True, "Show progress? [True]")
flags.DEFINE_integer("epochs", 8, "train epoch nums")
flags.DEFINE_integer("log_period", 100, "Log period [100]")
flags.DEFINE_integer("eval_period", 1000, "Eval period [1000]")
flags.DEFINE_integer("save_period", 1000, "Save Period [1000]")
flags.DEFINE_integer("max_to_keep", 20, "Max recent saves to keep [20]")

flags.DEFINE_bool("dump_eval", True, "dump eval? [True]")
flags.DEFINE_bool("dump_answer", True, "dump answer? [True]")
flags.DEFINE_bool("vis", False, "output visualization numbers? [False]")
flags.DEFINE_bool("dump_pickle", True, "Dump pickle instead of json? [True]")
flags.DEFINE_float("decay", 0.9, "Exponential moving average decay for logging values [0.9]")

# Thresholds for speed and less memory usage
flags.DEFINE_integer("word_count_th", 10, "word count th [100]")
flags.DEFINE_integer("char_count_th", 50, "char count th [500]")
flags.DEFINE_integer("sent_size_th", 400, "sent size th [64]")
flags.DEFINE_integer("num_sents_th", 8, "num sents th [8]")
flags.DEFINE_integer("ques_size_th", 30, "ques size th [32]")
flags.DEFINE_integer("word_size_th", 16, "word size th [16]")
flags.DEFINE_integer("para_size_th", 256, "para size th [256]")


# Advanced training options
flags.DEFINE_bool("lower_word", True, "lower word [True]")
flags.DEFINE_bool("squash", False, "squash the sentences into one? [False]")
flags.DEFINE_bool("swap_memory", True, "swap memory? [True]")
flags.DEFINE_string("data_filter", "max", "max | valid | semi [max]")
flags.DEFINE_bool("use_glove_for_unk", True, "use glove for unk [False]")
flags.DEFINE_bool("known_if_glove", True, "consider as known if present in glove [False]")
flags.DEFINE_string("logit_fcunc", "tri_linear", "logit func [tri_linear]")
flags.DEFINE_string("answer_func", "linear", "answer logit func [linear]")
flags.DEFINE_string("sh_logit_func", "tri_linear", "sh logit func [tri_linear]")

# Ablation options
flags.DEFINE_bool("use_char_emb", True, "use char emb? [True]")
flags.DEFINE_bool("use_word_emb", True, "use word embedding? [True]")
flags.DEFINE_bool("q2c_att", True, "question-to-context attention? [True]")
flags.DEFINE_bool("c2q_att", True, "context-to-question attention? [True]")
flags.DEFINE_bool("dynamic_att", False, "Dynamic attention [False]")
flags.DEFINE_bool("add_aoa", False, "Dynamic attention [False]")

flags.DEFINE_bool("has_other", False, "Dynamic attention [False]")
flags.DEFINE_integer("num_units", 150, "word count th [100]")
flags.DEFINE_integer("aff_num", 200, "word count th [100]")
flags.DEFINE_integer("num_heads", 1, "Max recent saves to keep [20]")
flags.DEFINE_string('--dilate_rates', "1, 2, 4, 8", 'dilate_rates')
flags.DEFINE_string('--windows', "3, 3, 3, 3", 'windows')
flags.DEFINE_integer("feature_nums", 150, "feature nums for cnn")
flags.DEFINE_integer("neg_samples", 4, "feature nums for cnn")
flags.DEFINE_float("pos_margin", 2.5, "feature nums for cnn")
flags.DEFINE_float("neg_margin", 0.5, "feature nums for cnn")

flags.DEFINE_integer("att_fea_nums", 200, "word count th [100]")
flags.DEFINE_integer("pos_emb_size", 25, "word count th [100]")
flags.DEFINE_integer("dc", 1000, "word count th [100]")
flags.DEFINE_integer("max_len_pos", 200, "word count th [100]")

flags.DEFINE_bool("type1_att", False, "Dynamic attention [False]")
flags.DEFINE_bool("type2_att", False, "Dynamic attention [False]")

flags.DEFINE_integer("u", 300, "word count th [100]")
flags.DEFINE_integer("w_l", 100, "word count th [100]")
flags.DEFINE_integer("start", 300, "word count th [100]")

flags.DEFINE_integer("max_sent_nums", 15, "num sents th [8]")
flags.DEFINE_integer("max_sent_size", 100, "num sents th [8]")
flags.DEFINE_integer("max_ques_size", 100, "num sents th [8]")
flags.DEFINE_bool("into_vocab", True, "softmax over vocabulary")
flags.DEFINE_bool("cut_q", False, "softmax over vocabulary")
flags.DEFINE_bool("no_passages", False, "softmax over vocabulary")
flags.DEFINE_bool("fine_tune", False, "fine tune word embeddings")
flags.DEFINE_integer("boundary", 10, "num sents th [8]")
flags.DEFINE_integer("candidate_nums", 9, "num sents th [8]")
flags.DEFINE_bool("monolingual", True, "fine tune word embeddings")
flags.DEFINE_string('emb_file', "/home/server3/dyn/data/dim_300/mix_vectors.txt", 'windows')

def main(_):
    config = flags.FLAGS
    import pdb
    import numpy as np
    #pdb.set_trace()
    np.random.seed(1234)
    m(config)

if __name__ == "__main__":
    tf.app.run()
