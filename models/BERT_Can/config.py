import argparse


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')

def get_args():
    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', str2bool)

    # Basics
    parser.add_argument('--debug',
                        type='bool',
                        default=False,
                        help='whether it is debug mode')

    parser.add_argument('--dump_eval',
                        type='bool',
                        default=True,
                        help='whether it is debug mode')

    parser.add_argument('--test_only',
                        type='bool',
                        default=False,
                        help='test_only: no need to run training process')

    parser.add_argument('--no_passages',
                        type='bool',
                        default=False,
                        help='test_only: no need to run training process')

    parser.add_argument('--cut_q',
                        type='bool',
                        default=False,
                        help='test_only: no need to run training process')

    parser.add_argument('--use_cuda',
                        type='bool',
                        default=False,
                        help='use cuda')

    parser.add_argument('--boundary',
                        type=int,
                        default=5,
                        help='Random seed')

    parser.add_argument('--random_seed',
                        type=int,
                        default=1013,
                        help='Random seed')

    parser.add_argument('--eval_period',
                        type=int,
                        default=500,
                        help='Random seed')

    parser.add_argument('--loss_period',
                        type=int,
                        default=100,
                        help='Random seed')

    parser.add_argument('--max_train',
                        type=int,
                        default=60000,
                        help='Random seed')

    # Data file
    parser.add_argument('--train_file',
                        type=str,
                        default=None,
                        help='Training file')

    parser.add_argument('--noun_path',
                        type=str,
                        default='data/pure_ch/noun_set',
                        help='Training file')

    parser.add_argument('--dev_file',
                        type=str,
                        default=None,
                        help='Development file')

    parser.add_argument('--entity_path',
                    type=str,
                    default=None,
                    help='answer set file')

    parser.add_argument('--pre_trained',
                        type=str,
                        default=0,
                        help='Pre-trained model.')

    parser.add_argument('--load_pre_trained',
                        type='bool',
                        default=False,
                        help='whether it is debug mode')

    parser.add_argument('--out_base_dir',
                        type=str,
                        default='out',
                        help='out base dir')

    parser.add_argument('--run_id',
                        type=int,
                        default=0,
                        help='run id')

    parser.add_argument('--model_name',
                        type=str,
                        default='basic',
                        help='Model file to save')

    parser.add_argument('--log_file',
                        type=str,
                        default=None,
                        help='Log file')

    parser.add_argument('--embedding_file',
                        type=str,
                        default=None,
                        help='Word embedding file')

    parser.add_argument('--max_dev',
                        type=int,
                        default=None,
                        help='Maximum number of dev examples to evaluate on')

    parser.add_argument('--relabeling',
                        type='bool',
                        default=True,
                        help='Whether to relabel the entities when loading the data')

    # Model details
    parser.add_argument('--embedding_size',
                        type=int,
                        default=200,
                        help='Default embedding size if embedding_file is not given')

    parser.add_argument('--answer_embed_size',
                        type=int,
                        default=200,
                        help='Default embedding size if embedding_file is not given')

    parser.add_argument('--word_emb_size',
                        type=int,
                        default=300,
                        help='Default embedding size if embedding_file is not given')

    parser.add_argument('--hidden_size',
                        type=int,
                        default=128,
                        help='Hidden size of RNN units')

    parser.add_argument('--bidir',
                        type='bool',
                        default=True,
                        help='bidir: whether to use a bidirectional RNN')

    parser.add_argument('--limit_train',
                        type='bool',
                        default=False,
                        help='limit the number of trainning samples')

    parser.add_argument('--num_layers',
                        type=int,
                        default=1,
                        help='Number of RNN layers')

    parser.add_argument('--rnn_type',
                        type=str,
                        default='gru',
                        help='RNN type: lstm or gru (default)')

    parser.add_argument('--att_func',
                        type=str,
                        default='bilinear',
                        help='Attention function: bilinear (default) or mlp or avg or last or dot')

    parser.add_argument('--pass_candidates',
                        type=int,
                        default=4,
                        help='candidates from passage')

    parser.add_argument('--candidate_nums',
                        type=int,
                        default=9,
                        help='candidates from passage')

    parser.add_argument('--top_n_candidates',
                        type=int,
                        default=4,
                        help='candidates from word embedding')

    # Optimization details
    parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        help='Batch size')

    parser.add_argument('--num_epoches',
                        type=int,
                        default=7,
                        help='Number of epoches')

    parser.add_argument('--eval_iter',
                        type=int,
                        default=100,
                        help='Evaluation on dev set after K updates')

    parser.add_argument('--dropout_rate',
                        type=float,
                        default=0.5,
                        help='Dropout rate')

    parser.add_argument('--optimizer',
                        type=str,
                        default='sgd',
                        help='Optimizer: sgd (default) or adam or rmsprop')

    parser.add_argument('--learning_rate', '-lr',
                        type=float,
                        default=0.001,
                        help='Learning rate for SGD')

    parser.add_argument('--start_ratio',
                        type=float,
                        default=0,
                        help='Learning rate for SGD')

    parser.add_argument('--end_ratio',
                        type=float,
                        default=1.0,
                        help='Learning rate for SGD')

    parser.add_argument('--grad_clipping',
                        type=float,
                        default=10.0,
                        help='Gradient clipping')

    return parser.parse_args()
