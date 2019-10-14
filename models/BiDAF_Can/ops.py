import tensorflow as tf
import tensorflow.contrib.cudnn_rnn as cudnn_rnn
import numpy as np
from functools import reduce
from operator import mul
import pdb

def m_dynamic_rnn(inputs, num_layers, hidden_size, weight_noise_std, is_training, i_m, input_drop_prob=1.0,
                     use_gru=True, scope=None):
    #pdb.set_trace()
    flat_inputs = flatten(inputs, 2)  # [-1, J, d]
    with tf.variable_scope(scope):
        flat_outputs, _ = dynamic_rnn(flat_inputs, num_layers, hidden_size, weight_noise_std, is_training,i_m,input_drop_prob,use_gru)
    outputs = reconstruct(flat_outputs, inputs, 2)
    #states = reconstruct(states,inputs,2)
    return outputs, None

def dynamic_rnn(inputs, num_layers, hidden_size, weight_noise_std, is_training, i_m, input_drop_prob=1.0,
                     use_gru=True, scope=None):
    lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=input_drop_prob)
    # Backward direction cell
    lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=input_drop_prob)
    outputs, output_states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, inputs, dtype=tf.float32)
    outputs = tf.concat(outputs, 2)
    #outputs = tf.transpose(outputs, [1, 0, 2])
    return outputs,None

def m_cudnn_lstm(inputs, num_layers, hidden_size, weight_noise_std, is_training,i_m,input_drop_prob=1.0,use_gru=True,scope=None):
    """Run the CuDNN LSTM.
    Arguments:
        - inputs:   A tensor of shape [batch, max_sents_num,max_sent_length, input_size] of inputs.
        - num_layers:   Number of RNN layers.
        - hidden_size:     Number of units in each layer.
        - is_training:     tf.bool indicating whether training mode is enabled.
    Return a tuple of (outputs, init_state, final_state).
    """
    #pdb.set_trace()
    flat_inputs = flatten(inputs, 2)  # [-1, J, d]
    with tf.variable_scope(scope):
        flat_outputs,states = cudnn_lstm(flat_inputs, num_layers, hidden_size, weight_noise_std, is_training,i_m,input_drop_prob,use_gru)
    outputs = reconstruct(flat_outputs, inputs, 2)
    #states = reconstruct(states,inputs,2)
    return outputs, None


def cudnn_lstm(inputs, num_layers, hidden_size, weight_noise_std, is_training,i_m,input_drop_prob,use_gru):
    """Run the CuDNN LSTM.
    Arguments:
        - inputs:   A tensor of shape [batch, length, input_size] of inputs.
        - layers:   Number of RNN layers.
        - hidden_size:     Number of units in each layer.
        - is_training:     tf.bool indicating whether training mode is enabled.
    Return a tuple of (outputs, init_state, final_state).
    """
    input_size = inputs.get_shape()[-1].value
    if input_size is None:
        raise ValueError("Number of input dimensions to CuDNN RNNs must be "
                         "known, but was None.")

    # CUDNN expects the inputs to be time major
    inputs = tf.transpose(inputs, [1, 0, 2])
    if use_gru:  #num_layers,NUM_UNITS,input_size,input_mode = 'auto_select',direction = 'unidirectional',dropout = 0.0,seed = 0)
        #cudnn_cell = tf.contrib.cudnn_rnn.CudnnGRU(
        #    num_layers, hidden_size, input_size,
        #    input_mode=i_m, direction="bidirectional",dropout=input_drop_prob)
        cudnn_cell = tf.contrib.cudnn_rnn.CudnnGRU(
            num_layers, hidden_size,
            input_mode=i_m, direction="bidirectional", dropout=input_drop_prob)

    else:
        #cudnn_cell = tf.contrib.cudnn_rnn.CudnnLSTM(
        #    num_layers, hidden_size, input_size,
        #    input_mode=i_m, direction="bidirectional", dropout=input_drop_prob)

        cudnn_cell = tf.contrib.cudnn_rnn.CudnnLSTM(
            num_layers, hidden_size,
            input_mode=i_m, direction="bidirectional",dropout=input_drop_prob)

    est_size = estimate_cudnn_parameter_size(
        use_gru=use_gru,
        num_layers=num_layers,
        hidden_size=hidden_size,
        input_size=input_size,
        input_mode=i_m,
        direction="bidirectional")

    cudnn_params = tf.get_variable(
        "RNNParams",
        shape=[est_size],
        initializer=tf.contrib.layers.variance_scaling_initializer())
    if weight_noise_std is not None:
        cudnn_params = weight_noise(
            cudnn_params,
            stddev=weight_noise_std,
            is_training=is_training)
# initial_state: a tuple of tensor(s) of shape`[num_layers * num_dirs, batch_size, num_units]
    init_state = tf.tile(
        tf.zeros([2 * num_layers, 1, hidden_size], dtype=tf.float32),
        [1, tf.shape(inputs)[1], 1])  # [2 * num_layers, batch_size, hidden_size]
    '''
    Args:
      inputs: `3-D` tensor with shape `[time_len, batch_size, input_size]`.
      initial_state: a tuple of tensor(s) of shape
        `[num_layers * num_dirs, batch_size, num_units]`. If not provided, use
        zero initial states. The tuple size is 2 for LSTM and 1 for other RNNs.
      training: whether this operation will be used in training or inference.
    Returns:
      output: a tensor of shape `[time_len, batch_size, num_dirs * num_units]`.
        It is a `concat([fwd_output, bak_output], axis=2)`.
      output_states: a tuple of tensor(s) of the same shape and structure as
        `initial_state`.
    '''
    #import pdb
    #pdb.set_trace()
    if not use_gru:
        #hiddens,output = cudnn_cell(
        #    inputs,
        #    init_state, # input_c
        #    init_state, # input_h
        #    params=cudnn_params,
        #    is_training=True)
        hiddens, output_h, output_c = cudnn_cell(
            inputs,
            input_h=init_state,
            input_c=init_state,
            params=cudnn_params,
            is_training=True)
        hiddens = tf.transpose(hiddens, [1, 0, 2])
        output_h = tf.transpose(output_h, [1, 0, 2])
        output_c = tf.transpose(output_c, [1, 0, 2])
        output = (output_h,output_c)
    else:
        hiddens, output = cudnn_cell(
            inputs,
            (init_state,))
        #hiddens, output = cudnn_cell(
        #    inputs,
        #    init_state,  # input_h
        #    params=cudnn_params,
        #    is_training=True)
        hiddens = tf.transpose(hiddens, [1, 0, 2])
        #output = tf.transpose(output, [1, 0, 2])

    # Convert to batch major

    #return hiddens, output_h, output_c
    return hiddens, output


def cudnn_lstm_parameter_size(input_size, hidden_size):
    """Number of parameters in a single CuDNN LSTM cell."""
    biases = 8 * hidden_size
    weights = 4 * (hidden_size * input_size) + 4 * (hidden_size * hidden_size)
    return biases + weights

def cudnn_gru_parameter_size(input_size, hidden_size):
    """Number of parameters in a single CuDNN LSTM cell."""
    biases = 6 * hidden_size
    weights = 3 * (hidden_size * input_size) + 3 * (hidden_size * hidden_size)
    return biases + weights

def direction_to_num_directions(direction):
    if direction == "unidirectional":
        return 1
    elif direction == "bidirectional":
        return 2
    else:
        raise ValueError("Unknown direction: %r." % (direction,))


def estimate_cudnn_parameter_size(use_gru,
                                  num_layers,
                                  input_size,
                                  hidden_size,
                                  input_mode,
                                  direction):
    """
    Compute the number of parameters needed to
    construct a stack of LSTMs. Assumes the hidden states
    of bidirectional LSTMs are concatenated before being
    sent to the next layer up.
    """
    num_directions = direction_to_num_directions(direction)
    params = 0
    isize = input_size
    for layer in range(num_layers):
        for direction in range(num_directions):
            if use_gru:
                params += cudnn_gru_parameter_size(
                    isize, hidden_size
                )
            else:
                params += cudnn_lstm_parameter_size(
                    isize, hidden_size
                )
        isize = hidden_size * num_directions
    return params

def parameter_count():
    """Return the total number of parameters in all Tensorflow-defined
    variables, using `tf.trainable_variables()` to get the list of
    variables."""
    return sum(np.product(var.get_shape().as_list())
               for var in tf.trainable_variables())

def flatten(tensor, keep):
    fixed_shape = tensor.get_shape().as_list()
    start = len(fixed_shape) - keep
    left = reduce(mul, [fixed_shape[i] or tf.shape(tensor)[i] for i in range(start)])
    out_shape = [left] + [fixed_shape[i] or tf.shape(tensor)[i] for i in range(start, len(fixed_shape))]
    flat = tf.reshape(tensor, out_shape)
    return flat

def reconstruct(tensor, ref, keep):
    ref_shape = ref.get_shape().as_list()  # context:[N,M,JX,d] question:[N,JQ,d]
    tensor_shape = tensor.get_shape().as_list()  # context:[N*M,JX,d] question:[N,JQ,d]
    ref_stop = len(ref_shape) - keep  # context:2,question:1
    tensor_start = len(tensor_shape) - keep  # 1
    pre_shape = [ref_shape[i] or tf.shape(ref)[i] for i in range(ref_stop)]
    keep_shape = [tensor_shape[i] or tf.shape(tensor)[i] for i in range(tensor_start, len(tensor_shape))]
    # pre_shape = [tf.shape(ref)[i] for i in range(len(ref.get_shape().as_list()[:-keep]))]
    # keep_shape = tensor.get_shape().as_list()[-keep:]
    target_shape = pre_shape + keep_shape
    out = tf.reshape(tensor, target_shape)
    return out

def weight_noise(weight, stddev, is_training):
    weight_shape = weight.get_shape().as_list()
    return tf.cond(is_training,
                   lambda: weight + tf.random_normal(shape=weight_shape,
                                                     stddev=stddev,
                                                     mean=0.0,
                                                     dtype=tf.float32),
                   lambda: weight)

def semibatch_matmul(values, matrix, name=None):
    """Multiply a batch of matrices by a single matrix.
    Unlike tf.batch_matmul, which requires 2 3-D tensors, semibatch_matmul
    requires one 3-D tensor and one 2-D tensor.
    Arguments:
        values: A tensor of shape `[batch, n, p]`.
        matrix: A tensor of shape `[p, m]`.
        name: (Optional) A name for the operation.
    Returns a tensor of shape `[batch, n, m]`, where the outputs are:
        output[i, ...] = tf.matmul(values[i, ...], matrix)
    """
    with tf.name_scope(name or "SemibatchMatmul"):
        values = tf.convert_to_tensor(values, name="Values")
        matrix = tf.convert_to_tensor(matrix, name="Matrix")

        # Reshape input to be amenable to standard matmul
        values_shape = tf.shape(values, "ValuesShape")
        batch, n, p = values_shape[0], values_shape[1], values_shape[2]
        reshaped = tf.reshape(values, [-1, p],name="CollapseBatchDim")
        output = tf.matmul(reshaped, matrix, name="Matmul")
        # Reshape output back to batched form
        m = matrix.get_shape()[1].value
        output = tf.reshape(output, [batch, n, m], name="Output")
        return output