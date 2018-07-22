import tensorflow as tf


class RNNConfig:
    """Config for this RNN."""
    layers = [512],
    dropout_keep_prob =0.8


class RNN:

    def __init__(self, config, batch_size, size_out):
        """Create RNN
        :param config an instance of rnn.RNNConfig
        :param batch_size size of a training/validation batch
        :param size_out number of units in the output vector."""
        self._config = config
        self._batch_size = batch_size
        self._size_out = size_out
        self._rnn_step, self._zero_state = self._build_rnn()

    def _build_rnn(self):
        with tf.variable_scope('rnn_step'):
            lstms = [tf.nn.rnn_cell.BasicLSTMCell(size_hidden) for size_hidden in
                     self._config['layers']]
            rnn = tf.nn.rnn_cell.MultiRNNCell(lstms)
            projection_layer = tf.layers.Dense(self._size_out)

        def rnn_step(inputs, state):
            rnn_output, new_state = rnn(inputs, state)
            output = projection_layer(rnn_output)
            return output, new_state

        return rnn_step, rnn.zero_state

    def _unwrap(self, rnn_step, init_inputs, init_state, next_inputs_fn, finished_fn, seq_len):
        state = init_state
        inputs = init_inputs
        zero_outputs = tf.zeros(shape=(self._batch_size, self._size_out))
        finished = tf.constant(False, shape=(self._batch_size,))
        output_seq = []
        for step in range(seq_len):
            outputs, new_state = rnn_step(inputs, state)
            finished = tf.logical_or(finished, finished_fn(step, outputs))
            outputs = tf.where(finished, zero_outputs, outputs)
            output_seq.append(outputs)
            inputs = next_inputs_fn(step, outputs, new_state)
        output_seq = [tf.expand_dims(output, axis=1) for output in output_seq]
        output_seq = tf.concat(output_seq, axis=1)
        return output_seq, new_state

    def __call__(self, init_inputs, init_state, finished_fn, seq_len, is_training=False,
                 labels=None):
        """Unwrap RNN for seq_len number of steps.
        When in training mode, labels at i-th position are used as inputs at step i+1.
        When in inference mode, outputs from the i-th step are used as inputs at step i+1.
        In both modes output state from i-th step is used as input state at step i+1.
        Once finish_fn tells that sequence should be terminated, this RNN will append
        zero outputs instead of actual model outputs to the output sequence.
        :param init_inputs: inputs at the step one.
        :param init_state: state at step one.
        :param finished_fn: callable, called with params (step, outputs) should return
        :param seq_len: number of steps to unwrap.
        :param is_training: whether in training mode or not.
        :param labels: desired outputs.
        """
        labels_seq_len = labels.shape[1]
        if labels_seq_len != seq_len:
            raise ValueError('Labels sequence length should be equal to seq_len ({} != {}'
                             .format(labels_seq_len, seq_len))
        if is_training and labels is None:
            raise ValueError('If model is in training mode, labels can not be None')

        def next_inputs_from_labels(step, outputs, state):
            del outputs, state # Unused
            return labels[:, step, :]

        def next_inputs_from_outputs(step, outputs, state):
            del step, state # Unused
            max_tokens = tf.argmax(outputs, axis=-1)
            return tf.one_hot(indices = max_tokens, depth=self._size_out, axis=-1)

        next_inputs = next_inputs_from_labels if is_training else next_inputs_from_outputs
        return self._unwrap(self._rnn_step, init_inputs, init_state,
                            next_inputs, finished_fn, seq_len)

    def zero_state(self, batch_size, dtype=tf.float32):
        return self._zero_state(batch_size, dtype)
