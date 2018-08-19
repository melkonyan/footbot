import tensorflow as tf
import unittest
import rnn

class RNNTest(unittest.TestCase):

    def setUp(self):
        tf.reset_default_graph()

    def testBuildRNN_inferenceMode(self):
        batch_size = 10
        seq_len = 5
        dict_size = 20
        model = rnn.RNN(rnn.RNNConfig(), batch_size=batch_size, dict_size=dict_size)
        init_inputs = tf.zeros(shape=(batch_size,), dtype=tf.int32)
        init_state = model.zero_state(batch_size)
        finished_fn = lambda step, outputs: tf.constant(True, shape=(batch_size,))
        t_outputs, t_state = model(init_inputs, init_state, finished_fn, seq_len)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            outputs, state = sess.run([t_outputs, t_state])
        self.assertEqual((batch_size, seq_len, dict_size), outputs.shape)

    def testBuildRNN_trainMode(self):
        batch_size = 10
        seq_len = 5
        dict_size = 20
        model = rnn.RNN(rnn.RNNConfig(embedding_size=10), batch_size=batch_size, dict_size=dict_size)
        init_inputs = tf.zeros(shape=(batch_size,), dtype=tf.int32)
        init_state = model.zero_state(batch_size)
        finished_fn = lambda step, outputs: tf.constant(True, shape=(batch_size,))
        labels = tf.zeros(shape=(batch_size, seq_len), dtype=tf.int32)
        t_outputs, t_state = model(init_inputs, init_state, finished_fn, seq_len,
                                   is_training=True, labels=labels)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            outputs, state = sess.run([t_outputs, t_state])
        self.assertEqual((batch_size, seq_len, dict_size), outputs.shape)


if __name__ == '__main__':
    unittest.main()