import tensorflow as tf

class KeyphraseEncoder(object):
    """
    A keyphrase encoder which encode title sentence by average word embeddings.
    """
    def __init__(
      self,
      vocab_size,
      embedding_size,
      l2_reg_lambda=0.0):

        # Placeholders for input title, candidates, weights, and tag.
        self.input_x_sent = tf.placeholder(tf.int32, [None, None], name="input_x_sent")
        self.input_x_cands = tf.placeholder(tf.int32, [None, None], name="input_x_cands")
        self.input_x_cands_weight_crf = tf.placeholder(tf.float32, [None, None], name="input_x_cands_weight_crf") # Word frequency.
        self.input_x_cands_weight_cnn = tf.placeholder(tf.float32, [None, None], name="input_x_cands_weight_cnn") # Cnn output score, as private weights.
        self.input_x_tag = tf.placeholder(tf.int32, [None,], name="input_x_tag")  # number_of_samples_in_one_batch, number_of_candidates_in_one_samples
        self.input_y = tf.placeholder(tf.float32, [None,], name="input_y") # 0 or 1.
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        with tf.name_scope("word_embedding"):
            self.W_embedding_for_category = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                trainable=True,
                name="W_embedding_for_category")
            self.W_embedding_for_compete = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                trainable=True,
                name="W_embedding_for_compete")
            self.W = tf.Variable(10.0, name = "W") # Cnn public weight.
            self.embedded_sent = tf.nn.embedding_lookup(self.W_embedding_for_category, self.input_x_sent)
            self.embedded_cands = tf.nn.embedding_lookup(self.W_embedding_for_compete, self.input_x_cands)
            self.embedded_cands = tf.transpose(self.embedded_cands, [2,0,1])
            self.embedded_weighted_cands_crf = tf.transpose(tf.multiply(self.input_x_cands_weight_crf, self.embedded_cands), [1,2,0])
            self.embedded_weighted_cands_cnn = tf.multiply(self.W, tf.transpose(tf.multiply(self.input_x_cands_weight_cnn, self.embedded_cands), [1,2,0]))
            self.embedded_weighted_cands =  tf.reduce_mean([self.embedded_weighted_cands_crf, self.embedded_weighted_cands_cnn], 0)
            self.embedded_tag_for_category = tf.nn.embedding_lookup(self.W_embedding_for_category, self.input_x_tag)
            self.embedded_tag_for_compete = tf.nn.embedding_lookup(self.W_embedding_for_compete, self.input_x_tag)

        # Sent average layer.
        self.embedded_sent_avg = tf.reduce_mean(self.embedded_sent, 1)
        print("embedded_sent.shape: ", self.embedded_sent.shape)
        print("embedded_sent_avg.shape: ", self.embedded_sent_avg.shape)
        
        # Candidate average layer.
        self.embedded_cands_avg = tf.reduce_mean(self.embedded_weighted_cands, 1)
        print("embedded_cands.shape: ", self.embedded_cands.shape)
        print("embedded_cands_avg.shape: ", self.embedded_cands_avg.shape)
        print("embedded_tag_for_category.shape: ", self.embedded_tag_for_category.shape)
        print("embedded_tag_for_compete.shape: ", self.embedded_tag_for_compete.shape)
        
        # Add dropout
        with tf.name_scope("dropout"):
            self.embedded_sent_avg_dropout = tf.nn.dropout(self.embedded_sent_avg, self.dropout_keep_prob)
            self.embedded_cands_avg_dropout = tf.nn.dropout(self.embedded_cands_avg, self.dropout_keep_prob)
            self.embedded_tag_for_category_dropout = tf.nn.dropout(self.embedded_tag_for_category, self.dropout_keep_prob)
            self.embedded_tag_for_compete_dropout = tf.nn.dropout(self.embedded_tag_for_compete, self.dropout_keep_prob)

        # Output layer
        with tf.name_scope("output"):
            embedded_sent_avg_dropout_norm = tf.nn.l2_normalize(self.embedded_sent_avg_dropout, 1)
            embedded_cands_avg_dropout_norm = tf.nn.l2_normalize(self.embedded_cands_avg_dropout, 1)
            embedded_tag_for_category_dropout_norm = tf.nn.l2_normalize(self.embedded_tag_for_category_dropout, 1)
            embedded_tag_for_compete_dropout_norm = tf.nn.l2_normalize(self.embedded_tag_for_compete_dropout, 1)
            self.prediction_sent_avg = tf.reduce_sum(tf.multiply(embedded_sent_avg_dropout_norm, embedded_tag_for_category_dropout_norm), 1, name="prediction_sent_avg")
            self.prediction_cands_avg = tf.reduce_sum(tf.multiply(embedded_cands_avg_dropout_norm, embedded_tag_for_compete_dropout_norm), 1, name = "prediction_cands_avg")
            self.prediction = tf.reduce_mean([self.prediction_sent_avg, self.prediction_cands_avg], 0, name="predictions")

        print("self.prediction.shape:", self.prediction.shape)
        print("self.input_y.shape:", self.input_y.shape)
        # Mse
#        with tf.name_scope("mse"):
#            self.mse = tf.reduce_mean(tf.square(tf.subtract(self.prediction, self.input_y)), name="mse")

        self.loss_pos = tf.subtract(1.0, self.prediction)
        self.loss_neg = tf.maximum(0.0, tf.subtract(self.prediction, 0.2))
        self.loss_all = tf.where(self.input_y > 0.1, self.loss_pos, self.loss_neg)
        with tf.name_scope("mse"):
            self.mse = tf.reduce_mean(tf.square(self.loss_all), name="mse")

        # Calculate mean perplexity loss
        with tf.name_scope("loss"):
            self.loss = self.mse
