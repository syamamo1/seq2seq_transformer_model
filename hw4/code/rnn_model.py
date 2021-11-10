import numpy as np
import tensorflow as tf

class RNN_Seq2Seq(tf.keras.Model):
	def __init__(self, french_window_size, french_vocab_size, english_window_size, english_vocab_size):
		###### DO NOT CHANGE ##############
		super(RNN_Seq2Seq, self).__init__()
		self.french_vocab_size = french_vocab_size # The size of the french vocab
		self.english_vocab_size = english_vocab_size # The size of the english vocab

		self.french_window_size = french_window_size # The french window size
		self.english_window_size = english_window_size # The english window size
		######^^^ DO NOT CHANGE ^^^##################


		# TODO:
		# 1) Define any hyperparameters

		# Define batch size and optimizer/learning rate
		self.batch_size = 100 # You can change this
		self.embedding_size = 128 # You should change this
	
		# 2) Define embeddings, encoder, decoder, and feed forward layers
		self.french_E = tf.Variable(tf.random.truncated_normal([self.french_vocab_size, self.embedding_size], stddev=.1))
		self.english_E = tf.Variable(tf.random.truncated_normal([self.english_vocab_size, self.embedding_size], stddev=.1))

		self.encoder_rnn = tf.keras.layers.LSTM(256, return_sequences=True, return_state=True)
		self.decoder_rnn = tf.keras.layers.LSTM(256, return_sequences=True, return_state=True)
		self.linear_layer1 = tf.keras.layers.Dense(256, activation='relu')
		self.linear_layer2 = tf.keras.layers.Dense(self.english_vocab_size, activation='softmax')

		self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

	@tf.function
	def call(self, encoder_input, decoder_input):
		"""
		:param encoder_input: batched ids corresponding to french sentences
		:param decoder_input: batched ids corresponding to english sentences
		:return prbs: The 3d probabilities as a tensor, [batch_size x window_size x english_vocab_size]
		"""
	
		# TODO:
		#1) Pass your french sentence embeddings to your encoder 
		french_embedding = tf.nn.embedding_lookup(self.french_E, encoder_input)
		whole_seq_output1, final_memory_state1, final_carry_state1 = self.encoder_rnn(french_embedding, initial_state=None)
		#2) Pass your english sentence embeddings, and final state of your encoder, to your decoder
		english_embedding = tf.nn.embedding_lookup(self.english_E, decoder_input)
		whole_seq_output2, final_memory_state2, final_carry_state2 = self.decoder_rnn(english_embedding, [final_memory_state1, final_carry_state1])
		#3) Apply dense layer(s) to the decoder out to generate probabilities
		linear1output = self.linear_layer1(whole_seq_output2) # probs [batch_size, window_size, units=vocab_size]
		probs = self.linear_layer2(linear1output)

		return probs

	def accuracy_function(self, prbs, labels, mask):
		"""
		DO NOT CHANGE

		Computes the batch accuracy
		
		:param prbs:  float tensor, word prediction probabilities [batch_size x window_size x english_vocab_size]
		:param labels:  integer tensor, word prediction labels [batch_size x window_size]
		:param mask:  tensor that acts as a padding mask [batch_size x window_size]
		:return: scalar tensor of accuracy of the batch between 0 and 1
		"""

		decoded_symbols = tf.argmax(input=prbs, axis=2)
		accuracy = tf.reduce_mean(tf.boolean_mask(tf.cast(tf.equal(decoded_symbols, labels), dtype=tf.float32),mask))
		return accuracy


	def loss_function(self, prbs, labels, mask):
		"""
		Calculates the total model cross-entropy loss after one forward pass. 
		Please use reduce sum here instead of reduce mean to make things easier in calculating per symbol accuracy.
		
		:param prbs:  float tensor, word prediction probabilities [batch_size x window_size x english_vocab_size]
		:param labels:  integer tensor, word prediction labels [batch_size x window_size]
		:param mask:  tensor that acts as a padding mask [batch_size x window_size]
		:return: the loss of the model as a tensor
		"""
		# Using boolean mask
		loss = tf.keras.losses.sparse_categorical_crossentropy(labels, prbs)
		sum_loss = tf.reduce_sum(tf.boolean_mask(loss, mask))

		return sum_loss		

