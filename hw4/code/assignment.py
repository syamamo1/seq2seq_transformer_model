import os
import numpy as np
import tensorflow as tf
import numpy as np
from preprocess import *
from transformer_model import Transformer_Seq2Seq
from rnn_model import RNN_Seq2Seq
import sys
import random
import time

from attenvis import AttentionVis
av = AttentionVis()

def train(model, train_french, train_english, eng_padding_index):
	"""
	Runs through one epoch - all training examples.

	:param model: the initialized model to use for forward and backward pass
	:param train_french: french train data (all data for training) of shape (num_sentences, 14)
	:param train_english: english train data (all data for training) of shape (num_sentences, 15)
	:param eng_padding_index: the padding index, the id of *PAD* token. This integer is used when masking padding labels.
	:return: None
	"""

	# NOTE: For each training step, you should pass in the french sentences to be used by the encoder, 
	# and english sentences to be used by the decoder
	# - The english sentences passed to the decoder have the last token in the window removed:
	#	 [STOP CS147 is the best class. STOP *PAD*] --> [STOP CS147 is the best class. STOP] 
	# 
	# - When computing loss, the decoder labels should have the first word removed:
	#	 [STOP CS147 is the best class. STOP] --> [CS147 is the best class. STOP] 
	start_time = time.time()
	decoder_in = train_english[:,:-1] # remove last PAD token
	decoder_ans = train_english[:,1:] # remove START token

	num_iterations = len(train_french)//model.batch_size
	for i in range(0, num_iterations):
		encoder_inputs = train_french[i*model.batch_size:(i+1)*model.batch_size]
		decoder_inputs = decoder_in[i*model.batch_size:(i+1)*model.batch_size]
		decoder_labels = decoder_ans[i*model.batch_size:(i+1)*model.batch_size]

		mask = tf.cast(tf.not_equal(decoder_labels, eng_padding_index), tf.float32)
		
		with tf.GradientTape() as tape:
			probs = model.call(encoder_inputs, decoder_inputs)
			loss = model.loss_function(probs, decoder_labels, mask)
			accuracy = model.accuracy_function(probs, decoder_labels, mask)
			print("{}% training complete - Accuracy: {} - Train Time: {}".format(round(100*i/num_iterations, 3), accuracy, (time.time()-start_time)/60))

		gradients = tape.gradient(loss, model.trainable_variables)
		model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

	return None

@av.test_func
def test(model, test_french, test_english, eng_padding_index):
	"""
	Runs through one epoch - all testing examples.

	:param model: the initialized model to use for forward and backward pass
	:param test_french: french test data (all data for testing) of shape (num_sentences, 14)
	:param test_english: english test data (all data for testing) of shape (num_sentences, 15)
	:param eng_padding_index: the padding index, the id of *PAD* token. This integer is used when masking padding labels.
	:returns: a tuple containing at index 0 the perplexity of the test set and at index 1 the per symbol accuracy on test set, 
	e.g. (my_perplexity, my_accuracy)
	"""

	# Note: Follow the same procedure as in train() to construct batches of data!
	start_time = time.time()
	decoder_in = test_english[:,:-1] # remove last PAD token
	decoder_ans = test_english[:,1:] # remove START token

	num_iterations = len(test_french)//model.batch_size
	total_loss = 0.0
	total_accuracy = 0.0
	total_words = 0.0
	for i in range(0, num_iterations):
		encoder_inputs = test_french[i*model.batch_size:(i+1)*model.batch_size]
		decoder_inputs = decoder_in[i*model.batch_size:(i+1)*model.batch_size]
		decoder_labels = tf.cast(decoder_ans[i*model.batch_size:(i+1)*model.batch_size], tf.int64)
		
		probs = model.call(encoder_inputs, decoder_inputs)
		mask = tf.cast(tf.not_equal(decoder_labels, eng_padding_index), tf.float32)

		loss = model.loss_function(probs, decoder_labels, mask)
		total_loss += loss

		words = tf.cast(tf.reduce_sum(mask), tf.float32)
		accuracy = model.accuracy_function(probs, decoder_labels, mask)
		total_accuracy += words*accuracy
		print("{}% Testing complete - Accuracy: {} - Test Time: {}".format(round(100*i/num_iterations, 3), accuracy, (time.time()-start_time)/60))
		total_words += words

	print('perplexity', np.exp(total_loss/total_words))
	print('accuracy per', total_accuracy/total_words)
	return np.exp(total_loss/total_words), total_accuracy/total_words


def main():	
	if len(sys.argv) != 2 or sys.argv[1] not in {"RNN","TRANSFORMER"}:
			print("USAGE: python assignment.py <Model Type>")
			print("<Model Type>: [RNN/TRANSFORMER]")
			exit()

	# Change this to "True" to turn on the attention matrix visualization.
	# You should turn this on once you feel your code is working.
	# Note that it is designed to work with transformers that have single attention heads.
	if sys.argv[1] == "TRANSFORMER":
		av.setup_visualization(enable=False)

	print("Running preprocessing...")
	train_english, test_english, train_french, test_french, english_vocab, french_vocab, eng_padding_index = get_data('../../data/fls.txt','../../data/els.txt','../../data/flt.txt','../../data/elt.txt')
	print("Preprocessing complete.")

	model_args = (FRENCH_WINDOW_SIZE, len(french_vocab), ENGLISH_WINDOW_SIZE, len(english_vocab))
	if sys.argv[1] == "RNN":
		model = RNN_Seq2Seq(*model_args)
	elif sys.argv[1] == "TRANSFORMER":
		model = Transformer_Seq2Seq(*model_args) 
	
	# TODO:
	# Train and Test Model for 1 epoch.
	train(model, train_french, train_english, eng_padding_index)
	test(model, test_french, test_english, eng_padding_index)

	# Visualize a sample attention matrix from the test set
	# Only takes effect if you enabled visualizations above
	av.show_atten_heatmap()
	pass

if __name__ == '__main__':
	main()
