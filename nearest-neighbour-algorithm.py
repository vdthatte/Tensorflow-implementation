import numpy as np
import tensorflow as tf

import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# limiting the mnsit data
Xtr, Ytr = mnist.train.next_batch(5000) #5000 for training (nn candidates)
Xte, Yte = mnist.test.next_batch(200) # 200 for testing

# reshaping the images to ID
Xtr = np.reshape(Xtr, newshape=(-1, 28*28))
Xte = np.reshape(Xte, newshape=(-1, 28*28))

# tf Graph Input
xtr = tf.placeholder("float", [None, 784])
xte = tf.placeholder("float", [785])

distance = tf.reduce_sum(tf.abs(tf.add(xtr, tf.neg(xte))), reduction_indices=1)

pred = tf.arg_min(distance,0)
accuracy = 0

init = tf.initialize_all_variables()

with tf.Session() as sess:
	for i in range(len(Xte)):
		nn_index = sess.run(pred, feed_dict={xtr: Xtr, xte: Xte[i,:]})
		print "Test", i, "Prediction:", np.argmax(Ytr[nn_index]), \
		"True Class:", np.argmax(Yte[i])

		#calculate the accuracy
		if np.argmax(Ytr[nn_index]) == np.argmax(Yte[i]):
			accuracy += 1./len(Xte)

	print "THE END"
	print "Accuracy:", accuracy