import tensorflow as tf
import numpy as np
import Layers
import Data_util

LoadModel = False

experiment_name = 'salary'

data = Data_util.read_data("data/adult.data")
training_data, training_labels = Data_util.class2vect(data)

split_ratio = 0.8
X_train, X_test = training_data[:int(len(training_data)*split_ratio)], training_data[int(len(training_data)*split_ratio):]
y_train, y_test = training_labels[:int(len(training_labels)*split_ratio)], training_labels[int(len(training_labels)*split_ratio):]


with tf.name_scope('input'):
    X = tf.placeholder(tf.float32, [None, len(X_train[0])], name='features')
    Y = tf.placeholder(tf.float32, [None, 2], name='labels')


with tf.name_scope('MLP'):
	t = Layers.dense(X,480,'layer_1')
	t = Layers.dense(t,300,'layer_2')
	t = Layers.dense(t,200,'layer_3')
	t = Layers.dense(t,100,'layer_4')
	t = Layers.dense(t,50,'layer_5')
	t = Layers.dense(t,25,'layer_6')
	y = Layers.fc(t,2,'fc',tf.nn.tanh)

with tf.name_scope('cross_entropy'):
	diff = Y * y
	with tf.name_scope('total'):
		cross_entropy = -tf.reduce_mean(diff)
	tf.summary.scalar('cross entropy', cross_entropy)

with tf.name_scope('accuracy'):
	with tf.name_scope('correct_prediction'):
		correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(Y, 1))
	with tf.name_scope('accuracy'):
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	tf.summary.scalar('accuracy', accuracy)

with tf.name_scope('learning_rate'):
	global_step = tf.Variable(0, trainable=False)
	learning_rate = tf.train.exponential_decay(0.0001,global_step,1000, 0.75, staircase=True)


with tf.name_scope('learning_rate'):
    tf.summary.scalar('learning_rate', learning_rate)

#train_step = tf.train.GradientDescentOptimizer(0.00001).minimize(cross_entropy)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy,global_step=global_step)
merged = tf.summary.merge_all()

Acc_Train = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Acc_train");
Acc_Test = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Acc_test");
MeanAcc_summary = tf.summary.merge([tf.summary.scalar('Acc_Train', Acc_Train),tf.summary.scalar('Acc_Test', Acc_Test)])

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")

print ("-----------------------------------------------------")
print ("-----------",experiment_name)
print ("-----------------------------------------------------")


sess = tf.Session()
sess.run(tf.global_variables_initializer())
writer = tf.summary.FileWriter(experiment_name, sess.graph)
saver = tf.train.Saver()
if LoadModel:
	saver.restore(sess, "./save/model.ckpt")

nbIt = 5000
batchsize = 100
for it in range(nbIt):
	for i in range(0,len(X_train), batchsize):

		start=i
		end=i+batchsize
		x_batch=X_train[start:end]
		y_batch=y_train[start:end]
		sess.run(train_step, feed_dict={X:x_batch , Y:y_batch})
		'''if i%100 == 0:
			acc,ce = sess.run([accuracy,cross_entropy], feed_dict={X:x_batch , Y:y_batch})
			print ("it= %6d - cross_entropy= %.4f - acc= %.4f" % (it,ce,acc ))
			summary_merged = sess.run(merged, feed_dict={X:x_batch , Y:y_batch})
			writer.add_summary(summary_merged, it)
		'''
	if it%10 == 0:
		Acc_Train_value = sess.run([accuracy], feed_dict={X: X_train, Y: y_train })[0]#,keep_prob:1.0})
		Acc_Test_value = sess.run([accuracy], feed_dict={X: X_test, Y: y_test })[0]#,keep_prob:1.0})
		print ("it: %d, mean accuracy train = %.4f  test = %.4f" % (it,Acc_Train_value,Acc_Test_value ))
		summary_acc = sess.run(MeanAcc_summary, feed_dict={Acc_Train:Acc_Train_value,Acc_Test:Acc_Test_value})
		writer.add_summary(summary_acc, it)

writer.close()
if not LoadModel:
	saver.save(sess, "./save/model.ckpt")
sess.close()
