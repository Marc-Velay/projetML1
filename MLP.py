import tensorflow as tf
import numpy as np
import Layers
import Data_util
from sklearn import model_selection
import prince


experiment_name = 'salary'
LoadModel = False

data = Data_util.read_data("data/adult.data")
data = Data_util.normalise_data_pandas(data, ["Age", "fnlwgt", "Education-Num", "Capital Gain", "Capital Loss", "Hours per week"])
training_data, training_labels = Data_util.class2vect(data)
#training_data = Data_util.normalise_data_np(np.array(training_data))
#split_ratio = 0.8
#X_train, X_test = training_data[:int(len(training_data)*split_ratio)], training_data[int(len(training_data)*split_ratio):]
#y_train, y_test = training_labels[:int(len(training_labels)*split_ratio)], training_labels[int(len(training_labels)*split_ratio):]

X_train, X_test, y_train, y_test = model_selection.train_test_split(training_data, training_labels, train_size=0.7, test_size=0.3)
'''
pca = prince.PCA(   n_components=90,
                    n_iter=3,
                    copy=True,
                    rescale_with_mean=True,
                    rescale_with_std=True,
                    engine='auto',
                    random_state=42)

pca = pca.fit(X_train)

X_train = np.array(pca.row_coordinates(X_train))
X_test = np.array(pca.row_coordinates(X_test))
'''

with tf.name_scope('input'):
    X = tf.placeholder(tf.float32, [None, len(X_train[0])], name='features')
    Y = tf.placeholder(tf.float32, [None, 2], name='labels')


with tf.name_scope('MLP'):
    t = Layers.dense(X,2500,'layer_1')
    #t = Layers.dense(t,4096,'layer_1c')
    #t = Layers.dense(t,2048,'layer_1b')
    t = Layers.dense(t,1024,'layer_1a')
    t = Layers.dense(t,512,'layer_2a')
    t = Layers.dense(t,128,'layer_2b')
    y = Layers.fc(t,2,'fc',tf.nn.tanh)


with tf.name_scope('cross_entropy'):
    #diff = Y * y
    classes_weights = tf.constant([1., 0.652])
    with tf.name_scope('total'):
        #cross_entropy = -tf.reduce_mean(diff)
        cross_entropy = tf.nn.weighted_cross_entropy_with_logits(targets=Y, logits=y, pos_weight=classes_weights)
    tf.summary.scalar('cross entropy', cross_entropy)

with tf.name_scope('accuracy'):
	with tf.name_scope('correct_prediction'):
		correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(Y, 1))
	with tf.name_scope('accuracy'):
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	tf.summary.scalar('accuracy', accuracy)

with tf.name_scope('learning_rate'):
	global_step = tf.Variable(0, trainable=False)
	learning_rate = tf.train.exponential_decay(0.00001,global_step,1000, 0.75, staircase=True)


with tf.name_scope('learning_rate'):
    tf.summary.scalar('learning_rate', learning_rate)

train_step = tf.train.AdamOptimizer(0.00001).minimize(cross_entropy)
#train_step = tf.train.GradientDescentOptimizer(0.00001).minimize(cross_entropy)
#train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy,global_step=global_step)
merged = tf.summary.merge_all()

Acc_Train = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Acc_train");
Acc_Test = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Acc_test");
MeanAcc_summary = tf.summary.merge([tf.summary.scalar('Acc_Train', Acc_Train),tf.summary.scalar('Acc_Test', Acc_Test)])

with tf.name_scope('confusion_matrix'):
    conf_matrix = tf.confusion_matrix(tf.argmax(Y, 1), tf.argmax(y, 1), 2)

print ("-----------------------------------------------------")
print ("-----------",experiment_name)
print ("-----------------------------------------------------")


sess = tf.Session()
sess.run(tf.global_variables_initializer())
writer = tf.summary.FileWriter(experiment_name, sess.graph)
saver = tf.train.Saver()
#if LoadModel:
	#saver.restore(sess, "./save/model.ckpt")

nbIt = 200
batchsize = 128
for it in range(nbIt):
    for i in range(0,len(X_train), batchsize):
        start=i
        end=i+batchsize
        x_batch=X_train[start:end]
        y_batch=y_train[start:end]

        sess.run(train_step, feed_dict={X:x_batch , Y:y_batch})
    if it%10 == 0:
        Acc_Train_value = sess.run([accuracy], feed_dict={X: X_train, Y: y_train })[0]#,keep_prob:1.0})
        Acc_Test_value = sess.run([accuracy], feed_dict={X: X_test, Y: y_test })[0]#,keep_prob:1.0})
        print ("epoch: %d, mean accuracy train = %.8f  test = %.8f" % (it,Acc_Train_value,Acc_Test_value ))
        confusion_matrix = sess.run(conf_matrix, feed_dict={X: X_train, Y: y_train})

        print(confusion_matrix)
        summary_acc = sess.run(MeanAcc_summary, feed_dict={Acc_Train:Acc_Train_value,Acc_Test:Acc_Test_value})
        writer.add_summary(summary_acc, it)

writer.close()
if not LoadModel:
	saver.save(sess, "./save/model.ckpt")
sess.close()
