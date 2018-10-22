import tensorflow as tf
import numpy as np


def variable_summaries(var, name):
	with tf.name_scope('summaries'):
		mean = tf.reduce_mean(var)
		tf.summary.scalar( name + '/mean', mean)
		with tf.name_scope('stddev'):
			stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
		tf.summary.scalar( name + '/sttdev' , stddev)
		tf.summary.scalar( name + '/max' , tf.reduce_max(var))
		tf.summary.scalar( name + '/min' , tf.reduce_min(var))
		tf.summary.histogram(name, var)

def fc(tensor, output_dim, name, act=tf.nn.relu):
	with tf.name_scope(name):
		input_dim = tensor.get_shape()[1].value
		Winit = tf.truncated_normal([input_dim, output_dim], stddev=0.1)
		W = tf.Variable(Winit)
		print (name,'input  ',tensor)
		print (name,'W  ',W.get_shape())
		variable_summaries(W, name + '/W')
		Binit = tf.constant(0.0, shape=[output_dim])
		B = tf.Variable(Binit)
		variable_summaries(B, name + '/B')
		tensor = tf.matmul(tensor, W) + B
		tensor = act(tensor)
	return tensor

def conv(tensor, outDim, filterSize, stride, name, act=tf.nn.relu):
	with tf.name_scope(name):
		inDimH = tensor.get_shape()[1].value
		inDimW = tensor.get_shape()[2].value
		inDimD = tensor.get_shape()[3].value
		Winit = tf.truncated_normal([filterSize,filterSize, inDimD,outDim], stddev=0.1)
		W = tf.Variable(Winit)
		print (name,'input  ',tensor)
		print (name,'W  ',W.get_shape())
		variable_summaries(W, name + '/W')
		Binit = tf.constant(0.0, shape=[outDim])
		B = tf.Variable(Binit)
		variable_summaries(B, name + '/B')
		tensor = tf.nn.conv2d(tensor, W, strides=[1, stride, stride, 1], padding='SAME') + B
		tf.summary.image(name+'_Filtre1', tensor[:,:,:,0:1], 5)
		tensor = act(tensor)
	return tensor

def maxpool(tensor, poolSize, name):
	with tf.name_scope(name):
		tensor = tf.nn.max_pool(tensor, ksize=(1,poolSize,poolSize,1), strides=(1,poolSize,poolSize,1), padding='SAME')
	return tensor

def flat(tensor):
	inDimH = tensor.get_shape()[1].value
	inDimW = tensor.get_shape()[2].value
	inDimD = tensor.get_shape()[3].value
	tensor = tf.reshape(tensor, [-1, inDimH * inDimW * inDimD])
	print ('flat output  ',tensor)
	return tensor

def unflat(tensor, outDimH,outDimW,outDimD):
	tensor = tf.reshape(tensor, [-1,outDimH,outDimW,outDimD])
	tf.summary.image('input', tensor, 5)
	print ('unflat output  ',tensor)
	return tensor


def dense(tensor, outDim, name):
	with tf.name_scope(name):
		inDim = tensor.get_shape()[1].value
		Winit = tf.Variable(tf.truncated_normal([inDim, outDim], mean=0, stddev=1 / np.sqrt(inDim)), name='weights1')
		W = tf.Variable(Winit)
		print (name,'input  ',tensor)
		print (name,'W  ',W.get_shape())
		variable_summaries(W, name + '/W')
		Binit = tf.Variable(tf.truncated_normal([outDim],mean=0, stddev=1 / np.sqrt(inDim)), name='bias1')
		B = tf.Variable(Binit)
		variable_summaries(B, name + '/B')
		tensor = tf.nn.tanh((tf.matmul(tensor, W)+B), name='activationLayer1')
	return tensor


def multilayer_perceptron(x):

    W1 = tf.Variable(tf.truncated_normal([n_input, n_neurons_in_h1], mean=0, stddev=1 / np.sqrt(n_input)), name='weights1')
    b1 = tf.Variable(tf.truncated_normal([n_neurons_in_h1],mean=0, stddev=1 / np.sqrt(n_input)), name='biases1')

    y1 = tf.nn.tanh((tf.matmul(X, W1)+b1), name='activationLayer1')

    #network parameters(weights and biases) are set and initialized(Layer2)
    W2 = tf.Variable(tf.random_normal([n_neurons_in_h1, n_neurons_in_h2],mean=0,stddev=1/np.sqrt(n_input)),name='weights2')
    b2 = tf.Variable(tf.random_normal([n_neurons_in_h2],mean=0,stddev=1/np.sqrt(n_input)),name='biases2')
    #activation function(sigmoid)
    y2 = tf.nn.sigmoid((tf.matmul(y1,W2)+b2),name='activationLayer2')

    #output layer weights and biasies
    Wo = tf.Variable(tf.random_normal([n_neurons_in_h2, n_classes], mean=0, stddev=1/np.sqrt(n_input)), name='weightsOut')
    bo = tf.Variable(tf.random_normal([n_classes], mean=0, stddev=1/np.sqrt(n_input)), name='biasesOut')
    #activation function(softmax)
    output_layer = tf.nn.softmax((tf.matmul(y2, Wo) + bo), name='activationOutputLayer')

    return output_layer
