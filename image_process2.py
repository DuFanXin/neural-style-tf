# -*- coding:utf-8 -*-
'''  
#====#====#====#====
# Project Name:     neural-style-tf 
# File Name:        image_process2 
# Date:             1/5/18 8:06 PM 
# Using IDE:        PyCharm Community Edition  
# From HomePage:    https://github.com/DuFanXin/neural-style-tf
# Author:           DuFanXin 
# BlogPage:         http://blog.csdn.net/qq_30239975  
# E-mail:           18672969179@163.com
# Copyright (c) 2018, All Rights Reserved.
#====#====#====#==== 
'''

import tensorflow as tf
import os
import sys
import argparse
# import matplotlib.pyplot as plt
from PIL import Image  # 注意Image,后面会用到
import glob

IMG_WIDE = 128
IMG_HEIGHT = 128
IMG_CHANNEL = 3


def write_img_to_tfrecords():

	train_set_writer = tf.python_io.TFRecordWriter(os.path.join(FLAGS.data_dir, 'train_set.tfrecords'))  # 要生成的文件
	development_set_writer = tf.python_io.TFRecordWriter(os.path.join(FLAGS.data_dir, 'train_set.tfrecords'))  # 要生成的文件
	test_set_writer = tf.python_io.TFRecordWriter(os.path.join(FLAGS.data_dir, 'test_set.tfrecords'))  # 要生成的文件

	files_path = glob.glob(os.path.join(FLAGS.data_dir, '*.jpg'))
	print('files num ' + str(len(files_path)))
	length = len(files_path)
	development_set_size, test_set_size = length >= 100 and (length//100, length//100) or (1, 1)
	train_set_size = length - development_set_size - test_set_size
	# train_set_size = development_set_size = test_set_size = 4
	train_set_path = files_path[:train_set_size]
	development_set_path = files_path[train_set_size:train_set_size + development_set_size]
	test_set_path = files_path[train_set_size + development_set_size:]
	print('train files num ' + str(len(train_set_path)))
	print('train files num ' + str(len(development_set_path)))
	print('test files num ' + str(len(test_set_path)))
	for index, image_path in enumerate(train_set_path):
		img = Image.open(image_path)
		img = img.resize((128, 128))
		img_raw = img.tobytes()  # 将图片转化为二进制格式
		example = tf.train.Example(features=tf.train.Features(feature={
			"label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
			'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
		}))  # example对象对label和image数据进行封装
		train_set_writer.write(example.SerializeToString())  # 序列化为字符串
		print(index)
	train_set_writer.close()
	print('Done train_set write')

	for index, image_path in enumerate(development_set_path):
		img = Image.open(image_path)
		img = img.resize((128, 128))
		img_raw = img.tobytes()  # 将图片转化为二进制格式
		example = tf.train.Example(features=tf.train.Features(feature={
			"label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
			'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
		}))  # example对象对label和image数据进行封装
		train_set_writer.write(example.SerializeToString())  # 序列化为字符串
		print(index)
	development_set_writer.close()
	print('Done development_set write')

	for index, image_path in enumerate(test_set_path):
		img = Image.open(image_path)
		img = img.resize((128, 128))
		img_raw = img.tobytes()  # 将图片转化为二进制格式
		example = tf.train.Example(features=tf.train.Features(feature={
			"label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
			'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
		}))  # example对象对label和image数据进行封装
		test_set_writer.write(example.SerializeToString())  # 序列化为字符串
		print(index)
	test_set_writer.close()
	print('Done test_set write')


def read_img_from_tfrecord_and_save():
	# 读入流中
	filename_queue = tf.train.string_input_producer([os.path.join(FLAGS.data_dir, 'images.tfrecords')])
	reader = tf.TFRecordReader()
	_, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
	features = tf.parse_single_example(serialized_example, features={
							'label': tf.FixedLenFeature([], tf.int64),
							'img_raw': tf.FixedLenFeature([], tf.string),
							})  # 取出包含image和label的feature对象
	image = tf.decode_raw(features['img_raw'], tf.uint8)
	image = tf.reshape(image, [128, 128, 3])
	label = tf.cast(features['label'], tf.int32)
	print(image)
	with tf.Session() as sess:  # 开始一个会话
		sess.run(tf.global_variables_initializer())
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)
		for i in range(6):
			example, la = sess.run([image, label])  # 在会话中取出image和label
			img = Image.fromarray(example, 'RGB')  # 这里Image是之前提到的
			# img.save(str(i) + '_''Label_' + str(la) + '.jpg')  # 存下图片
			# print(example, la)
		coord.request_stop()
		coord.join(threads)
	print("save done")


def read_image(file_queue):
	reader = tf.TFRecordReader()
	# key, value = reader.read(file_queue)
	_, serialized_example = reader.read(file_queue)
	features = tf.parse_single_example(
		serialized_example,
		features={
			'image_raw': tf.FixedLenFeature([], tf.string),
			'label': tf.FixedLenFeature([], tf.int64),
			})

	image = tf.decode_raw(features['image_raw'], tf.uint8)
	# print('image ' + str(image))
	image = tf.reshape(image, [IMG_HEIGHT, IMG_WIDE, IMG_CHANNEL])  # reshape为128*128的3通道图片
	# image.set_shape([IMG_HEIGH * IMG_WIDE * IMG_CHANNEL])
	# image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
	label = tf.cast(features['label'], tf.int32)
	return image, label


def read_image_batch(file_queue, batch_size):
	img, label = read_image(file_queue)
	capacity = 3 * batch_size
	image_batch, label_batch = tf.train.batch([img, label], batch_size=batch_size, capacity=capacity, num_threads=10)
	# one_hot_labels = tf.to_float(tf.one_hot(label_batch, 10, 1, 0))
	one_hot_labels = tf.reshape(label_batch, [batch_size, 1])
	return image_batch, one_hot_labels


def main():
	# train
	train_file_path = os.path.join(FLAGS.data_dir, "train_set.tfrecords")
	# development
	development_file_path = os.path.join(FLAGS.data_dir, "development_set.tfrecords")
	# test
	test_file_path = os.path.join(FLAGS.data_dir, "test_set.tfrecords")
	# check point
	ckpt_path = os.path.join(FLAGS.model_dir, "model.ckpt")

	train_image_filename_queue = tf.train.string_input_producer(
		tf.train.match_filenames_once(train_file_path))
	train_images, train_labels = read_image_batch(train_image_filename_queue, 4)

	development_image_filename_queue = tf.train.string_input_producer(
		tf.train.match_filenames_once(development_file_path))
	development_images, development_labels = read_image_batch(development_image_filename_queue, 1)

	test_image_filename_queue = tf.train.string_input_producer(
			tf.train.match_filenames_once(test_file_path))
	test_images, test_labels = read_image_batch(test_image_filename_queue, 1)

	# x = tf.reshape(train_images, [-1, 784])  # 行数为-1表示未指定，要形成784列的数据，而行数要python自己计算有多少行
	# y = tf.to_float(train_labels)
	print(train_images)
	print(train_labels)
	a = tf.Variable(1.0, 'a')
	with tf.Session() as sess:  # 开始一个会话
		sess.run(tf.global_variables_initializer())
		sess.run(tf.local_variables_initializer())
		print(sess.run(a))
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)
		for i in range(5):
			example, lablel = sess.run([train_images, train_labels])  # 在会话中取出image和label
			print(lablel)
			print('\n')
			# to do minibatch
		# image = Image.fromarray(example, 'RGB')  # 这里Image是之前提到的
		# image.save('Label_' + str(lablel) + '.jpg')  # 存下图片
		coord.request_stop()
		coord.join(threads)
	print("save done")


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	# 输入地址
	parser.add_argument(
		'--data_dir', type=str, default='/home/dufanxin/Documents/Pycharm/image',
		help='input data path')

	# 模型保存地址
	parser.add_argument(
		'--model_dir', type=str, default='',
		help='output model path')

	# 日志地址
	parser.add_argument(
		'--output_dir', type=str, default='',
		help='output data path')
	FLAGS, _ = parser.parse_known_args()
	# print(FLAGS.data_dir)
	# tf.app.run(main=main)
	# write_img_to_tfrecords()
	# read_img_from_tfrecord_and_save()
	main()
