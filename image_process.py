# -*- coding:utf-8 -*-
'''  
#====#====#====#====
# Project Name:     neural-style-tf 
# File Name:        image_process 
# Date:             2017/12/31 16:56 
# Using IDE:        PyCharm  
# From HomePage:    https://github.com/DuFanXin/neural-style-tf
# Author:           DuFanXin 
# BlogPage:         http://blog.csdn.net/qq_30239975  
# E-mail:           18672969179@163.com
# Copyright (c) 2017, All Rights Reserved.
#====#====#====#==== 
'''


# from __future__ import absolute_import,division,print_function


import tensorflow as tf
import os
import sys
import argparse
import matplotlib.pyplot as plt
from PIL import Image  # 注意Image,后面会用到
import glob


def write_img_to_tfrecords():

	writer = tf.python_io.TFRecordWriter(os.path.join(FLAGS.data_dir, 'images.tfrecords'))  # 要生成的文件
	for index, image_path in enumerate(glob.glob(os.path.join(FLAGS.data_dir, '*.jpg'))):
		img = Image.open(image_path)
		img = img.resize((128, 128))
		img_raw = img.tobytes()  # 将图片转化为二进制格式
		example = tf.train.Example(features=tf.train.Features(feature={
			"label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
			'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
		}))  # example对象对label和image数据进行封装
		writer.write(example.SerializeToString())  # 序列化为字符串
	writer.close()
	print('write done')


def read_img_from_tfrecord_and_use():
	# 生成一个queue队列
	filename_queue = tf.train.string_input_producer([os.path.join(FLAGS.data_dir, 'images.tfrecords')])
	reader = tf.TFRecordReader()
	_, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
	# 将image数据和label取出来
	features = tf.parse_single_example(serialized_example, features={
							'label': tf.FixedLenFeature([], tf.int64),
							'img_raw': tf.FixedLenFeature([], tf.string),
							})
	img = tf.decode_raw(features['img_raw'], tf.uint8)
	img = tf.reshape(img, [128, 128, 3])  # reshape为128*128的3通道图片
	img = tf.cast(img, tf.float32) * (1. / 255) - 0.5  # 在流中抛出img张量
	label = tf.cast(features['label'], tf.int32)  # 在流中抛出label张量
	print(img)
	print(label)
	return img, label


def save_img_as_result(img=None):
	writer = tf.python_io.TFRecordWriter(os.path.join(FLAGS.data_dir, 'test.tfrecords'))  # 要生成的文件
	with tf.Session() as sess:  # 开始一个会话
		sess.run(tf.global_variables_initializer())
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)
		img = tf.cast(img, tf.uint8)
		img_arry = sess.run(img)
		img_raw = img_arry.tostring()
		example = tf.train.Example(features=tf.train.Features(feature={
			"label": tf.train.Feature(int64_list=tf.train.Int64List(value=[1])),
			'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
		}))  # example对象对label和image数据进行封装
		writer.write(example.SerializeToString())  # 序列化为字符串
		writer.close()
		coord.request_stop()
		coord.join(threads)
	print("Done save image as result")


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
	with tf.Session() as sess:  # 开始一个会话
		sess.run(tf.global_variables_initializer())
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)
		for i in range(6):
			example, la = sess.run([image, label])  # 在会话中取出image和label
			img = Image.fromarray(example, 'RGB')  # 这里Image是之前提到的
			img.save(str(i) + '_''Label_' + str(la) + '.jpg')  # 存下图片
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
	image.set_shape([784])
	image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
	label = tf.cast(features['label'], tf.int32)
	return image, label


def read_image_batch(file_queue, batch_size):
	img, label = read_image(file_queue)
	capacity = 3 * batch_size
	image_batch, label_batch = tf.train.batch([img, label], batch_size=batch_size, capacity=capacity, num_threads=10)
	one_hot_labels = tf.to_float(tf.one_hot(label_batch, 10, 1, 0))
	return image_batch, one_hot_labels


def main(_):
	#
	train_file_path = os.path.join(FLAGS.data_dir, "train.tfrecords")
	#
	test_file_path = os.path.join(FLAGS.data_dir, "test.tfrecords")
	#
	ckpt_path = os.path.join(FLAGS.model_dir, "model.ckpt")

	train_image_filename_queue = tf.train.string_input_producer(
			tf.train.match_filenames_once(train_file_path))
	train_images, train_labels = read_image_batch(train_image_filename_queue, 100)

	test_image_filename_queue = tf.train.string_input_producer(
			tf.train.match_filenames_once(test_file_path))
	test_images, test_labels = read_image_batch(test_image_filename_queue, 100)

	x = tf.reshape(train_images, [-1, 784])  # 行数为-1表示未指定，要形成784列的数据，而行数要python自己计算有多少行
	y = tf.to_float(train_labels)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	# 输入地址
	parser.add_argument(
		'--data_dir', type=str, default='C:\\Users\\yzc\\Desktop\\image',
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
	read_img_from_tfrecord_and_save()
# main()
