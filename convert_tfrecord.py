from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import random
import csv
import os
import sys
import time

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('input_dir', None, 'The directory of the image files.')
tf.app.flags.DEFINE_string('csv_name', 'list.csv', 'The CSV file name.')
#tf.app.flags.DEFINE_string('format', 'jpg', 'The input image format')
tf.app.flags.DEFINE_string('labels_filename', 'labels.txt', '')
tf.app.flags.DEFINE_integer('num_data', 500, '')
tf.app.flags.DEFINE_integer('validations', 2500, 'The num of validation images')
tf.app.flags.DEFINE_string('dataset_name', 'dataset', '')
 

def read_csv_with_shuffle(root, filename):
	data = []
	path = os.path.join(root, filename)
	with open(path, 'r') as f:
		for row in csv.reader(f):
			file_path = os.path.join(root, row[0].strip())
			data.append([file_path, row[1].strip()])
	random.shuffle(data)

	filepaths = [path for inner in data for path in inner[::2]]
	labels = [label for inner in data for label in inner[1::2]]
	return filepaths, labels

def label_to_integer(labels):
	label_to_id = [] 
	for i in xrange(len(labels)):
		if labels[i] not in label_to_id:
			label_to_id.append(labels[i])
		labels[i] = label_to_id.index(labels[i])
	return labels, label_to_id
	
def convert_tfrecord_and_write(filepaths, labels, units, validations):
	trains_end = len(filepaths) // units * units
	train_filepaths = zip(*[iter(filepaths[validations:trains_end])]*units)
	train_labels = zip(*[iter(labels[validations:trains_end])]*units)
	validation_filepaths = zip(*[iter(filepaths[0:validations])]*units)
	validation_labels = zip(*[iter(labels[0:validations])]*units)
	
	write_tfrecord('train', train_filepaths, train_labels)
	write_tfrecord('validation', validation_filepaths, validation_labels)

def _int64_feature(values):
	if not isinstance(values, (tuple, list)):
		values = [values]
	return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def _bytes_feature(values):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

def image_to_tfexample(image_data, image_format, height, width, class_id):
	return tf.train.Example(features=tf.train.Features(feature={
		'image/encoded': _bytes_feature(image_data),
		'image/format': _bytes_feature(image_format),
		'image/class/label': _int64_feature(class_id),
		'image/height': _int64_feature(height),
		'image/width': _int64_feature(width)}))

def write_tfrecord(split_name, filepath_lists, label_lists):
	jpeg_path = tf.placeholder(dtype=tf.string)
	jpeg_data = tf.read_file(jpeg_path)
	decode_jpeg = tf.image.decode_jpeg(jpeg_data, channels=3)

	with tf.Session() as sess:
		for i, filepath_list in enumerate(filepath_lists):
			output_filename = '%s_%s_%05d-of-%05d.tfrecord'%(FLAGS.dataset_name, split_name, i, len(filepath_lists))
			with tf.python_io.TFRecordWriter(output_filename) as writer:
				for j,filepath in enumerate(filepath_list):
					sys.stdout.write('\r>> Converting image %d/%d'%(j+1, len(filepath_list)))
					sys.stdout.flush()
					image_data, image = sess.run([jpeg_data, decode_jpeg], feed_dict={jpeg_path:filepath})
					example = image_to_tfexample(image_data, 'jpg', image.shape[0], image.shape[1], label_lists[i][j])
					writer.write(example.SerializeToString())
			print(' Finished: %s'%(output_filename))

def write_label_map(label_list, output_filename):
	with tf.gfile.Open(output_filename, 'w') as f:
		for i, label in enumerate(label_list):
			f.write('%d:%s\n'%(i,label))
	
def main(_):
	if not FLAGS.input_dir:
		raise ValueError('You must supply the image directory with --input_dir')
	if not FLAGS.validations % FLAGS.num_data == 0 :
		raise ValueError('<validations> must be divisible by <num_data>')

	filepaths, labels = read_csv_with_shuffle(FLAGS.input_dir, FLAGS.csv_name)
	labels, label_to_id = label_to_integer(labels)

	convert_tfrecord_and_write(filepaths, labels, FLAGS.num_data, FLAGS.validations)
	write_label_map(label_to_id, FLAGS.labels_filename)

if __name__ == '__main__':
	tf.app.run()
