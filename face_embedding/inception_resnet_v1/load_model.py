import os
import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
import cv2
import multiprocessing as mp
import time
from functools import wraps

def timer(function):
	@wraps(function)
	def function_timer(*args, **kwargs):
		start = time.time()
		result = function(*args, **kwargs)
		end = time.time()
		print(f"{function.__name__}:: {end-start} s")
		return result
	return function_timer


class InceptionResnet(object):

	def __init__(self, gpu_memory_fraction_to_use=0.5):
		self.embedding_graph = tf.Graph()
		cores = mp.cpu_count()
		with self.embedding_graph.as_default():
			tf_config = tf.compat.v1.ConfigProto()
			if tf.test.is_gpu_available() and tf.test.is_built_with_cuda():
				# fraction of GPU memory to use
				gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction_to_use)
				tf_config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
				tf_config.gpu_options.allow_growth = True
			# no of physical cpu cores to allocate
			tf_config.intra_op_parallelism_threads = cores
			# no of physical cpu cores to allocate
			tf_config.intra_op_parallelism_threads = cores

			with gfile.GFile(os.path.join(os.path.dirname(os.path.abspath(__file__)), \
								'model/inception_resnet_v1_128d.pb'), 'rb') as f:
				graph_def = tf.compat.v1.GraphDef()
				graph_def.ParseFromString(f.read())
				tf.import_graph_def(graph_def, input_map=None, name='')
				# Get input and output tensors
				self.images_placeholder = self.embedding_graph.get_tensor_by_name("input:0")
				self.embeddings = self.embedding_graph.get_tensor_by_name("embeddings:0")
				self.phase_train_placeholder = self.embedding_graph.get_tensor_by_name("phase_train:0")

				self.sess = tf.compat.v1.Session(graph=self.embedding_graph, config=tf_config)
	
	def prewhiten(self, x):
		"""Normalzing the face patch"""
		mean = np.mean(x)
		std = np.std(x)
		std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
		y = np.multiply(np.subtract(x, mean), 1 / std_adj)
		return y

	@timer
	def get_embeddings(self, img, bbs):
		"""Calculating the embedding for a face patch"""
		face_patches = []
		for left, top, right, bottom in bbs:
			# cropping the face patch from the frame
			face_patch = img[top:bottom, left:right, :]
			# resizing the cropped image to the input of the model
			resized = cv2.resize(face_patch, (160, 160), interpolation=cv2.INTER_LINEAR)
			# normalizing the image
			prewhitened = self.prewhiten(resized)
			face_patches.append(prewhitened)

		if len(face_patches):
			# stack face patches to form a single batch of input
			face_patches = np.stack(face_patches)
			# reshaping face patch based on the model input
			reshaped = face_patches.reshape(-1, 160, 160, 3)
			# feeding input  to the model
			feed_dict = {self.images_placeholder: reshaped, self.phase_train_placeholder: False}
			embeddings = self.sess.run(self.embeddings, feed_dict=feed_dict)
			return embeddings
		return face_patches
	
	@timer
	def get_embeddings_from_face_patches(self, face_patches):
		"""Calculating the embedding for a face patch"""
		_face_patches = []
		for face_patch in face_patches:
			# resizing the cropped image to the input of the model
			resized = cv2.resize(face_patch, (160, 160), interpolation=cv2.INTER_LINEAR)
			# normalizing the image
			prewhitened = self.prewhiten(resized)
			_face_patches.append(prewhitened)

		if len(_face_patches):
			# stack face patches to form a single batch of input
			_face_patches = np.stack(_face_patches)
			# reshaping face patch based on the model input
			reshaped = _face_patches.reshape(-1, 160, 160, 3)
			# feeding input  to the model
			feed_dict = {self.images_placeholder: reshaped, self.phase_train_placeholder: False}
			embeddings = self.sess.run(self.embeddings, feed_dict=feed_dict)
			return embeddings
		return []
