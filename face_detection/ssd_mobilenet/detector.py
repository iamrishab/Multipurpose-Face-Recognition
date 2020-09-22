#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Rishab Pal'

import io
import cv2
import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
import os
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


class SSDMobilentFaceDetector(object):
	"""Tensorflow face detector
	""" 
	
	def __init__(self, gpu_memory_fraction_to_use=0.5):
		"""Tensorflow detector
		"""
		super(SSDMobilentFaceDetector, self).__init__()
		self.detection_graph = tf.Graph()
		cores = mp.cpu_count()
		
		with self.detection_graph.as_default():
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

			od_graph_def = tf.compat.v1.GraphDef()
			with tf.io.gfile.GFile(os.path.join(os.path.dirname(os.path.abspath(__file__)), \
									'data/models/frozen_inference_graph_face.pb'), 'rb') as fid:
				serialized_graph = fid.read()
				od_graph_def.ParseFromString(serialized_graph)
				tf.import_graph_def(od_graph_def, name='')

			self.sess = tf.compat.v1.Session(graph=self.detection_graph, config=tf_config)

	def forward(self, image):
		"""image: bgr image
		return (boxes, scores, classes, num_detections)
		"""
		# the array based representation of the image will be used later in order to prepare the
		# result image with boxes and labels on it.
		# Expand dimensions since the model expects images to have shape: [1, None, None, 3]
		image_np_expanded = np.expand_dims(image, axis=0)
		image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
		# Each box represents a part of the image where a particular object was detected.
		boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
		# Each score represent how level of confidence for each of the objects.
		# Score is shown on the result image, together with the class label.
		scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
		classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
		num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
		# Actual detection.
		(boxes, scores, classes, num_detections) = self.sess.run(
			[boxes, scores, classes, num_detections],
			feed_dict={image_tensor: image_np_expanded})
		return boxes, scores, classes, num_detections

	@timer
	def _detect_faces(self, img, confidence_threshold=0.8, padding_ratio=0.15, min_face_size=20):
		"""
			detect and process face boxes
		"""
		h, w = img.shape[:2]
		bbs = []
		(boxes, scores, classes, num_detections) = self.forward(img)
		boxes = np.squeeze(boxes)
		classes = np.squeeze(classes).astype(np.int32)
		scores = np.squeeze(scores)
		for i in range(boxes.shape[0]):
			if scores[i] >= confidence_threshold:
				ymin, xmin, ymax, xmax = boxes[i].tolist()
				# de-normalizing the face boxes
				left, right, top, bottom = xmin * w, xmax * w, ymin * h, ymax * h
				# constraing the boxes to lie inside the frame
				left, top, right, bottom = max(0, left), max(0, top), min(right, w), min(bottom, h)
				# calculating the face width and height
				face_w, face_h = right - left, bottom - top
				# checking if the face lie within the minimum size range
				if face_w >= min_face_size and face_h >= min_face_size:
					# calculating the padding size based on the above
					face_pad_w, face_pad_h = face_w*padding_ratio, face_h*padding_ratio
					# adding the padding
					left -= face_pad_w
					top -= face_pad_h 
					right += face_pad_w
					bottom += face_pad_h
					# adjust the padded face boxes so that if it does not exceed frame size
					left, top, right, bottom = max(0, int(left)), max(0, int(top)), min(w, int(right)), min(h, int(bottom))
					bbs.append([left, top, right, bottom])
		return bbs
	
	@timer
	def detect_faces(self, img, confidence_threshold=0.8, padding_ratio=0.15, min_face_size=20, scale=5):
		"""
			detect and process face boxes
		"""
		h, w = img.shape[:2]
		r_h, r_w = int(h // scale), int(w // scale)
		
		frame_resized = cv2.resize(img.copy(), (r_w, r_h), interpolation = cv2.INTER_AREA)
		
		bbs = []
		(boxes, scores, classes, num_detections) = self.forward(frame_resized)
		boxes = np.squeeze(boxes)
		classes = np.squeeze(classes).astype(np.int32)
		scores = np.squeeze(scores)
		for i in range(boxes.shape[0]):
			if scores[i] >= confidence_threshold:
				ymin, xmin, ymax, xmax = boxes[i].tolist()
				# de-normalizing the face boxes
				left, right, top, bottom = xmin * r_w, xmax * r_w, ymin * r_h, ymax * r_h
				# rescaling
				left, right, top, bottom = left*scale, right*scale, top*scale, bottom*scale
				# constraing the boxes to lie inside the frame
				left, top, right, bottom = max(0, left), max(0, top), min(right, w), min(bottom, h)
				# calculating the face width and height
				face_w, face_h = right - left, bottom - top
				# checking if the face lie within the minimum size range
				if face_w >= min_face_size and face_h >= min_face_size:
					# calculating the padding size based on the above
					face_pad_w, face_pad_h = face_w*padding_ratio, face_h*padding_ratio
					# adding the padding
					left -= face_pad_w
					top -= face_pad_h 
					right += face_pad_w
					bottom += face_pad_h
					# adjust the padded face boxes so that if it does not exceed frame size
					left, top, right, bottom = max(0, int(left)), max(0, int(top)), min(w, int(right)), min(h, int(bottom))
					bbs.append([left, top, right, bottom])
		return bbs


class HaarCascadeFaceDetector(object):
	"""Haar Cascade Face Detection
	"""
	def __init__(self):
		super(HaarCascadeFaceDetector, self).__init__()
		# load cascade classifier training file for lbpcascade
		self.haar_face_cascade = cv2.CascadeClassifier(os.path.join(os.path.dirname(os.path.abspath(__file__)), \
								 'data/xmls/haarcascade_frontalface_default.xml'))

	@timer
	def detect_haar_face(self, rgb_face_patch, scale_factor=1.1, min_neighbours=5):
		# making a copy of the original image
		img_copy = rgb_face_patch.copy()

		# convert the test image to gray image as opencv face detector expects gray images
		gray_img = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)

		# let's detect multiscale (some images may be closer to camera than others) images
		faces = self.haar_face_cascade.detectMultiScale(gray_img, scaleFactor=scale_factor, minNeighbors=min_neighbours)

		return faces
