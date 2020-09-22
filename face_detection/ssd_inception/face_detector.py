import os
import cv2
import time
import numpy as np
import tensorflow as tf
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


class SSDInceptionFaceDetector:
	def __init__(self, gpu_memory_fraction=0.25):
		"""
		Arguments:
			model_path: a string, path to a pb file.
			gpu_memory_fraction: a float number.
		"""
		with tf.io.gfile.GFile(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model/model.pb'), 'rb') as f:
			graph_def = tf.compat.v1.GraphDef()
			graph_def.ParseFromString(f.read())

		graph = tf.Graph()
		with graph.as_default():
			tf.import_graph_def(graph_def, name='import')

		self.input_image = graph.get_tensor_by_name('import/image_tensor:0')
		self.output_ops = [
			graph.get_tensor_by_name('import/boxes:0'),
			graph.get_tensor_by_name('import/scores:0'),
			graph.get_tensor_by_name('import/num_boxes:0'),
		]

		gpu_options = tf.compat.v1.GPUOptions(
			per_process_gpu_memory_fraction=gpu_memory_fraction
		)
		config_proto = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
		self.sess = tf.compat.v1.Session(graph=graph, config=config_proto)

	@timer
	def _detect_faces(self, image, score_threshold=0.5, padding_ratio=0.15, min_face_size=20):
		"""Detect faces.

		Arguments:
			image: a numpy uint8 array with shape [height, width, 3],
				that represents a RGB image.
			score_threshold: a float number.
		Returns:
			boxes: a float numpy array of shape [num_faces, 4].
			scores: a float numpy array of shape [num_faces].

		Note that box coordinates are in the order: ymin, xmin, ymax, xmax!
		"""
		h, w, _ = image.shape
		image = np.expand_dims(image, 0)

		boxes, scores, num_boxes = self.sess.run(
			self.output_ops, feed_dict={self.input_image: image}
		)
		num_boxes = num_boxes[0]
		boxes = boxes[0][:num_boxes]
		scores = scores[0][:num_boxes]

		to_keep = scores > score_threshold
		boxes = boxes[to_keep]
		scores = scores[to_keep]

		scaler = np.array([h, w, h, w], dtype='float32')
		boxes = boxes * scaler
		
		bbs = []
		for score, (ymin, xmin, ymax, xmax) in zip(scores, boxes):
			if score >= score_threshold:
				# calculating the face width and height
				face_w, face_h = xmax - xmin, ymax - ymin
				# checking if the face lie within the minimum size range
				if face_w >= min_face_size and face_h >= min_face_size:
					# calculating the padding size based on the above
					face_pad_w, face_pad_h = face_w*padding_ratio, face_h*padding_ratio
					# adding the padding
					xmin -= face_pad_w
					ymin -= face_pad_h 
					xmax += face_pad_w
					ymax += face_pad_h
					# adjust the padded face boxes so that if it does not exceed frame size
					xmin, ymin, xmax, ymax = max(0, int(xmin)), max(0, int(ymin)), min(w, int(xmax)), min(h, int(ymax))
					# adding
					bbs.append([xmin, ymin, xmax, ymax])
		return bbs
	
	@timer
	def detect_faces(self, image, score_threshold=0.5, padding_ratio=0.15, min_face_size=20, scale=2):
		"""Detect faces.

		Arguments:
			image: a numpy uint8 array with shape [height, width, 3],
				that represents a RGB image.
			score_threshold: a float number.
		Returns:
			boxes: a float numpy array of shape [num_faces, 4].
			scores: a float numpy array of shape [num_faces].

		Note that box coordinates are in the order: ymin, xmin, ymax, xmax!
		"""
		h, w = image.shape[:2]
		r_h, r_w = int(h // scale), int(w // scale)
		
		frame_resized = cv2.resize(image.copy(), (r_w, r_h), interpolation = cv2.INTER_AREA)
		
		frame_resized = np.expand_dims(frame_resized, 0)
		
		boxes, scores, num_boxes = self.sess.run(
			self.output_ops, feed_dict={self.input_image: frame_resized}
		)
		num_boxes = num_boxes[0]
		boxes = boxes[0][:num_boxes]
		scores = scores[0][:num_boxes]

		to_keep = scores > score_threshold
		boxes = boxes[to_keep]
		scores = scores[to_keep]

		scaler = np.array([r_h, r_w, r_h, r_w], dtype='float32')
		boxes = boxes * scaler
		
		bbs = []
		for score, (ymin, xmin, ymax, xmax) in zip(scores, boxes):
			if score >= score_threshold:
				# rescaling co-ordinates wrt actual image
				ymin, xmin, ymax, xmax = ymin*scale, xmin*scale, ymax*scale, xmax*scale
				# calculating the face width and height
				face_w, face_h = xmax - xmin, ymax - ymin
				# checking if the face lie within the minimum size range
				if face_w >= min_face_size and face_h >= min_face_size:
					# calculating the padding size based on the above
					face_pad_w, face_pad_h = face_w*padding_ratio, face_h*padding_ratio
					# adding the padding
					xmin -= face_pad_w
					ymin -= face_pad_h 
					xmax += face_pad_w
					ymax += face_pad_h
					# adjust the padded face boxes so that if it does not exceed frame size
					xmin, ymin, xmax, ymax = max(0, int(xmin)), max(0, int(ymin)), min(w, int(xmax)), min(h, int(ymax))
					# adding
					bbs.append([xmin, ymin, xmax, ymax])
		return bbs
