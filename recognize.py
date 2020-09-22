#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
	Compare face embeddings to recognize persons in the frame
"""

__author__ = 'Rishab Pal'

import os
import sys
import pickle
import math
import numpy as np
# from sklearn.svm import SVC
from sklearn.metrics import pairwise_distances

import config
if config.DEBUG:
	from pdb import set_trace


class Recognize(object):
	"""Keeps track of known identities and calculates id matches"""

	def __init__(self):
		# Initializing the parameters
		self.known_ids = []
		self.known_names = []
		self.known_encodings = []
		
		self.known_ids_multiple = []
		self.known_names_multiple = []
		self.known_encodings_multiple = []

	def fetch_known_encodings(self, path):
		"""get known encodings from a pickle file"""
		try:
			if os.path.exists(path) and os.path.getsize(path) > 0:
				with open(path, 'rb') as fr:
					[self.known_ids, self.known_names, self.known_encodings] = pickle.load(fr)
		except Exception as e:
			print('Exception occured in reading from pickle file')
			print(e)
			
	def fetch_known_encodings_multiple(self, path1, path2):
		"""get known encodings from a pickle file"""
		try:
			if os.path.exists(path1) and os.path.getsize(path1) > 0:
				with open(path1, 'rb') as fr:
					[self.known_ids, self.known_names, self.known_encodings] = pickle.load(fr)
			
			if os.path.exists(path2) and os.path.getsize(path2) > 0:
				with open(path2, 'rb') as fr:
					[self.known_ids_multiple, self.known_names_multiple, self.known_encodings_multiple] = pickle.load(fr)
					
		except Exception as e:
			print('Exception occured in reading from pickle file')
			print(e)
	
	def l2_normalize(self, x):
		return x / np.sqrt(np.sum(np.power(x, 2), axis=1))
	
	def find_people(self, current_encodings, distance_threshold=0.50, percent_thres=80):
		"""
		:param features_arr: a list of 128d Features of all faces on screen
		:param thres: distance threshold
		:param percent_thres: confidence percentage
		:return: person name and percentage
		"""
		# using sklearn pairwise_distances
		current_persons = []
		distance_matrix = pairwise_distances(current_encodings, self.known_encodings, metric='euclidean', n_jobs=-1, force_all_finite=True)
		for distance_row in distance_matrix:
			min_index = np.argmin(distance_row)
			percentage = min(100, 100 * distance_threshold // distance_row[min_index])
			if percentage <= percent_thres:
				current_persons.append(('Unknown', 'Unknown', percentage))
			else:
				current_persons.append((self.known_ids[min_index], self.known_names[min_index], percentage))
		return current_persons

	def find_people_fast(self, current_encodings, distance_threshold=0.50, percent_thres=55, distance_metric=0):
		# using nxn comparison using euclidean distance
		current_persons = []
		for current_encoding in current_encodings:
			current_id = "Unknown"
			current_name = "Unknown"
			smallest = sys.maxsize
			if len(current_encoding.shape) == 1:
				current_encoding = np.expand_dims(current_encoding, axis=0)
			for (i, known_encoding) in enumerate(self.known_encodings):
				if len(known_encoding.shape) == 1:
					known_encoding = np.expand_dims(known_encoding, axis=0)
				# distance = np.sqrt(np.sum(np.square(known_encoding - current_encoding)))
				# distance = np.sqrt(np.sum(np.square(self.l2_normalize(known_encoding) - self.l2_normalize(current_encoding))))
				distance = self.distance(known_encoding, current_encoding, distance_metric)
				if distance <= smallest:
					smallest = distance
					current_id = self.known_ids[i]
					current_name = self.known_names[i]
			percentage = min(100, 100 * distance_threshold // (smallest + 1e-3))
			print(f'Percentage: {percentage}, Distance: {smallest}')
			if percentage <= percent_thres:
			# if smallest <= distance_threshold:
				current_id = "Unknown"
				current_name = "Unknown"
			current_persons.append((current_id, current_name, percentage))
		return current_persons

	def find_people_multiple(self, current_encodings, distance_threshold=0.50, percent_thres=55, distance_metric=0):
		# using nxn comparison using euclidean distance
		current_persons = []
		for current_encoding in current_encodings:
			current_id = "Unknown"
			current_name = "Unknown"
			smallest = sys.maxsize
			if len(current_encoding.shape) == 1:
				current_encoding = np.expand_dims(current_encoding, axis=0)
			for (i, encodings) in enumerate(self.known_encodings_multiple):
				for encoding in encodings:
					if len(encoding.shape) == 1:
						encoding = np.expand_dims(encoding, axis=0)
					# distance = np.sqrt(np.sum(np.square(encoding - current_encoding)))
					# distance = np.sqrt(np.sum(np.square(self.l2_normalize(encoding) - self.l2_normalize(current_encoding))))
					distance = self.distance(encoding, current_encoding, distance_metric)
					if distance <= smallest:
						smallest = distance
						current_id = self.known_ids_multiple[i]
						current_name = self.known_names_multiple[i]
			percentage = min(100, 100 * distance_threshold // (smallest + 1e-3))
			print(f'Percentage: {percentage}, Distance: {smallest}')
			if percentage <= percent_thres:
			# if smallest <= distance_threshold:
				current_id = "Unknown"
				current_name = "Unknown"
			current_persons.append((current_id, current_name, percentage))
		return current_persons

	
	def distance(self, embeddings1, embeddings2, distance_metric=0):
		if distance_metric == 0:
			# Euclidian distance
			embeddings1 = embeddings1/np.linalg.norm(embeddings1, axis=1, keepdims=True)
			embeddings2 = embeddings2/np.linalg.norm(embeddings2, axis=1, keepdims=True)
			# diff = np.subtract(embeddings1, embeddings2)
			# dist = np.sum(np.square(diff), 1)
			dist = np.sqrt(np.sum(np.square(np.subtract(embeddings1, embeddings2))))
			return dist
		elif distance_metric == 1:
			# Distance based on cosine similarity
			dot = np.sum(np.multiply(embeddings1, embeddings2), axis=1)
			norm = np.linalg.norm(embeddings1, axis=1) * np.linalg.norm(embeddings2, axis=1)
			similarity = dot/norm
			dist = np.arccos(similarity) / math.pi
			return dist[0]
		else:
			raise 'Undefined distance metric %d' % distance_metric
