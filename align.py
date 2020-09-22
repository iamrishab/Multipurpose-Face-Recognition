#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Rishab Pal'


import cv2
import numpy as np

import config
from face_alignment import fan

if config.DEBUG:
	from pdb import set_trace
	
	
class AlignFace(object):
	def __init__(self, device='cuda', flip_input=False, load_model=True):
		super(AlignFace, self).__init__()
		if load_model:
			# Run the 3D face alignment on a test image, without CUDA.
			self.fa = fan.FaceAlignment(fan.LandmarksType._2D, device=device, flip_input=flip_input)
		self.landmark_positions = {
						  'face': (0, 17),
						  'eyebrow1': (17, 22),
						  'eyebrow2': (22, 27),
						  'nose': (27, 31),
						  'nostril': (31, 36),
						  'eye1': (36, 42),
						  'eye2': (42, 48),
						  'lips': (48, 60),
						  'teeth': (60, 68),
				  }

	def align_face(self, img_rgb, detected_faces, face_landmarks, scale=0.9, mtcnn=False):
		'''
		face alignment API for all image, get the landmark of eyes and nose and do warpaffine transformation
		:return: an aligned single face image
		'''
		aligned = []
		if not face_landmarks:
			face_landmarks = self.fa.get_landmarks(img_rgb, detected_faces)
		if len(face_landmarks):
			for face_landmark, detected_face in zip(face_landmarks, detected_faces):
				left_eye_center = face_landmark['left_eye'] if mtcnn else self._find_center_pt(face_landmark[self.landmark_positions['eye1'][0]:self.landmark_positions['eye1'][1], :])
				right_eye_center = face_landmark['right_eye'] if mtcnn else self._find_center_pt(face_landmark[self.landmark_positions['eye2'][0]:self.landmark_positions['eye2'][1], :])
				nose_center = face_landmark['nose'] if mtcnn else self._find_center_pt(face_landmark[self.landmark_positions['nose'][0]:self.landmark_positions['nose'][1], :])
				x1, y1, x2, y2 = detected_face
				face_img = img_rgb[y1:y2, x1:x2, :]
				if config.DEBUG: cv2.imwrite('data/actual.jpg', cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))
				h, w = face_img.shape[:2]
				trotate = self._get_rotation_matrix(left_eye_center, right_eye_center, nose_center, scale)
				warped = cv2.warpAffine(face_img, trotate, (w, h))
				aligned.append(warped)
				if config.DEBUG: cv2.imwrite('data/warped.jpg', cv2.cvtColor(warped, cv2.COLOR_RGB2BGR))
		return aligned

	def _find_center_pt(self, points):
		'''
		find centroid point by several points that given
		'''
		x = 0
		y = 0
		num = len(points)
		for pt in points:
			x += pt[0]
			y += pt[1]
		x //= num
		y //= num
		return (x,y)

	def _angle_between_2_pt(self, p1, p2):
		'''
		to calculate the angle rad by two points
		'''
		delta = 1e-3
		x1, y1 = p1
		x2, y2 = p2
		tan_angle = (y2 - y1) / ((x2 - x1) + delta)
		return (np.degrees(np.arctan(tan_angle)))

	def _get_rotation_matrix(self, left_eye_pt, right_eye_pt, nose_center, scale):
		'''
		to get a rotation matrix by using skimage, including rotate angle, transformation distance and the scale factor
		'''
		eye_angle = self._angle_between_2_pt(left_eye_pt, right_eye_pt)
		M = cv2.getRotationMatrix2D((nose_center[0]/2, nose_center[1]/2), eye_angle, scale )

		return M

	def _dist_nose_tip_center_and_img_center(self, nose_pt, img_shape):
		'''
		find the distance between nose tip's centroid and the centroid of original image
		'''
		y_img, x_img, _ = img_shape
		img_center = (x_img//2, y_img//2)
		return ((img_center[0] - nose_pt[0]), -(img_center[1] - nose_pt[1]))

	def _crop_face(self, img, face_loc, padding_size=0):
		'''
		crop face into small image, face only, but the size is not the same
		'''
		h, w, c = img.shape
		top = face_loc[0] - padding_size
		right = face_loc[1] + padding_size
		down = face_loc[2] + padding_size
		left = face_loc[3] - padding_size

		if top < 0:
			top = 0
		if right > w - 1:
			right = w - 1
		if down > h - 1:
			down = h - 1
		if left < 0:
			left = 0
		img_crop = img[top:down, left:right]
		return img_crop

	def _face_locations_raw(self, img, scale):
	#     img_scale = (tf.resize(img, (img.shape[0]//scale, img.shape[1]//scale)) * 255).astype(np.uint8)
		h, w, c = img.shape
		img_scale = cv2.resize(img, (int(img.shape[1]//scale), int(img.shape[0]//scale)))
		face_loc_small = fr.face_locations(img_scale)
		face_loc = []
		for ff in face_loc_small:
			tmp = [pt*scale for pt in ff]
			if tmp[1] >= w:
				tmp[1] = w
			if tmp[2] >= h:
				tmp[2] = h
			face_loc.append(tmp)
		return face_loc

	def _face_locations_small(self, img):
		for scale in [16, 8, 4, 2, 1]:
			face_loc = _face_locations_raw(img, scale)
			if face_loc != []:
				return face_loc
		return []
