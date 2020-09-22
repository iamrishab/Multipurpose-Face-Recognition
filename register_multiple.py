#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Rishab Pal'

import cv2
import os
import sys
import uuid
import pickle
import argparse
import datetime
import  numpy as np
import tensorflow as tf

from recognize import Recognize
from utils import drawing_utils, preprocessing_utils
import config

if config.DEBUG:
	from pdb import set_trace

SUPPORTED_IMAGE_FORMATS = ['jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG']
recog = Recognize()

parser = argparse.ArgumentParser(description='Register New Person')
parser.add_argument('--source', required=True, choices=['folder', 'webcam'], type=str, help='folder | webcam')
parser.add_argument('--path', required=True, type=str, help='path to source')
args = parser.parse_args()

# creating face detector objects
from face_detection.mtcnn import MTCNN
face_detector = MTCNN()

# creating face embedding calculator objects
if config.INCEPTION_RESNET:
	from face_embedding.inception_resnet_v1.load_model import InceptionResnet
	face_emb = InceptionResnet(config.GPU_MEMORY_FRACTION_TO_USE_RECOGNITION)
else:
	raise Exception('Model for embedding calculation not specified. Specify INCEPTION_RESNET=True in config.py')
	
# face alignment
if config.ALIGN_FACE:
	from align import AlignFace
	align = AlignFace(device='cuda', flip_input=False, load_model=False)
	
# if config.HEAD_POSE:
# 	from face_alignment.hopenet.pose import FaceRotation
# 	face_rotation = FaceRotation()

os.makedirs(config.FOLDER_TO_REGISTERED, exist_ok=True)


def save_encoding(name_id : str, name: str, encoding : list, current_person: list):
	"""
		saving known encoding to a pickle file
	"""
	
	if os.path.exists(config.SAVED_ENCODING_FILE) and os.path.exists(config.SAVED_ENCODING_FILE_MULTIPLE):
		
		recog.fetch_known_encodings_multiple(config.SAVED_ENCODING_FILE, config.SAVED_ENCODING_FILE_MULTIPLE)
		known_ids, known_names, known_encodings = recog.known_ids, recog.known_names, recog.known_encodings
		known_ids_multiple, known_names_multiple, known_encodings_multiple = recog.known_ids_multiple, recog.known_names_multiple, recog.known_encodings_multiple
		
		_id, _name, _confidence = current_person
		
		if isinstance(name, str):
			name = name.lower()
		else:
			name = ''
		if _id != 'Unknown':
			id_index = known_ids.index(_id)
			known_encodings_multiple[id_index].append(encoding)
			print(f'Encoding updated for:: {_id}')
		else:
			known_ids.append(name_id)
			known_names.append(name)
			known_encodings.append(encoding)
			
			known_ids_multiple.append(name_id)
			known_names_multiple.append(name)
			known_encodings_multiple.append([encoding])
			
			# save file with single encoding for each person
			with open(config.SAVED_ENCODING_FILE, 'wb') as fw:
				pickle.dump([known_ids, known_names, known_encodings], fw, protocol=pickle.HIGHEST_PROTOCOL)
				
			print(f'Encoding added for :: {name_id}')
		# save file with multiple encoding for each person
		with open(config.SAVED_ENCODING_FILE_MULTIPLE, 'wb') as fw:
			pickle.dump([known_ids_multiple, known_names_multiple, known_encodings_multiple], fw, protocol=pickle.HIGHEST_PROTOCOL)
		return True
	else:
		os.makedirs(os.path.dirname(config.SAVED_ENCODING_FILE), exist_ok=True)
		os.makedirs(os.path.dirname(config.SAVED_ENCODING_FILE_MULTIPLE), exist_ok=True)
		with open(config.SAVED_ENCODING_FILE, 'wb') as fw:
			pickle.dump([[name_id], [name.lower()], [encoding]], fw, protocol=pickle.HIGHEST_PROTOCOL)
		with open(config.SAVED_ENCODING_FILE_MULTIPLE, 'wb') as fw:
			pickle.dump([[name_id], [name.lower()], [[encoding]]], fw, protocol=pickle.HIGHEST_PROTOCOL)
		print(f'Encoding added for:: {name_id}')
		return True

		
def check_frontal_face(face_w, face_h, landmarks):
	left_eye = landmarks['left_eye']
	right_eye = landmarks['right_eye']
	nose = landmarks['nose']
	mouth_left = landmarks['mouth_left']
	mouth_right = landmarks['mouth_right']
	
	eye_distance = np.sqrt(np.sum(np.square(np.array(left_eye) - np.array(right_eye))))
	mouth_distance = np.sqrt(np.sum(np.square(np.array(mouth_left) - np.array(mouth_right))))
	
	normalized_eye_distance = eye_distance / float(face_w)
	normalized_mouth_distance = mouth_distance / float(face_h)
	
	print(f'face_W: {face_w}, face_h: {face_h}, eye_dis: {normalized_eye_distance}, mouth_dis: {normalized_mouth_distance}')
	if normalized_eye_distance >= config.MIN_NORMALIZED_DISTANCE_BETWEEN_EYES and normalized_mouth_distance >= config.MIN_NORMALIZED_DISTANCE_BETWEEN_MOUTH:
		return True
	return False


def run_inference(pic_bgr):
	if config.PREPROCESS_IMAGE:
		# Automated brightness and contrast code
		pic_bgr, alpha, beta = preprocessing_utils.automatic_brightness_and_contrast(pic_bgr, config.CLIP_HIST_PERCENTAGE)

	# converting the image to RGB
	pic_rgb = cv2.cvtColor(pic_bgr, cv2.COLOR_BGR2RGB)
	pic_rgb_flip = tf.image.flip_left_right(pic_rgb)

	# detect face
	bounding_boxes, landmarks = face_detector.detect_faces(pic_rgb, config.FACE_DETECTION_CONFIDENCE,
					 config.FACE_PADDING_RATIO, config.MIN_FACE_SIZE, 1)

	if len(bounding_boxes) == 1:
		
# 		if config.HEAD_POSE:
# 			poses = face_rotation.get_rotation(pic_rgb, bounding_boxes)

		# for aligned face
		if config.ALIGN_FACE:
			aligned_face_patches = align.align_face(pic_rgb, bounding_boxes, landmarks, mtcnn=True)
			embeddings = face_emb.get_embeddings_from_face_patches(aligned_face_patches)
# 		else:
# 			# without face alignment
# 			embeddings = face_emb.get_embeddings(pic_rgb, bounding_boxes)
			
		face_w, face_h = bounding_boxes[0][2] - bounding_boxes[0][0], bounding_boxes[0][3] - bounding_boxes[0][1]
		is_frontal = check_frontal_face(face_w, face_h, landmarks[0])
		if is_frontal:
			embeddings = face_emb.get_embeddings(pic_rgb, bounding_boxes)
			current_persons = recog.find_people_fast(embeddings, config.DISTANCE_THRESHOLD, config.PERCENTGE_THRESHOLD_RECOGNITION)
			return bounding_boxes, embeddings, current_persons[0]
		else:
			print('Frontal face check failed')
	else:
		print('More than one face detected')
	return [], [], []


def register_from_folder(source: str):
	"""
	Register all the persons present in the current subfolder
	Min. one image of each person must be there in the subfolder
	"""
	# import pdb; pdb.set_trace()
	count = 0
	blurr_pic_count = 0
	for folder in os.listdir(source):
		uid = str(uuid.uuid4())
		for i, file in enumerate([f for f in os.listdir(os.path.join(source, folder)) if f.split('.')[-1] in SUPPORTED_IMAGE_FORMATS]):
			pic_bgr = cv2.imread(os.path.join(source, folder, file))
			# image preprocessing
			is_blurr = preprocessing_utils.blurr_detection(pic_bgr, config.BLURR_THRESHOLD)
			# if no blurr is detected then process further or else continue
			if not is_blurr:
				
				# TODO: CHANGE NAME/ID LOGIC BASED ON CURRENT DATA/SAVE FORMAT
				if folder.strip()[0].isdigit():
					person_name_id = " ".join(folder.strip().split(' ')[1:]).lower()
				else:
					person_name_id = folder.strip().lower()

				bounding_boxes, embeddings, current_person = run_inference(pic_bgr)
				
				if len(bounding_boxes):
						# save known_encodings
						saved = save_encoding(uid, person_name_id, embeddings, current_person)
						if not saved:
							print(f'Failed to register encoding for: {person_name_id}')
							continue
						registered_img_save_path = os.path.join(config.FOLDER_TO_REGISTERED, uid)
						os.makedirs(registered_img_save_path, exist_ok=True)
						cv2.imwrite(os.path.join(registered_img_save_path, f'{str(datetime.datetime.utcnow())}.jpg'), pic_bgr)
						count += 1
				else:
					continue
			else:
				print(f'Blurr image received: {file}')
				continue
	print(f'Number of new registeries: {count}')


# starting of program execution
if __name__ == '__main__':
	# dynamic function calling based on the set parameter
	getattr(sys.modules[__name__], f"register_from_{args.source}")(args.path)