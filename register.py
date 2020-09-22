#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
	Two kinds of registration to be done:
		1. From Webcam: Enter name/id from console
		2. From folder:
			a. Each subfolder inside the folder will represent one class
			b. Put atleast one image of the person in the subfolder
			c. Subfolder name will be considered as the id/name of the person
"""

__author__ = 'Rishab Pal'

import cv2
import os
import sys
import uuid
import pickle
import argparse
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
# parser.add_argument('--detector', required=False, type=str, help='face detection model')
# parser.add_argument('--recognizer', required=True, type=str, help='face recognition model')
args = parser.parse_args()

# creating face detector objects
from face_detection.ssd_mobilenet.detector import SSDMobilentFaceDetector
face_detector = SSDMobilentFaceDetector(config.GPU_MEMORY_FRACTION_TO_USE_DETECTION)

# creating face embedding calculator objects
if config.INCEPTION_RESNET:
	from face_embedding.inception_resnet_v1.load_model import InceptionResnet
	face_emb = InceptionResnet(config.GPU_MEMORY_FRACTION_TO_USE_RECOGNITION)
elif config.RESNET_50:
	from face_embedding.arcface.embs import ArcFaceResNet50
	face_emb = ArcFaceResNet50()
else:
	raise Exception('Model for embedding calculation not specified')
	
# face alignment
if config.ALIGN_FACE:
	from align import AlignFace
	align = AlignFace(device='cuda', flip_input=False)
	
if config.HEAD_POSE:
	from face_alignment.hopenet.pose import FaceRotation
	face_rotation = FaceRotation()


def save_encoding(name_id : str, name: str, encoding : list, current_person: list):
	"""
		saving known encoding to a pickle file
	"""
	try:
		if os.path.exists(config.SAVED_ENCODING_FILE):
			recog.fetch_known_encodings(config.SAVED_ENCODING_FILE)
			known_ids, known_names, known_encodings = recog.known_ids, recog.known_names, recog.known_encodings
			_id, _name, _confidence = current_person
			if isinstance(name, str):
				name = name.lower()
			else:
				name = ''
			if _id != 'Unknown':
				print(f'Record with name: {name} already exists with id: {_id}, name: {_name}, confidence: {_confidence}')
				response = input('Enter YES to overwrite existing and NO to skip: ')
				if 'y' in response.lower():
					name_id = _id
					id_index = known_ids.index(_id)
					known_encodings[id_index] = encoding
					known_names[id_index] = name
				else:
					return False
			else:
				known_ids.append(name_id)
				known_names.append(name)
				known_encodings.append(encoding)
			with open(config.SAVED_ENCODING_FILE, 'wb') as fw:
				pickle.dump([known_ids, known_names, known_encodings], fw, protocol=pickle.HIGHEST_PROTOCOL)
			print(f'Encoding updated:: {name_id}')
			return True
		else:
			os.makedirs(os.path.dirname(config.SAVED_ENCODING_FILE), exist_ok=True)
			with open(config.SAVED_ENCODING_FILE, 'wb') as fw:
				pickle.dump([[name_id], [name.lower()], [encoding]], fw, protocol=pickle.HIGHEST_PROTOCOL)
			print(f'Encoding updated:: {name_id}')
			return True
	except Exception as e:
		print('Exception occured in reading from Pickle file')
		print(e)

		
def run_inference(pic_bgr):
	# Automated brightness and contrast code
	if config.PREPROCESS_IMAGE:
		pic_bgr, alpha, beta = preprocessing_utils.automatic_brightness_and_contrast(pic_bgr, config.CLIP_HIST_PERCENTAGE)

	# converting the image to RGB
	pic_rgb = cv2.cvtColor(pic_bgr, cv2.COLOR_BGR2RGB)
	# pic_rgb_flip = tf.image.flip_left_right(pic_rgb)
	landmarks = None
	# detect face
	bounding_boxes = face_detector.detect_faces(pic_rgb, config.FACE_DETECTION_CONFIDENCE,
					 config.FACE_PADDING_RATIO, config.MIN_FACE_SIZE, 1)

	if len(bounding_boxes):
		if config.HEAD_POSE:
			poses = face_rotation.get_rotation(pic_rgb, bounding_boxes)
		# for aligned face
		if config.ALIGN_FACE:
			aligned_face_patches = align.align_face(pic_rgb, bounding_boxes, landmarks)
			embeddings = face_emb.get_embeddings_from_face_patches(aligned_face_patches)
		else:
			# without face alignment
			embeddings = face_emb.get_embeddings(pic_rgb, bounding_boxes)
		current_persons = recog.find_people_fast(embeddings, config.DISTANCE_THRESHOLD, config.PERCENTGE_THRESHOLD_REGISTRATION)
		return bounding_boxes, embeddings, current_persons
	return [], [], []
	

def register_from_folder(source: str):
	"""
	Register all the persons present in the current subfolder
	Min. one image of each person must be there in the subfolder
	"""
	count = 0
	blurr_pic_count = 0
	for folder in os.listdir(source):
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
				
				bounding_boxes, embeddings, current_persons = run_inference(pic_bgr)
				
				if len(bounding_boxes) == 1:
					# save known_encodings
					saved = save_encoding(str(uuid.uuid4()), person_name_id, embeddings, current_persons[0])
					if saved:
						count += 1
						break
					else:
						print(f'Failed to register encoding for: {person_name_id}')
						continue
				else:
					print(f'More than one face detected: {file}')
			else:
				print(f'Blurr image received: {file}')
				continue

	print(f'Number of new registeries: {count}')


def register_from_webcam(source: str):
	"""
	Register person using webcam
	"""
	# reading camera feed
	cap = cv2.VideoCapture(int(source))

	cv2.namedWindow('Capture')
	response = True

	print("Press 'S' to save encodings")
	while response:
		name_id = str(uuid.uuid4())
		name = input('Enter your name: ')
		# reading feed from camera
		while cap.isOpened():
			ret, frame_bgr = cap.read()
			# close if the input source cannot fetch any frame
			if not ret:
				break		
			# image preprocessing
			is_blurr = preprocessing_utils.blurr_detection(frame_bgr, config.BLURR_THRESHOLD)
			# if no blurr is detected then process further or else continue
			if not is_blurr:
			
				bounding_boxes, embeddings, current_persons = run_inference(frame_bgr)
				
				for x1, y1, x2, y2 in bounding_boxes:
					if len(bounding_boxes) == 1:
						drawing_utils.draw_face_box(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 5, 5)
						cv2.putText(frame_bgr, 'Press S to capture', (x1, y1), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0, (255, 255, 255), 2)
					else:
						drawing_utils.draw_face_box(frame_bgr, (x1, y1), (x2, y2), (0, 0, 255), 5, 5)
						cv2.putText(frame_bgr, 'More than one face detected', (x1, y1), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0, (255, 255, 255), 2)
					
				cv2.imshow('Capture', frame_bgr)

				if cv2.waitKey(1) == ord('s'):
					# save image
					cv2.destroyWindow('Capture')
					if len(bounding_boxes) == 1:						
						# save known_encodings
						saved = save_encoding(name_id, name, embeddings, current_persons[0])
						if saved:
							os.makedirs(os.path.join(config.FOLDER_TO_REGISTERED, name_id), exist_ok=True)
							cv2.imwrite(os.path.join(config.FOLDER_TO_REGISTERED, name_id, name_id+'.png'), frame_bgr)
							break
						else:
							print(f'Failed to register encoding for: {folder}')
							continue
					else:
						print('More than one face detected')
						continue
			else:
				print(f'Blurr frame received')
				continue

			if cv2.waitKey(1) == ord('q'):
				break
		response = input('Enter YES to continue and NO to exit: ')
		if 'y' in response.lower():
			continue
		else:
			break
	cv2.destroyAllWindows()
	cap.release()


# starting of program execution
if __name__ == '__main__':
	# dynamic function calling based on the set parameter
# 	getattr(sys.modules[__name__], f"register_from_{args.source}")(args.path)
	register_from_folder(args.path)