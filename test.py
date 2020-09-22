#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Rishab Pal'

import os
import cv2
import time
import datetime
import numpy as np
from threading import Thread
from multiprocessing import Queue

from recognize import Recognize
from utils import drawing_utils, preprocessing_utils
import config

if config.DEBUG:
	from pdb import set_trace

# creating detector objects
if config.SSD_MOBILENET:
	from face_detection.ssd_mobilenet.detector import SSDMobilentFaceDetector
	face_detector = SSDMobilentFaceDetector(config.GPU_MEMORY_FRACTION_TO_USE_DETECTION)
elif config.SSD_INCEPTION:
	from face_detection.ssd_inception.face_detector import SSDInceptionFaceDetector
	face_detector = SSDInceptionFaceDetector(config.GPU_MEMORY_FRACTION_TO_USE_DETECTION)
elif config.MTCNN:
	from face_detection.mtcnn import MTCNN
	face_detector = MTCNN()
else:
	raise Exception('Model for face detection not specified')

# creating face embedding calculator objects
if config.INCEPTION_RESNET:
	from face_embedding.inception_resnet_v1.load_model import InceptionResnet
	face_emb = InceptionResnet(config.GPU_MEMORY_FRACTION_TO_USE_RECOGNITION)
elif config.RESNET_50:
	from face_embedding.arcface.embs import ArcFaceResNet50
	face_emb = ArcFaceResNet50()
else:
	raise Exception('Model for face embedding not specified')
	
# face alignment
if config.ALIGN_FACE:
	from align import AlignFace
	align = AlignFace(device='cuda', flip_input=False)
	
if config.HEAD_POSE:
	from face_alignment.hopenet.pose import FaceRotation
	face_rotation = FaceRotation()
	
# face comparison object
recog = Recognize()
# fetch all recognized person's info
recog.fetch_known_encodings(config.SAVED_ENCODING_FILE)


def run_inference(frame_bgr):
	# # image preprocessing
	is_blurr = preprocessing_utils.blurr_detection(frame_bgr.copy(), config.BLURR_THRESHOLD)

	# if no blurr is detected then process further or else continue
	if not is_blurr:
		if config.PREPROCESS_IMAGE:
			## Automated brightness and contrast code
			frame_bgr, alpha, beta = preprocessing_utils.automatic_brightness_and_contrast(frame_bgr, config.CLIP_HIST_PERCENTAGE)
		
		# converting the image to RGB
		frame_rgb = cv2.cvtColor(frame_bgr.copy(), cv2.COLOR_BGR2RGB)
		landmarks = None
		# detect face
		if config.MTCNN:
				bounding_boxes, landmarks = face_detector.detect_faces(frame_rgb, config.FACE_DETECTION_CONFIDENCE, config.FACE_PADDING_RATIO, config.MIN_FACE_SIZE)
		else:
			bounding_boxes = face_detector.detect_faces(frame_rgb, config.FACE_DETECTION_CONFIDENCE, config.FACE_PADDING_RATIO, config.MIN_FACE_SIZE, config.FRAME_RESIZE_FACTOR)
		print('Faces detected:: ', len(bounding_boxes))
		if len(bounding_boxes):
			# Here bounding_box in the method argument must be in the format
			# x1, y1, x2(x1+w), y2(y1+h)
			
			if config.HEAD_POSE:
				poses = face_rotation.get_rotation(frame_rgb, bounding_boxes)
			
			# for aligned face
			if config.ALIGN_FACE:
				aligned_face_patches = align.align_face(frame_rgb, bounding_boxes, landmarks)
				embeddings = face_emb.get_embeddings_from_face_patches(aligned_face_patches)
			else:
				# without face alignment
				embeddings = face_emb.get_embeddings(frame_rgb, bounding_boxes)
			
			current_persons = recog.find_people_fast(embeddings, config.DISTANCE_THRESHOLD, config.PERCENTGE_THRESHOLD_RECOGNITION)
			print('Persons detected:: ', current_persons)
			return bounding_boxes, current_persons
	return [], []
			


def draw_on_frame(frame_bgr, bounding_boxes, current_persons):
	for (x1, y1, x2, y2), current_person in zip(bounding_boxes, current_persons):
		uid, name, confidence = current_person
		if name == 'Unknown':
			drawing_utils.draw_face_box(frame_bgr, (x1, y1), (x2, y2), (0, 0, 255), 5, 5)
			# cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)
			cv2.putText(frame_bgr, f'{name}: {confidence}', (x1, y1), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0, (255, 255, 255), 2)
		else:
			drawing_utils.draw_face_box(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 5, 5)
			# cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
			cv2.putText(frame_bgr, f'{name}: {confidence}', (x1, y1), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0, (255, 255, 255), 2)
			if confidence >= config.PERCENTGE_THRESHOLD_REGISTRATION:
				os.makedirs(os.path.join(config.FOLDER_TO_REGISTERED, uid), exist_ok=True)
				cv2.imwrite(os.path.join(config.FOLDER_TO_REGISTERED, uid, str(datetime.datetime.utcnow())+'.png'), frame_bgr[y1:y2, x1:x2, :])

	return frame_bgr
	
	
def main():
	# reading camera feed
	cap = cv2.VideoCapture(config.VIDEO_SOURCE)
	
	if not isinstance(config.VIDEO_SOURCE, str):
		# setting camera parameters
		cap.set(config.FRAME_WIDTH, cv2.CAP_PROP_FRAME_WIDTH)
		cap.set(config.FRAME_HEIGHT, cv2.CAP_PROP_FRAME_HEIGHT)

	# reading feed from camera
	while cap.isOpened():
		ret, frame_bgr = cap.read()

		# close if the input source cannot fetch any frame
		if not ret:
			break

		result = run_inference(frame_bgr)

# 		uncomment when using UI
# 		frame_bgr = draw_on_frame(frame_bgr)
# 		cv2.imshow('frame', frame_bgr)
#       if cv2.waitKey(1) == ord('q'):
#           break
	
	cv2.destroyAllWindows()
	cap.release()


if __name__ == '__main__':
	main()