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
	from face_embedding.resnet_v1.base import ResnetBaseServer
	face_emb = BaseServer()
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
# recog.fetch_known_encodings(config.SAVED_ENCODING_FILE)
recog.fetch_known_encodings_multiple(config.SAVED_ENCODING_FILE, config.SAVED_ENCODING_FILE_MULTIPLE)

os.makedirs(config.FOLDER_TO_REGISTERED, exist_ok=True)

# start time
start = time.time()


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
			bounding_boxes, landmarks = face_detector.detect_faces(frame_rgb, config.FACE_DETECTION_CONFIDENCE, config.FACE_PADDING_RATIO, config.MIN_FACE_SIZE, config.FRAME_RESIZE_FACTOR)
		else:
			bounding_boxes = face_detector.detect_faces(frame_rgb, config.FACE_DETECTION_CONFIDENCE, config.FACE_PADDING_RATIO, config.MIN_FACE_SIZE, config.FRAME_RESIZE_FACTOR)
		print('Faces detected:: ', len(bounding_boxes))
		if len(bounding_boxes):
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
			for (x1, y1, x2, y2), current_person, embedding in zip(bounding_boxes, current_persons, embeddings):
				uid, name, confidence = current_person
				if confidence <= config.PERCENTGE_THRESHOLD_MULTIPLE_RECOGNITION:
					new_current_person = recog.find_people_multiple([embedding], config.DISTANCE_THRESHOLD, config.PERCENTGE_THRESHOLD_RECOGNITION)
					new_uid, new_name, new_confidence = new_current_person[0]
					if new_uid != uid:
						uid, name, confidence = new_current_person[0]
				if confidence >= config.PERCENTGE_THRESHOLD_REGISTRATION and uid != 'Unknown':
					registered_img_save_path = os.path.join(config.EVALUATION_RESULT_DIR, uid)
					os.makedirs(registered_img_save_path, exist_ok=True)
					cv2.imwrite(os.path.join(registered_img_save_path, f'{str(datetime.datetime.utcnow())}.png'), frame_bgr[y1:y2, x1:x2, :])
					
				if name == 'Unknown':
					drawing_utils.draw_face_box(frame_bgr, (x1, y1), (x2, y2), (0, 0, 255), 5, 5)
					# cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)
					cv2.putText(frame_bgr, f'{name}: {confidence}', (x1, y1), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0, (255, 255, 255), 2)
				else:
					drawing_utils.draw_face_box(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 5, 5)
					# cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
					cv2.putText(frame_bgr, f'{name}: {confidence}', (x1, y1), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0, (255, 255, 255), 2)
	return frame_bgr


class VideoGet:
	"""
	Class that continuously gets frames from a VideoCapture object
	with a dedicated thread.
	"""

	def __init__(self, idx, src=0):
		self.src = src
		self.stream = cv2.VideoCapture(src)
		if type(src) != str:
			# setting camera parameters
			self.stream.set(config.FRAME_WIDTH, cv2.CAP_PROP_FRAME_WIDTH)
			self.stream.set(config.FRAME_HEIGHT, cv2.CAP_PROP_FRAME_HEIGHT)
		frame_width = int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
		frame_height = int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
		fps = self.stream.get(cv2.CAP_PROP_FPS)
		# Define the codec and create VideoWriter object
		video_save_path = os.path.join(config.EVALUATION_RESULT_DIR, str(datetime.datetime.now())+'_'+str(idx)+'.avi')
		print(f'Write video to: {video_save_path}')
		print(f'video width: {frame_width}')
		print(f'video height: {frame_height}')
		print(f'video fps: {fps}')
		self.out = cv2.VideoWriter(f'{video_save_path}', cv2.VideoWriter_fourcc('M','J','P','G'), 20., (frame_width, frame_height))
		(self.grabbed, self.frame) = self.stream.read()
		self.stopped = False

	def start(self):    
		Thread(target=self.get, args=()).start()
		return self

	def get(self):
		while not self.stopped:
			if not self.grabbed:
				self.stop()
			else:
				(self.grabbed, self.frame) = self.stream.read()
				# print(self.src)

	def stop(self):
		self.stopped = True
		self.stream.release()
		self.out.release()

		
def run_video_thread(idx, source=0):
	video_getter = VideoGet(idx, source)
	video_getter.daemon = True
	video_getter.start()

	while True:
		if video_getter.stopped:
			video_getter.stop()
			break
		frame = video_getter.frame
		print(f'Camera {idx} inference below:')
		result = run_inference(frame)
		video_getter.out.write(result)
		if time.time() - start >= 50.:
			video_getter.stopped = True
		

class VideoStream(object):
	def __init__(self, idx, path, queueSize=128):
		# initialize the file video stream along with the boolean
		# used to indicate if the thread should be stopped or not
		self.stream = cv2.VideoCapture(path)
		frame_width = int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
		frame_height = int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
		fps = self.stream.get(cv2.CAP_PROP_FPS)
		# Define the codec and create VideoWriter object
		video_save_path = os.path.join(config.EVALUATION_RESULT_DIR, str(datetime.datetime.now())+'_'+str(idx)+'.avi')
		print(f'Write video to: {video_save_path}')
		self.out = cv2.VideoWriter(f'{video_save_path}', cv2.VideoWriter_fourcc('M','J','P','G'), 20., (frame_width, frame_height))
		self.stopped = False
		# initialize the queue used to store frames read from
		# the video file
		self.Q = Queue(maxsize=queueSize)

	def start(self):
		# start a thread to read frames from the file video stream
		t = Thread(target=self.update, args=())
		t.daemon = True
		t.start()
		return self

	def update(self):
		# keep looping infinitely
		while True:
			# if the thread indicator variable is set, stop the
			# thread
			if self.stopped:
				return
			# otherwise, ensure the queue has room in it
			if not self.Q.full():
				# read the next frame from the file
				(grabbed, frame) = self.stream.read()
				# if the `grabbed` boolean is `False`, then we have
				# reached the end of the video file
				if not grabbed:
					self.stop()
					return
				# add the frame to the queue
				self.Q.put(frame)

	def read(self):
		# return next frame in the queue
		return self.Q.get()

	def more(self):
		# return True if there are still frames in the queue
		# return not self.Q.empty()
		return self.Q.qsize() > 0

	def stop(self):
		# indicate that the thread should be stopped
		self.stopped = True
		self.stream.release()
		self.out.release()
		
		
def process_video(idx, source):
	fvs = VideoStream(idx, source).start()
	time.sleep(1)
	# loop over frames from the video file stream		
	while fvs.more():
		frame = fvs.read()
		print(f'Camera {idx} inference below:')
		# PROCESS FRAME HERE
		result = run_inference(frame)
		fvs.out.write(result)
	fvs.stop()
		
		
def main():
	# video input source
	for idx, cam_id in enumerate(config.TEST_SOURCES.keys()): # change the source here accordingly
		# for processing video from cctv uncomment below line
# 		thread = Thread(target=run_video_thread, args=(cam_id, config.CAM_SOURCES[cam_id],))
		# for processing video from a file uncomment below line
		thread = Thread(target=process_video, args=(cam_id, config.TEST_SOURCES[cam_id],))
		thread.start()


if __name__ == '__main__':
	main()
