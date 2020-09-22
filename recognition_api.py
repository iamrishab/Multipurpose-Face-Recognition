#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Rishab Pal'

import os
import cv2
import numpy as np
import requests
import uuid
import datetime
from flask import Flask, request, json
from flask_cors import CORS
from flask_restplus import Api, Resource, reqparse
from werkzeug.datastructures import FileStorage
from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

from recognize import Recognize
from utils import drawing_utils, preprocessing_utils
import config


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
recog.fetch_known_encodings(config.SAVED_ENCODING_FILE)

os.makedirs(config.UPLOAD_IMAGE_FOLDER, exist_ok=True)

app = Flask(__name__)
CORS(app)

api = Api(app, version='0.1', title="ONEBCG FACE RECOGNITION", validate=False, description='ONEBCG-FR')
ns = api.namespace('pipeline', description='API Operations')

upload_parser = api.parser()
upload_parser.add_argument('file', location='files', type=FileStorage, required=True)


def run_inference(img_bgr):
	if config.PREPROCESS_IMAGE:
		## Automated brightness and contrast code
		img_bgr, alpha, beta = preprocessing_utils.automatic_brightness_and_contrast(img_bgr, config.CLIP_HIST_PERCENTAGE)

	# converting the image to RGB
	img_rgb = cv2.cvtColor(img_bgr.copy(), cv2.COLOR_BGR2RGB)

	landmarks = None
	# detect face
	if config.MTCNN:
		bounding_boxes, landmarks = face_detector.detect_faces(img_rgb, config.FACE_DETECTION_CONFIDENCE, config.FACE_PADDING_RATIO, config.MIN_FACE_SIZE, config.FRAME_RESIZE_FACTOR)
	else:
		bounding_boxes = face_detector.detect_faces(img_rgb, config.FACE_DETECTION_CONFIDENCE, config.FACE_PADDING_RATIO, config.MIN_FACE_SIZE, config.FRAME_RESIZE_FACTOR)

	if len(bounding_boxes):
		if config.HEAD_POSE:
			poses = face_rotation.get_rotation(img_rgb, bounding_boxes)

		# for aligned face
		if config.ALIGN_FACE:
			aligned_face_patches = align.align_face(img_rgb, bounding_boxes, landmarks)
			embeddings = face_emb.get_embeddings_from_face_patches(aligned_face_patches)
		else:
			# without face alignment
			embeddings = face_emb.get_embeddings(img_rgb, bounding_boxes)
			
		current_persons = recog.find_people_fast(embeddings, config.DISTANCE_THRESHOLD, config.PERCENTGE_THRESHOLD_RECOGNITION)
		return bounding_boxes, current_persons
	
	return [], []


@ns.route('/')
@api.expect(upload_parser)
class Upload(Resource):
	@api.response(200, 'response success.')
	def post(self):
		args = upload_parser.parse_args()
		uploaded_file = args['file']  # This is FileStorage instance
		file_ext = uploaded_file.filename.split('.')[1]
		filename = str(uuid.uuid1()) + f'.{file_ext}'
		uploaded_file.save(os.path.join(config.UPLOAD_IMAGE_FOLDER, filename))
		file_path = os.path.join(config.UPLOAD_IMAGE_FOLDER, filename)
		img_bgr = cv2.imread(file_path)
		result = {'response':[]}
		# # image preprocessing
		is_blurr = preprocessing_utils.blurr_detection(img_bgr, config.BLURR_THRESHOLD)
		# if no blurr is detected then process further or else continue
		if not is_blurr:
			bounding_boxes, current_persons = run_inference(img_bgr)
			for (x1, y1, x2, y2), current_person in zip(bounding_boxes, current_persons):
				uid, name, confidence = current_person
				if confidence >= config.PERCENTGE_THRESHOLD_REGISTRATION:
					os.makedirs(os.path.join(config.FOLDER_TO_REGISTERED, uid), exist_ok=True)
					cv2.imwrite(os.path.join(config.FOLDER_TO_REGISTERED, uid, str(datetime.datetime.utcnow())+'.png'), img_bgr[y1:y2, x1:x2, :])
				result['response'].append({'id':uid, 'name':name, 'boundingRect':[x1, y1, x2, y2]})
		os.remove(file_path)
		return result, 200

	
if __name__ == '__main__':
	app.run(config.RECOGNITION_IP, port=config.RECOGNITION_PORT, debug=False, threaded=True)