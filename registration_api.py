#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Rishab Pal'

import os
import uuid
import cv2
import requests
import datetime
import pickle
from flask import Flask
from flask_cors import CORS
from flask_restplus import reqparse
from flask_restplus import Api, Resource
from werkzeug.datastructures import FileStorage
from flask import Flask, Response, jsonify, request
from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

from recognize import Recognize
from utils import drawing_utils, preprocessing_utils
import config

# creating face detector objects
from face_detection.ssd_mobilenet.detector import SSDMobilentFaceDetector
face_detector = SSDMobilentFaceDetector(config.GPU_MEMORY_FRACTION_TO_USE_DETECTION)

# creating face embedding calculator objects
if config.INCEPTION_RESNET:
	from face_embedding.inception_resnet_v1.load_model import InceptionResnet
	face_emb = InceptionResnet(config.GPU_MEMORY_FRACTION_TO_USE_RECOGNITION)
elif config.RESNET_50:
	from face_embedding.resnet_v1.base import ResnetBaseServer
	face_emb = BaseServer()
else:
	raise Exception('Model for embedding calculation not specified')
	
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

api = Api(app, version='0.1', title="ONEBCG FACE REGISTRATION", validate=False, description='ONEBCG-FR')
ns = api.namespace('pipeline', description='API Operations')

upload_parser = api.parser()
args_parser = reqparse.RequestParser()

upload_parser.add_argument('file', location='files', type=FileStorage, required=True)
args_parser.add_argument('name', required=False, help="given name")


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
				return _id
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
		
		
def run_inference(img_bgr):
	# Automated brightness and contrast code
	if config.PREPROCESS_IMAGE:
		img_bgr, alpha, beta = preprocessing_utils.automatic_brightness_and_contrast(img_bgr, config.CLIP_HIST_PERCENTAGE)

	# converting the image to RGB
	img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

	landmarks = None
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

		current_persons = recog.find_people_fast(embeddings, config.DISTANCE_THRESHOLD, config.PERCENTGE_THRESHOLD_REGISTRATION)
		return bounding_boxes, embeddings, current_persons
	return [], [], []


@ns.route('/')
@api.expect(upload_parser, args_parser)
class Upload(Resource):
	@api.response(200, 'response success.')
	def post(self):
		args = upload_parser.parse_args()
		params = args_parser.parse_args()
		
		u_id = str(uuid.uuid4())
		uploaded_file = args['file']  # This is FileStorage instance
		name = params['name']
		
		file_ext = uploaded_file.filename.split('.')[1]		
		filename = u_id + f'.{file_ext}'
		uploaded_file.save(os.path.join(config.UPLOAD_IMAGE_FOLDER, filename))
		file_path = os.path.join(config.UPLOAD_IMAGE_FOLDER, filename)
		img_bgr = cv2.imread(file_path)
		result = {'response':[]}
		# image preprocessing
		is_blurr = preprocessing_utils.blurr_detection(img_bgr, config.BLURR_THRESHOLD)
		# if no blurr is detected then process further or else continue
		if not is_blurr:
			bounding_boxes, embeddings, current_persons = run_inference(img_bgr)
			if len(bounding_boxes) == 1:
				x1, y1, x2, y2 = bounding_boxes[0]
				current_person = current_persons[0]
				# save known_encodings
				saved = save_encoding(u_id, name, embeddings, current_person)
				if isinstance(saved, bool) and saved:
					os.makedirs(os.path.join(config.FOLDER_TO_REGISTERED, u_id), exist_ok=True)
					cv2.imwrite(os.path.join(config.FOLDER_TO_REGISTERED, u_id, str(datetime.datetime.utcnow())+'.png'), img_bgr[y1:y2, x1:x2, :])
					result['response'].append({'id':u_id, 'boundingRect':[x1, y1, x2, y2]})
				elif isinstance(saved, str):
					result['response'].append({'id':saved, 'boundingRect':[x1, y1, x2, y2]})
				else:
					result['response'].append({'id':'', 'boundingRect':[]})
			else:
				result['response'].append({'id':'', 'boundingRect':[]})

		os.remove(file_path)
		return result, 200

	
if __name__ == '__main__':
	app.run(config.REGISTRATION_IP, port=config.REGISTRATION_PORT, debug=False, threaded=True)
