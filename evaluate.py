#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Rishab Pal'


import os
import sys
import cv2
from math import ceil
import numpy as np
import argparse
import datetime

# from monitor import CheckHardwareUtilization
from recognize import Recognize
from utils import drawing_utils, preprocessing_utils
import config


if config.DEBUG:
	from pdb import set_trace
	
	
parser = argparse.ArgumentParser(description='Benchmark Model')
parser.add_argument('--path', required=True, type=str, help='path to source')
args = parser.parse_args()

SUPPORTED_IMAGE_FORMATS = ['jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG']

test_run_id = str(datetime.datetime.utcnow())
eval_dir = config.EVALUATION_RESULT_DIR
eval_img_dir = os.path.join(eval_dir, test_run_id, 'img')
eval_txt_dir = os.path.join(eval_dir, test_run_id, 'txt')
eval_log_dir = os.path.join(eval_dir, test_run_id)
os.makedirs(eval_img_dir, exist_ok=True)
os.makedirs(eval_txt_dir, exist_ok=True)


# creating face detector objects
if config.MTCNN:
	from face_detection.mtcnn import MTCNN
	face_detector = MTCNN()
elif config.SSD_MOBILENET:
	from face_detection.ssd_mobilenet.detector import SSDMobilentFaceDetector
	face_detector = SSDMobilentFaceDetector(config.GPU_MEMORY_FRACTION_TO_USE_DETECTION)
elif config.SSD_INCEPTION:
	from face_detection.ssd_inception.face_detector import SSDInceptionFaceDetector
	face_detector = SSDInceptionFaceDetector(config.GPU_MEMORY_FRACTION_TO_USE_DETECTION)
else:
	raise Exception('Both detector value cannot be True. Please specified only sone!')

# creating face embedding calculator objects
if config.INCEPTION_RESNET:
	from face_embedding.inception_resnet_v1.load_model import InceptionResnet
	face_emb = InceptionResnet(config.GPU_MEMORY_FRACTION_TO_USE_RECOGNITION)
elif config.RESNET_50:
	from face_embedding.resnet_v1.base import ResnetBaseServer
	face_emb = ResnetBaseServer()
else:
	raise Exception('Model for embedding calculation not specified')
	
# face alignment
if config.ALIGN_FACE:
	from align import AlignFace
	align = AlignFace(device='cuda', flip_input=False)
	
if config.HEAD_POSE:
	from face_alignment.hopenet.pose import FaceRotation
	face_rotation = FaceRotation()
	
	
recog = Recognize()
recog.fetch_known_encodings(config.SAVED_ENCODING_FILE)


def calculate_precision_recall(tp, tn, fp, fn):
	"""
	precision-recall curves are appropriate for imbalanced datasets.
	"""
	delta = 1e-3
	precision = tp / (tp + fp + delta)
	recall = tp / (tp + fn + delta)
	f1 = (2 * precision * recall) / (precision + recall + delta)
	accuracy = (tp + tn) / (tp + tn + fp + fn + delta)
	return precision, recall, f1, accuracy


def calculate_AUC(tp, tn, fp, fn):
	"""
	Receiver Operating Characteristic
	ROC curves are appropriate when the observations are balanced between each class, 
	"""
	delta = 1e-3
	# sensitivity = tp / (tp + fn)
	tpr = tp / (tp + fn + delta)

	fpr = fp / (fp + tn + delta)
	# specificity = tn / (tn + fp)
	# fpr = 1 - specificity
	return tpr, fpr


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
			
			import uuid
			os.makedirs('data/eval/align', exist_ok=True)
			raw = [img_bgr[y1:y2, x1:x2, :] for x1, y1, x2, y2 in bounding_boxes]
			aligned = [cv2.cvtColor(face_patch, cv2.COLOR_RGB2BGR) for face_patch in aligned_face_patches]
			[cv2.imwrite(f'data/eval/align/{str(uuid.uuid4())}.jpg', np.concatenate((img1, img2), axis=1)) for img1, img2 in zip(raw, aligned)]
			
		else:
			# without face alignment
			embeddings = face_emb.get_embeddings(img_rgb, bounding_boxes)

		current_persons = recog.find_people_fast(embeddings, config.DISTANCE_THRESHOLD, config.PERCENTGE_THRESHOLD_RECOGNITION, config.DISTANCE_METRIC)
		return bounding_boxes, current_persons
	return [], []


def benchmark_folder_with_single_person_image(source: str):
	"""
	To evaluate on images specify the folder
	"""
	# test samples
	test_samples = os.listdir(source)
	# files
	total_samples = len(test_samples)
	tp, tn, fp, fn = 0, 0, 0, 0
	
	blurr_pic_count = 0
	for file in [f for f in test_samples if f.split('.')[-1] in SUPPORTED_IMAGE_FORMATS]:
		print(f'Processing file: {file}')
		try:
			img_bgr = cv2.imread(os.path.join(source, file))
			# image preprocessing
			is_blurr = preprocessing_utils.blurr_detection(img_bgr, config.BLURR_THRESHOLD)
			# if no blurr is detected then process further or else continue
			if not is_blurr:
				# TODO: CHANGE THE NAME LOGIC BASED ON CURRENT FORMAT
				name_gt = file.strip().split('-')[-1].strip().split('.')[0].lower()
				
				bounding_boxes, current_persons = run_inference(img_bgr)
				
				if len(bounding_boxes) == 1:
					h, w = img_bgr.shape[:2]
					fa = open(os.path.join(eval_log_dir, 'logs.txt'), 'a')
					with open(os.path.join(eval_txt_dir, file.split('.')[0]+'.txt'), 'w') as fw:
						fw.write(f'image: {file} shape: ({h},{w}) persons: {current_persons}')

					for (x1, y1, x2, y2), current_person in zip(bounding_boxes, current_persons):
						uid, name, confidence = current_person
						name = name.lower()

						# TODO: CHANGE THE NAME COMPARISON LOGIC BASED ON CURRENT FORMAT
						for _name in name.split(' '):
							if _name in name_gt.split(' '):
								name_gt = name
								break

						if name_gt in recog.known_names:
							if name_gt == name:
								tp += 1.
							elif name == 'unknown':
								fn += 1.
							elif name_gt != name:
								fp += 1.
								fa.write(f'False Positive 1 - Actual: {name_gt}, Pred: {name} \n')
						else:
							if name == 'unknown':
								tn += 1.
							else:
								fp += 1.
								fa.write(f'False Positive 2 - Actual: {name_gt}, Pred: {name} \n')
						if name == 'unknown':
							drawing_utils.draw_face_box(img_bgr, (x1, y1), (x2, y2), (0, 0, 255), 5, 5)
							cv2.putText(img_bgr, f'{name}: {confidence}', (x1, y1), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0, (255, 255, 255), 2)
						else:
							drawing_utils.draw_face_box(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 5, 5)
							cv2.putText(img_bgr, f'{name}: {confidence}', (x1, y1), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0, (255, 255, 255), 2)
					cv2.imwrite(os.path.join(eval_img_dir, file), img_bgr)
				else:
					print('More than one face detected in evaluation image')
					continue
			else:
				blurr_pic_count += 1
			total_samples += 1
		except Exception as e:
			with open(os.path.join(eval_log_dir, 'logs.txt'), 'a') as fw:
				fw.write(f'Exception occurred while processing file: {file} \n {e} \n\n')
				print(f'Exception occurred while processing file: {file} \n {e}')
			continue
	
	precision, recall, f1, accuracy = calculate_precision_recall(tp, tn, fp, fn)
	
	with open(os.path.join(eval_log_dir, 'logs.txt'), 'a') as fw:
		fw.write(f'Total number of test samples: {total_samples} \n')
		fw.write(f'Blurr frames: {blurr_pic_count} \n')
		fw.write(f'Total number of test samples processed: {total_samples - blurr_pic_count} \n')
		fw.write(f'Total number of registered persons: {len(recog.known_ids)} \n')
		fw.write(f'Total false positive: {fp}, Total false negative: {fn} \n')
		fw.write(f"confidence thershold: {config.PERCENTGE_THRESHOLD_RECOGNITION},  precision: {ceil(precision*100)}, recall: {ceil(recall*100)}, f1: {ceil(f1*100)}, accuracy: {ceil(accuracy*100)} \n")
	
	print('\n\n **************** RESULT **************** \n')
	print(f'Total number of test samples: {total_samples}')
	print(f'Blurr frames: {blurr_pic_count}')
	print(f'Total number of test samples processed: {total_samples - blurr_pic_count}')
	print(f'Total number of registered persons: {len(recog.known_ids)}')
	print(f'Total false positive: {fp}, Total false negative: {fn}')
	print(f"confidence thershold: {config.PERCENTGE_THRESHOLD_RECOGNITION},  precision: {ceil(precision*100)}, recall: {ceil(recall*100)}, f1: {ceil(f1*100)}, accuracy: {ceil(accuracy*100)}") 
	
	
def benchmark_folder_multiple_single_person_image(source: str):
	"""
	Specify a folder of persons
	One folder represent single persons with one or more images
	"""
	# files
	total_samples = 0
	blurr_pic_count = 0
	tp, tn, fp, fn = 0, 0, 0, 0
	
	for root, subfolders, files in os.walk(source):		
		# print(os.listdir(os.path.join(root, subfolders)))
		for folder in [_folder for _folder in subfolders if _folder[0].isalpha() or _folder[0].isdigit()]:
			# TODO: CHANGE NAME/ID LOGIC BASED ON CURRENT DATA/SAVE FORMAT
			if folder.strip()[0].isdigit():
				name_gt = " ".join(folder.strip().split(' ')[1:]).lower()
			else:
				name_gt = folder.strip().lower()
			for file in [f for f in os.listdir(os.path.join(root, folder)) if f.split('.')[-1] in SUPPORTED_IMAGE_FORMATS]:
				print(f'Processing file: {file}')
				try:
					img_bgr = cv2.imread(os.path.join(root, folder, file))
					h, w = img_bgr.shape[:2]
					# image preprocessing
					is_blurr = preprocessing_utils.blurr_detection(img_bgr, config.BLURR_THRESHOLD)
					# if no blurr is detected then process further or else continue
					if not is_blurr:
						
						bounding_boxes, current_persons = run_inference(img_bgr)
						
						if len(bounding_boxes) == 1:
							
							h, w = img_bgr.shape[:2]
							fa = open(os.path.join(eval_log_dir, 'logs.txt'), 'a')
							with open(os.path.join(eval_txt_dir, file.split('.')[0]+'.txt'), 'w') as fw:
								fw.write(f'image: {file} shape: ({h},{w}) persons: {current_persons}')

							for (x1, y1, x2, y2), current_person in zip(bounding_boxes, current_persons):
								uid, name, confidence = current_person
								name = name.lower()

								# TODO: CHANGE THE NAME COMPARISON LOGIC BASED ON CURRENT FORMAT
								for _name in name.split(' '):
									if _name in name_gt.split(' '):
										name_gt = name
										break

								if name_gt in recog.known_names:
									if name_gt == name:
										tp += 1.
									elif name == 'unknown':
										fn += 1.
									elif name_gt != name:
										fp += 1.
										fa.write(f'False Positive 1 - Actual: {name_gt}, Pred: {name} \n')
								else:
									if name == 'unknown':
										tn += 1.
									else:
										fp += 1.
										fa.write(f'False Positive 2 - Actual: {name_gt}, Pred: {name} \n')
								if name == 'unknown':
									drawing_utils.draw_face_box(img_bgr, (x1, y1), (x2, y2), (0, 0, 255), 5, 5)
									# cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)
									cv2.putText(img_bgr, f'{name}: {confidence}', (x1, y1), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0, (255, 255, 255), 2)
								else:
									drawing_utils.draw_face_box(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 5, 5)
									# cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
									cv2.putText(img_bgr, f'{name}: {confidence}', (x1, y1), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0, (255, 255, 255), 2)
						cv2.imwrite(os.path.join(eval_img_dir, file), img_bgr)
					else:
						blurr_pic_count += 1
					total_samples += 1
				except Exception as e:
					with open(os.path.join(eval_log_dir, 'logs.txt'), 'a') as fw:
						fw.write(f'Exception occurred while processing file: {file} \n {e} \n\n')
						print(f'Exception occurred while processing file: {file} \n {e}')
					continue
	
	precision, recall, f1, accuracy = calculate_precision_recall(tp, tn, fp, fn)
	
	with open(os.path.join(eval_log_dir, 'logs.txt'), 'a') as fw:
		fw.write(f'Total number of test samples: {total_samples} \n')
		fw.write(f'Blurr frames: {blurr_pic_count} \n')
		fw.write(f'Total number of test samples processed: {total_samples - blurr_pic_count} \n')
		fw.write(f'Total number of registered persons: {len(recog.known_ids)} \n')
		fw.write(f'Total false positive: {fp}, Total false negative: {fn} \n')
		fw.write(f"confidence thershold: {config.PERCENTGE_THRESHOLD_RECOGNITION},  precision: {ceil(precision*100)}, recall: {ceil(recall*100)}, f1: {ceil(f1*100)}, accuracy: {ceil(accuracy*100)} \n")
	
	print('\n\n **************** RESULT **************** \n')
	print(f'Total number of test samples: {total_samples}')
	print(f'Blurr frames: {blurr_pic_count}')
	print(f'Total number of test samples processed: {total_samples - blurr_pic_count}')
	print(f'Total number of registered persons: {len(recog.known_ids)}')
	print(f'Total false positive: {fp}, Total false negative: {fn}')
	print(f"confidence thershold: {config.PERCENTGE_THRESHOLD_RECOGNITION},  precision: {ceil(precision*100)}, recall: {ceil(recall*100)}, f1: {ceil(f1*100)}, accuracy: {ceil(accuracy*100)}") 


if __name__ == '__main__':
	# dynamic function calling based on the set parameter
# 	getattr(sys.modules[__name__], f"benchmark_folder_with_single_person_image")(args.path)
	benchmark_folder_with_single_person_image(args.path)
