import os
import sys
import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F

from . import model, utils


class FaceRotation(object):
	def __init__(self):
		self.device_id = torch.cuda.current_device()
		# self.device = 'gpu' if torch.cuda.is_available() else 'cpu'
		cudnn.enabled = True
		
		snapshot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pretrained/hopenet_robust_alpha1.pkl')

		# ResNet50 structure
		self.model = model.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)

		# Load snapshot
		saved_state_dict = torch.load(snapshot_path)
		self.model.load_state_dict(saved_state_dict)

		self.transformations = transforms.Compose([transforms.Scale(224),
		transforms.CenterCrop(224), transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
		
		self.model.cuda(self.device_id)

		# Test the Model
		self.model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
		
		self.idx_tensor = torch.FloatTensor([idx for idx in range(66)]).cuda(self.device_id)
		
	
	def pose_estimation(self, face):
		face = Image.fromarray(np.uint8(face))
		# Transform
		face = self.transformations(face)
		face_shape = face.size()
		face = face.view(1, face_shape[0], face_shape[1], face_shape[2])
		
		face = Variable(face).cuda(self.device_id)

		yaw, pitch, roll = self.model(face)
		
		yaw_predicted = F.softmax(yaw, dim=1)
		pitch_predicted = F.softmax(pitch, dim=1)
		roll_predicted = F.softmax(roll, dim=1)
		# Get continuous predictions in degrees.
		yaw_predicted = torch.sum(yaw_predicted.data[0] * self.idx_tensor) * 3 - 99
		pitch_predicted = torch.sum(pitch_predicted.data[0] * self.idx_tensor) * 3 - 99
		roll_predicted = torch.sum(roll_predicted.data[0] * self.idx_tensor) * 3 - 99
		return yaw_predicted, pitch_predicted, roll_predicted
		
	def get_rotation(self, img, bounding_boxes):
		poses = []
		for x1, y1, x2, y2 in bounding_boxes:
			face = img[y1:y2, x1:x2, :]
			yaw_predicted, pitch_predicted, roll_predicted = self.pose_estimation(face)
			yaw_predicted, pitch_predicted, roll_predicted = float(yaw_predicted.cpu().numpy()), float(pitch_predicted.cpu().numpy()), float(roll_predicted.cpu().numpy())
			
			# pitch_predicted
			if pitch_predicted > 25:
				pitch = "up"
			elif pitch_predicted < -25:
				pitch = "down"
			else:
				pitch = "front"

			# rool_pose
			if roll_predicted > 35:
				roll = "left"
			elif roll_predicted < -15:
				roll = "right"
			else:
				roll = "front"

			# yaw_predicted
			if yaw_predicted > 0:
				yaw = "right"
			elif yaw_predicted < -45:
				yaw = "left"
			else:
				yaw = "front"
			
			poses.append([(yaw, yaw_predicted), (pitch, pitch_predicted), (roll, roll_predicted)])

		return poses