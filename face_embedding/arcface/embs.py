import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import numpy as np
import tensorflow as tf

from .modules.models import ArcFaceModel
from .modules.utils import set_memory_growth, load_yaml, l2_norm


class ArcFaceResNet50:
    def __init__(self):
        set_memory_growth()
        self.cfg = load_yaml(os.path.join(os.path.dirname(os.path.abspath(__file__)), \
                                     './configs/arc_res50.yaml'))

        self.model = ArcFaceModel(size=self.cfg['input_size'],
                             backbone_type=self.cfg['backbone_type'],
                             training=False)

        ckpt_path = tf.train.latest_checkpoint(os.path.join(os.path.dirname(os.path.abspath(__file__)), \
                                                            './checkpoints/' + self.cfg['sub_name']))
        if ckpt_path is not None:
            print("[*] load ckpt from {}".format(ckpt_path))
            self.model.load_weights(ckpt_path)
        else:
            print("[*] Cannot find ckpt from {}.".format(ckpt_path))
            exit()
        
    def get_embeddings(self, frame_rgb, bounding_boxes):
            faces = []
            for x1, y1, x2, y2 in bounding_boxes:
                face_patch = frame_rgb[y1:y2, x1:x2, :]
                resized = cv2.resize(face_patch, (self.cfg['input_size'], self.cfg['input_size']), interpolation=cv2.INTER_AREA)
                normalize = resized.astype(np.float32) / 255.
                faces.append(normalize)
            faces = np.stack(faces)
            if len(faces.shape) == 3:
                faces = np.expand_dims(faces, 0)
            # Run prediction
            embeddings = l2_norm(self.model(faces))
            return embeddings