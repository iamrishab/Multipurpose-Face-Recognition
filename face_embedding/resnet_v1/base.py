import os
import cv2
import numpy as np
import tensorflow as tf
import tensorlayer as tl

# from losses.face_losses import arcface_loss
from .nets.L_Resnet_E_IR_fix_issue9 import get_resnet
# from nets.L_Resnet_E_IR import get_resnet


from pdb import set_trace


class ResnetBaseServer():
    def __init__(self):
        self.image_placeholder = tf.placeholder(name='img_inputs', shape=[None, 112, 112, 3], dtype=tf.float32)
        labels = tf.placeholder(name='img_labels', shape=[None, ], dtype=tf.int64)
        dropout_rate = tf.placeholder(name='dropout_rate', dtype=tf.float32)
        w_init_method = tf.contrib.layers.xavier_initializer(uniform=False)
        net = get_resnet(self.image_placeholder, 50, type='ir', w_init=w_init_method, trainable=False, keep_rate=dropout_rate)
        self.embedding_tensor = net.outputs
        
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        tf.global_variables_initializer().run(session=self.sess)
        saver = tf.train.Saver()
        path = '/home/rishab/models/baidu_model/InsightFace_iter_best_710000.ckpt'
        saver.restore(self.sess, path)
        
        self.feed_dict_infer = {}
        self.feed_dict_infer.update(tl.utils.dict_to_one(net.all_drop))
        self.feed_dict_infer[dropout_rate] = 1.0
                           
    def normalize(self, patch):
        # standardize pixel values across channels (global)
        tmp = patch.copy()
        tmp = tmp.astype(np.float32)
        tmp -= 127.5
        tmp *= 0.0078125
        return patch
    
    def get_embeddings(self, frame_rgb, bounding_boxes):
        faces = []
        for x1, y1, x2, y2 in bounding_boxes:
            face_patch = frame_rgb[y1:y2, x1:x2, :]
            resized = cv2.resize(face_patch, (112, 112), interpolation=cv2.INTER_AREA)
            normalize = self.normalize(resized)
            faces.append(normalize)
        faces = np.stack(faces)
        # Run prediction
        self.feed_dict_infer[self.image_placeholder] = faces
        embeddings = self.sess.run(self.embedding_tensor, self.feed_dict_infer)
        return embeddings