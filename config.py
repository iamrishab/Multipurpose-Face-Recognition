# mode
DEBUG = True

# video source
CAM_SOURCES = {
				'CAM1' : 'rtsp://deepak.dey:DeepakDeyFR@192.168.7.144:554/cam/realmonitor?channel=1&subtype=0',
				'CAM2' : 'rtsp://deepak.dey:DeepakDeyFR@192.168.7.145:554/cam/realmonitor?channel=1&subtype=0'
}

TEST_SOURCES = {
				'CAM1':'/home/rishab/data/test/1.mkv',
				'CAM2':'/home/rishab/data/test/2.mkv'		
}

VIDEO_SOURCE = '/home/rishab/data/test/1.mkv'

# camera parameters
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

EVALUATION_RESULT_DIR = 'data/eval'
FOLDER_TO_REGISTERED = 'data/registered'
UPLOAD_IMAGE_FOLDER = 'data/upload'
# path to saved encoding file
SAVED_ENCODING_FILE = 'data/pickle/person_data.pickle'
SAVED_ENCODING_FILE_MULTIPLE = 'data/pickle/person_data_multiple.pickle'

# Choose which Face Detector to use
HAAR = False
MTCNN = False
SSD_MOBILENET = False
SSD_INCEPTION = True

# Choose which Face Embedding Calculator to use
INCEPTION_RESNET = True
RESNET_50 = False

# face detection HAAR CASCADE
SCALE_FACTOR = 1.1
MIN_NEIGHBOURS = 5

# face detection
GPU_MEMORY_FRACTION_TO_USE_DETECTION = 0.25
FACE_DETECTION_CONFIDENCE = 0.9
MIN_FACE_SIZE = 40
FACE_PADDING_RATIO = 0.10
FRAME_RESIZE_FACTOR = 2

# frontal face check
ALIGN_FACE = False
HEAD_POSE = False
MIN_NORMALIZED_DISTANCE_BETWEEN_EYES = 0.25
MIN_NORMALIZED_DISTANCE_BETWEEN_MOUTH = 0.15

# face embedding
GPU_MEMORY_FRACTION_TO_USE_RECOGNITION = 0.25

# Face Comparison/Recognition
DISTANCE_THRESHOLD = 1.32
PERCENTGE_THRESHOLD_REGISTRATION = 80
PERCENTGE_THRESHOLD_RECOGNITION = 50
PERCENTGE_THRESHOLD_MULTIPLE_RECOGNITION = 45
DISTANCE_METRIC = 0 # 0 for Euclidean distance and 1 for Cosine Similarity

DISTANCE_THRESHOLDS = {
    'inception_resnet_v1':{
            'DISTANCE_THRESHOLD': 0.50,
            'PERCENTGE_THRESHOLD_REGISTRATION': 80,
            'PERCENTGE_THRESHOLD_RECOGNITION': 50,
            'PERCENTGE_THRESHOLD_MULTIPLE_RECOGNITION': 45,
            'DISTANCE_METRIC': 0, # 0 for Euclidean distance and 1 for Cosine Similarity
    },
    'arcface':{
            'DISTANCE_THRESHOLD': 1.32,
            'PERCENTGE_THRESHOLD_REGISTRATION': 80,
            'PERCENTGE_THRESHOLD_RECOGNITION': 50,
            'PERCENTGE_THRESHOLD_MULTIPLE_RECOGNITION': 45,
            'DISTANCE_METRIC': 0, # 0 for Euclidean distance and 1 for Cosine Similarity
    }
}

# IMAGE PREPROCESSING
# blurr threshold
BLURR_THRESHOLD = 10
# whether to apply image adjustments
PREPROCESS_IMAGE = False
# applying adaptive histogram equalization
CLAHE_GRID_SIZE = 8
# clip hist percent
CLIP_HIST_PERCENTAGE = 15

# API
REGISTRATION_IP = '0.0.0.0'
REGISTRATION_PORT = 8001

RECOGNITION_IP = '0.0.0.0'
RECOGNITION_PORT = 8002
