
## Overview

**Face Detection models:**

1. SSD-Mobilenet
2. MTCNN
3. SSD-Inception

**Face Embedding models:**

1. Inception Resnet v1
2. Inception Resnet v2

### Run the code in virtual environment (often ran into poor GPU utilization)

$ `pip3 install -r requirements.txt`

### To run the Face Recognition test
$ `python test.py`

### To run the Face Recognition test on multicam or multiple videos
$ `python multicam.py`

### To benchmark the current Face Recognition architecture
$ `python evaluate.py --path path/to/folder`

### To register new person

There are 2 ways:

1. **Register from webcam:**	$ `python register.py --source webcam --path 0`

2. **Register from folder:**	$ `python register.py --source folder --path path/to/folder`

The folder structure should be:

```
.
├── person1
│   ├── img1.jpg
│   ├── img2.jpg
│	└── ...
├── person2
│	├── img1.jpg
│   ├── img2.jpg
│	└── ...
└── ...
```

### To tune model parameters
You can find the parameters in the `config.py` file.

## APIs

### To run the Face Registration API
$ `python registration_api.py`

### To run the Face Recognition API
$ `python recognition_api.py`

**Note: This code is compatible with Python >= 3.6.0.**
Although it might work with other Python versions with some minor changes
