
## Overview

**Face Detection models:**

1. SSD-Mobilenet
2. MTCNN
3. SSD-Inception

**Face Embedding models:**

1. Inception Resnet v1

### Clone the the repository

$ `git clone https://rishab-pal-onebcg@bitbucket.org/onebcg/facialrecognitionv1.git`

### Run the code in virtual environment (often ran into poor GPU utilization)
$ `pip3 install virtualenv`

$ `virtualenv fr_env`

$ `source fr_env/bin/activate`

$ `pip3 install -r requirements.txt`

$ `cd facialrecognitionv1`

$ `git checkout <branch-name>`

### Run the code in CONDA environment (recommended)

$ `wget https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh`

$ `sh Anaconda3-2020.02-Linux-x86_64.sh`

$ `source anaconda3/bin/activate`

$ `conda create -n fr_env python==3.6.8`

$ `conda install -c anaconda tensorflow-gpu==1.13.*`

$ `conda install -c conda-forge opencv`

$ `conda install -c anaconda scikit-learn`

$ `conda install -c anaconda scikit-image`

$ `conda install -c anaconda numpy==1.16.4`

$ `cd facialrecognitionv1`

$ `git checkout <branch-name>`

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