#! /bin/bash

set -e
cd /home/facialrecognitionv1;/home/fr_env/bin/python recognition_api.py &
cd /home/facialrecognitionv1;/home/fr_env/bin/python registration_api.py &