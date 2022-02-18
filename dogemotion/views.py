from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse
from dogemotion import DogModel
# import os
# module_dir = os.path.dirname(__file__)
# file_path = os.path.join(module_dir, 'landmarkDetector.dat')
# print(file_path)

def index(request):
    image_path = "/home/soyounghyun/ml-django/dogemotion/dog.jpg"
    landmark_detector_path = "/home/soyounghyun/ml-django/dogemotion/landmarkDetector.dat"
    dog_head_detector_path = "/home/soyounghyun/ml-django/dogemotion/dogHeadDetector.dat"
    model_path = "/home/soyounghyun/ml-django/dogemotion/classifierRotatedOn100Ratio90Epochs100.h5"
    
    result = DogModel.DogModel(image_path,landmark_detector_path,dog_head_detector_path,model_path).predict()

    return HttpResponse(result)
