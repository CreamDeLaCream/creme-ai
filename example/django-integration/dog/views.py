from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse
from dogemotion import dog_model
from humanemotion import human_model
import cv2


def index(request):
    image_path = "sample image path"
    landmark_detector_path = "landmark detector path"
    dog_head_detector_path = "dog head detector path"
    model_path = "model path"
    dog_result = dog_model.DogModel(image_path, landmark_detector_path, dog_head_detector_path, model_path).predict()

    human_image_path = "humanemotion/image.png"
    img = cv2.imread(human_image_path)
    human_result = human_model.RMN().detect_emotion_for_single_frame(img)

    result = dog_result ,human_result

    return HttpResponse(result)
