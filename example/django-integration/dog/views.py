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
    human_image_path = "humanemotion/image.png"

    result = dog_model.DogModel(
        image_path, landmark_detector_path, dog_head_detector_path, model_path).predict()
    m = human_model.RMN()
    img = cv2.imread(human_image_path)
    human_result = m.detect_emotion_for_single_frame(
        img)
    return HttpResponse(result, human_result)
