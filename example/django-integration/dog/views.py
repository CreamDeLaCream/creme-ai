from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse
from dogemotion import dog_model
from rmn import human_model


def index(request):
    image_path = "sample image path"
    landmark_detector_path = "landmark detector path"
    dog_head_detector_path = "dog head detector path"
    model_path = "model path"
    human_image_path = "sample human image path"

    result = dog_model.DogModel(
        image_path, landmark_detector_path, dog_head_detector_path, model_path).predict()

    human_result = human_model.RMN.detect_emotion_for_single_frame(
        human_image_path)
    return HttpResponse(result)
