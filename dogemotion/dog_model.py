from keras.models import load_model
import dlib
import cv2
from imutils import face_utils
import imutils
import math
import numpy as np
import pandas as pd

class DogModel: 
    def __init__(self, landmark_detector_path, dog_head_detector_path, model_path):
        # ---------------------------------------------------------
        # landmarks detector
        pathPred = landmark_detector_path
        self.predictor = dlib.shape_predictor(pathPred)

        # face detector
        pathDet = dog_head_detector_path
        self.detector = dlib.cnn_face_detection_model_v1(pathDet)

        # load model
        pathModel = model_path
        self.model = load_model(pathModel)
        # ---------------------------------------------------------
    
    def predict(self, image_path):
        
        # image size for prediction
        img_width = 100
        img_height = 100

        # scale factor for preprocessing
        picSize = 200
        rotation = True
        def predict_emotion(model, img):
            """
            Use a trained model to predict emotional state
            """

            prediction = model.predict(img)
            
            d = {'emotion': ['Angry', 'Scared', 'Happy', 'Sad'],
                'prob': prediction[0]}
            df = pd.DataFrame(d, columns=['emotion', 'prob'])

            return df

        def preprocess(path):
            """
              returns the first dog face found
            """
            # read image from path
            orig = cv2.imread(path)

            if orig.any() == True:
                # resize
                height, width, channels = orig.shape  # read size
                ratio = picSize / height
                image = cv2.resize(orig, None, fx=ratio, fy=ratio)

                # detect face(s)
                dets = self.detector(image, upsample_num_times=1)

                imageList = []  # for return
                for i, d in enumerate(dets):
                    # save coordinates
                    x1 = max(int(d.rect.left() / ratio), 1)
                    y1 = max(int(d.rect.top() / ratio), 1)
                    x2 = min(int(d.rect.right() / ratio), width - 1)
                    y2 = min(int(d.rect.bottom() / ratio), height - 1)

                    # detect landmarks
                    shape = face_utils.shape_to_np(self.predictor(image, d.rect))
                    points = []
                    index = 0
                    for (x, y) in shape:
                        x = int(round(x / ratio))
                        y = int(round(y / ratio))
                        index = index + 1
                        if index == 3 or index == 4 or index == 6:
                            points.append([x, y])
                    points = np.array(points)  # right eye, nose, left eye

                    # rotate
                    if rotation == True:
                        xLine = points[0][0] - points[2][0]
                        if points[2][1] < points[0][1]:
                            yLine = points[0][1] - points[2][1]
                            angle = math.degrees(math.atan(yLine / xLine))
                        else:
                            yLine = points[2][1] - points[0][1]
                            angle = 360 - math.degrees(math.atan(yLine / xLine))
                        rotated = imutils.rotate(orig, angle)
                        # detectFace(rotated, picSize)

                    # highlight face and landmarks
                    cv2.polylines(orig, [points], True, (0, 255, 0), 1)
                    cv2.rectangle(orig, (x1, y1), (x2, y2), (255, 0, 0), 1)
                    imageList.append(orig)

                    # prepare for prediction
                    # crop and resize
                    little = cv2.resize(
                        (rotated[y1:y2, x1:x2]), (img_width, img_height))
                    x = np.expand_dims(little, axis=0)
                    x = x.reshape((-1, 100, 100, 3))
                    imageList.append(x)
                    return (imageList, ((x1, y1), (x2, y2))) # order: marked picture, input for classifier
            return None

        # read and preprocess image
        images, coordinate = preprocess(image_path)
        if images != None:  # found face on image
            x = images[1]

            df = predict_emotion(self.model, x)

            # sort and extract most probable emotion
            df = df.sort_values(by='prob', ascending=False)
            # Take a dictionary as input to your DataFrame 
            coordinate_dict = {"emotion": ['Coordinate'], "prob": [coordinate]}
            df = df.append(pd.DataFrame(coordinate_dict))
            js = df.to_json(orient = 'records')
            return js
        return None

# import time
# start = time.time()

# image_path = "/home/team14/creme-ai/dogemotion/dog.jpg"
# landmark_detector_path = "/home/team14/creme-ai/dogemotion/landmarkDetector.dat"
# dog_head_detector_path = "/home/team14/creme-ai/dogemotion/dogHeadDetector.dat"
# model_path = "/home/team14/creme-ai/dogemotion/ResNet50_LR00005_BS16_EP30_L2_newdata.h5"

# test = DogModel(landmark_detector_path, dog_head_detector_path, model_path)
# print(test.predict(image_path))

# print("time :", time.time() - start)