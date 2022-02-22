# -*- coding: utf-8 -*-
"""rmn_demo.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ZEbJ6rJuGZ6UzmOFE1XCzSgacTUJiN-H

# Facial Expression Recognition using Residual Masking Network
"""

# !pip install rmn==3.0.0

# !curl https://user-images.githubusercontent.com/24642166/117108145-f7e6a480-adac-11eb-9786-a3e50a3cdea6.jpg -o image.png

from rmn import RMN
import cv2
from IPython.display import Image
Image('image.png', width=400)


m = RMN()

image = cv2.imread("image.png")
assert image is not None
results = m.detect_emotion_for_single_frame(image)

print(results)

image = m.draw(image, results)
cv2.imwrite("output.png", image)

Image("output.png", width=400)