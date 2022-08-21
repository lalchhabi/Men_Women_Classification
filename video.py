import cv2
from tensorflow import keras
from tensorflow.keras.applications.vgg16 import decode_predictions
import time
from keras.preprocessing import image
import numpy as np

model = keras.models.load_model('model.h5')

def predict(frame):
    test_image = cv2.resize(frame,(120,120))
    test_image = test_image.reshape(1,120,120,3)
    # test_image = image.img_to_array(test_image)
    # test_image = np.expand_dims(test_image, axis = 0)
    result = model.predict(test_image)
    if result[0][0] == 1:
        prediction = 'Women'
        print(prediction)
    else:
        prediction = 'Men'
        print(prediction)


cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    predict(frame)
    cv2.imshow("hello", frame)
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break
cv2.destroyAllWindows()
