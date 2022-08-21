from flask import Flask, render_template, request
import os
import cv2
from tensorflow import keras

#### Loading model
model = keras.models.load_model('model.h5')

app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template("index.html")
    

@app.route('/predict', methods = ['post'])
def predict():
    image = request.files['myfile']
    image.save(os.path.join("static", image.filename))
    user_img = cv2.imread('static/'+ image.filename)
    user_img = cv2.resize(user_img,(120,120))
    user_img = user_img/ 255.0
    user_img = user_img.reshape(1,120,120,3)
    print(user_img.shape)

#### Predict
    x = (model.predict(user_img) > 0.5).astype("int32")

    if (x[0][0] == 0):
        label = "Men"
    else:
        label = 'Women'
    
    print(x[0][0])

    return render_template('index.html',label = label, img_path = 'static/' + image.filename)

if __name__ == "__main__":
    app.run(debug=True, port = 8000)

