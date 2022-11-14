from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg19 import VGG19,preprocess_input,decode_predictions
import tensorflow
import numpy as np


app = Flask(__name__)




def predict_label(img_path):
	
	model = load_model('best_model.h5')

	model.make_predict_function()
	i = tensorflow.keras.utils.load_img(img_path, target_size=(256,256))
	i = tensorflow.keras.utils.img_to_array(i)
	i = i.reshape(1, 256,256,3)
	pred=np.argmax(model.predict(i))
	return pred

def predict_label1(img_path):
	
	model = load_model('veg.h5')

	model.make_predict_function()
	i = tensorflow.keras.utils.load_img(img_path, target_size=(256,256))
	i = tensorflow.keras.utils.img_to_array(i)
	i = i.reshape(1, 256,256,3)
	pred=np.argmax(model.predict(i))
	return pred

# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/about")
def about_page():
	return "Please subscribe  Artificial Intelligence Hub..!!!"

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['image']
		plant= request.form['plant']
		img_path = "static/" + img.filename	
		img.save(img_path)
		if(plant =='vegetable'):
			p = predict_label1(img_path)
			print(p)
			return render_template("submit1.html", prediction = p, img_path = img_path)
		if(plant =='fruit'):
			p = predict_label(img_path)
			print(p)
			return render_template("submit.html", prediction = p, img_path = img_path)
	


if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)