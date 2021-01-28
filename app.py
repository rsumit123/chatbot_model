
import io
import tensorflow as tf
import flask
from PIL import Image
from keras.models import load_model as l_m
import random
import json
import os
import numpy as np
import pickle
from keras.applications import ResNet50
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

app = flask.Flask(__name__)
intents = json.loads(open('intents.json').read())
orders = json.loads(open('order_db.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = None
graph = None


@app.route('/')
def my_form():
	return flask.render_template('myform.html')


def load_model():
	global graph
	# graph = tf.get_default_graph()
	global model
	model = l_m('chatbot_model.h5')

	# model = ResNet50(weights="imagenet")
################################text preparation##########################
# def prepare_image(image, target):
# 	if image.mode != "RGB":
# 		image = image.convert("RGB")

# 	image = image.resize(target)
# 	image = img_to_array(image)
# 	image = np.expand_dims(image, axis=0)
# 	image = imagenet_utils.preprocess_input(image)

# 	return image


def clean_up_sentence(sentence):
	# tokenize the pattern - split words into array
	sentence_words = nltk.word_tokenize(sentence)
	# stem each word - create short form for word
	sentence_words = [lemmatizer.lemmatize(
		word.lower()) for word in sentence_words]
	return sentence_words


def bow(sentence, words, show_details=True):
	# tokenize the pattern
	sentence_words = clean_up_sentence(sentence)
	# bag of words - matrix of N words, vocabulary matrix
	bag = [0]*len(words)
	for s in sentence_words:
		for i, w in enumerate(words):
			if w == s:
				# assign 1 if current word is in the vocabulary position
				bag[i] = 1
				if show_details:
					print("found in bag: %s" % w)
	return(np.array(bag))


def predict_class(sentence, model):
	# filter out predictions below a threshold
	p = bow(sentence, words, show_details=False)
	global graph
	# with graph.as_default():
	res = model.predict(np.array([p]))[0]

	ERROR_THRESHOLD = 0.25
	results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
	# sort by strength of probability
	results.sort(key=lambda x: x[1], reverse=True)
	return_list = []
	for r in results:
		return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
	return return_list


def get_order_info():
	return "Please Enter order number"



def getResponse(ints, intents_json):
	tag = ints[0]['intent']
	list_of_intents = intents_json['intents']
	for i in list_of_intents:
		if(i['tag'] == tag):
			if tag == "order_details":
				with open("history_d.json","w+") as d:
					json.dump({"history":[{"step":1}]},d)

				result = "Please Enter order number.."
			else:
				result = random.choice(i['responses'])
			break
	return result,tag
				


def chatbot_response(text):
	ints = predict_class(text, model)
	res,tag = getResponse(ints, intents)
	return res,tag

##############################################################################
@app.route("/test", methods=["GET","POST"])
def test():
	data = {"status":200}
	return flask.jsonify(data)


def order_details_1(text):
	with open('order_db.json',"r") as read:
		data = json.load(read)
	for orders in data['orders']:
		if orders["order_number"]== text.strip():
			res = "Current status for "+orders['order_name']+" is: "+orders['order_status']+" | The price for the item is "+orders['order_amount']
			os.remove('history_d.json')


			return res
	else:
		return "Order No Not valid .. Please Enter Again "



	

@app.route("/", methods=["POST"])
def predict():
	data = {"success": False}
	keywords ={1:order_details_1}

	if flask.request.method == "POST":
		# if flask.request.files.get("image"):
			
		# image = flask.request.files["image"].read()
		# image = Image.open(io.BytesIO(image))

		# image = prepare_image(image, target=(224, 224))

		# text = flask.request.get_json(force=True)
		text = flask.request.form['text']
		data['query'] = text
		try:
			with open('history_d.json','r') as r:
				read_d = json.load(r)
			res=keywords[read_d['history'][-1]['step']](text)
		except Exception as e:
			print(e)
			
			res,tag = chatbot_response(text)
			data['tag'] = tag


		
		# global graph
		# with graph.as_default():
		# 	preds = model.predict(image)
		# results = imagenet_utils.decode_predictions(preds)
		# data["predictions"] = []

		
		# for (imagenetID, label, prob) in results[0]:
		# 	r = {"label": label, "probability": float(prob)}
		# 	data["predictions"].append(r)

			
		# 	data["success"] = True
		data['success']= True
		data['response']= res


	return flask.jsonify(data)

if __name__ == "__main__":
	print(("* Loading Keras model and Flask starting server..."
		"please wait until server has fully started"))
	load_model()
	app.run(host='0.0.0.0', port=5000,debug=True)
