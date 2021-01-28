from flask import Flask, escape, request,jsonify,render_template
import numpy as np
import PIL
from PIL import Image
from keras.models import load_model
import tensorflow as tf
 
classes_dict={1:"car",2:"bird",3:"cat",4:"deer",5:"dog",6:"frog",7:"horse",8:"ship",9:"truck",0:"Aeroplane"}
app = Flask(__name__)
model = None
graph = None
def load_model_1():
    global model
    global graph
    model = load_model('object_Classifications.h5')
    graph = tf.get_default_graph()

@app.route('/')  
def upload():  
    return ''' <html>  
<head>  
    <title>upload</title>  
</head>  
<body>  
    <form action = "/success" method = "post" enctype="multipart/form-data">  
        <input type="file" name="file" />  
        <input type = "submit" value="Upload">  
    </form>  
</body>  
</html>'''  

@app.route('/success', methods=['POST'])
def predict_image():
    if request.method == 'POST':  
        
       # Preprocess the image so that it matches the training input
       image = request.files['file']
       image = Image.open(image)
       image = np.asarray(image.resize((32,32)))
       
       # Use the loaded model to generate a prediction.
     
       #x = image.img_to_array(image)
       #print(image.shape)
       
       x = np.expand_dims(image, axis=0)/255.0
       
        
       images = np.vstack([x])
       
       global graph
       with graph.as_default():
           classes = model.predict_classes(images, batch_size=10)
       # Prepare and send the response.
       
       prediction = {'Object':classes_dict[classes[0]]}
       return jsonify(prediction)
       #return '''<script>alert('''+x+''')</script>'''
if __name__=="__main__":
    load_model_1()
    app.run()
