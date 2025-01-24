# from flask import Flask, render_template, request, url_for, Markup, jsonify
# import pickle
# import pandas as pd
# import numpy as np
# import pandas as pd
# import numpy as np
# import sys
# import os
# import glob
# import re
# import numpy as np
# import tensorflow as tf
# import tensorflow as tf
# from flask import Markup


# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession

# config = ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.2
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)
# # Keras
# from tensorflow.keras.applications.resnet50 import preprocess_input
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# import numpy as np
# import pandas as pd
# import pickle
# from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
# import keras.models
# from keras.models import model_from_json

# # Flask utils
# from flask import Flask, redirect, url_for, request, render_template
# from werkzeug.utils import secure_filename
# from keras.preprocessing.text import Tokenizer
# from keras.preprocessing import sequence
# from keras.models import Model, Input, Sequential, load_model
# import pickle
# import h5py

# # create Flask application
# app = Flask(__name__)

# # read object TfidfVectorizer and model from disk
# MODEL_PATH ='cnn.h5'
# model = load_model(MODEL_PATH)
 
# with open('tokenizer.pickle', 'rb') as handle:
#     tokenizer = pickle.load(handle )


# @app.route('/')
# @app.route('/first') 
# def first():
# 	return render_template('first.html')
# @app.route('/login') 
# def login():
# 	return render_template('login.html')    
    
 
 
# @app.route('/upload') 
# def upload():
# 	return render_template('upload.html') 
# @app.route('/preview',methods=["POST"])
# def preview():
#     if request.method == 'POST':
#         dataset = request.files['datasetfile']
#         df = pd.read_csv(dataset,encoding = 'unicode_escape')
#         df.set_index('Id', inplace=True)
#         return render_template("preview.html",df_view = df)    

 
# @app.route('/home')
# def home():
#     return render_template('index.html')


# @app.route('/predict', methods=['POST'])
# def predict():
#     error = None
#     if request.method == 'POST':
#         # message
#         msg = request.form['message']
#         msg = pd.DataFrame(index=[0], data=msg, columns=['data'])

#         # transform data
#         new_text = sequence.pad_sequences((tokenizer.texts_to_sequences(msg['data'].astype('U'))), maxlen=547)
          
#         # model
#         result = model.predict(new_text,batch_size=1,verbose=2)
         
#         if result >0.5:
#             result = 'Fake'
#         else:
#             result = 'Real'

#         return render_template('index.html', prediction_value=result)
#     else:
#         error = "Invalid message"
#         return render_template('index.html', error=error)
# @app.route('/chart') 
# def chart():
# 	return render_template('chart.html')

# if __name__ == "__main__":
#     app.run()
from flask import Flask, render_template, request, url_for, jsonify
from markupsafe import Markup
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto, InteractiveSession
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
# Set up GPU configuration
config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# Initialize the Flask app
app = Flask(__name__)

# Load the model
MODEL_PATH = 'cnn.h5'
model = load_model(MODEL_PATH)

# Verify tokenizer file
TOKENIZER_PATH = 'tokenizer.pkl'


tfvect = TfidfVectorizer(stop_words='english', max_df=0.7)
loaded_model = pickle.load(open('model.pkl', 'rb'))
dataframe = pd.read_csv('news.csv')
x = dataframe['text']
y = dataframe['label']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

def fake_news_det(news):
    tfid_x_train = tfvect.fit_transform(x_train)
    tfid_x_test = tfvect.transform(x_test)
    input_data = [news]
    vectorized_input_data = tfvect.transform(input_data)
    prediction = loaded_model.predict(vectorized_input_data)
    return prediction





if not os.path.exists(TOKENIZER_PATH) or os.path.getsize(TOKENIZER_PATH) == 0:
    print("Tokenizer file does not exist or is empty. Recreating the tokenizer.")
    from tensorflow.keras.preprocessing.text import Tokenizer
    texts = ["example text data", "more text data for tokenizer"]
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(texts)
    with open(TOKENIZER_PATH, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=4)
else:
    print("Tokenizer file exists and is not empty.")

# Load the tokenizer
with open(TOKENIZER_PATH, 'rb') as handle:
    tokenizer = pickle.load(handle)

@app.route('/')
@app.route('/first')
def first():
    return render_template('first.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/preview', methods=["POST"])
def preview():
    if request.method == 'POST':
        dataset = request.files['datasetfile']
        df = pd.read_csv(dataset, encoding='unicode_escape')
        df.set_index('Id', inplace=True)
        return render_template("preview.html", df_view=df)

@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        pred = fake_news_det(message)
        print(pred)
        return render_template('index.html', prediction=pred)
    else:
        return render_template('index.html', prediction="Something went wrong")

@app.route('/chart')
def chart():
    return render_template('chart.html')

if __name__ == "__main__":
    app.run(debug=True)
