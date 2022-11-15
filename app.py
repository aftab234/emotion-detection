# DataFrame
import pandas as pd

# Matplot
import matplotlib.pyplot as plt

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer

# Keras
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Conv1D, MaxPooling1D, LSTM
from tensorflow.keras import utils
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# nltk
import nltk
from nltk.corpus import stopwords
from  nltk.stem import SnowballStemmer

# Word2vec
#import gensim

# Utility
import re
import numpy as np
import os
from collections import Counter
import logging
import time
import pickle
import itertools

# Set log
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import cv2
import pyaudio
import wave

import os
import pandas as pd 
import numpy as np 
import flask
import pickle
from flask import Flask, render_template, request

#from save_audio import save_audio
#from configure import auth_key
import pandas as pd
from time import sleep
import urllib.request
import plotly.express as px
import plotly.graph_objects as go
from urllib.request import urlopen
from bs4 import BeautifulSoup
import json
import requests

import numpy as np
import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import pandas as pd
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

import pylab
import imageio

import tensorflow
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing import image
import cv2
import numpy as np

# DATASET
DATASET_COLUMNS = ["target", "ids", "date", "flag", "user", "text"]
DATASET_ENCODING = "ISO-8859-1"
TRAIN_SIZE = 0.8

# TEXT CLENAING
TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

# WORD2VEC 
W2V_SIZE = 300
W2V_WINDOW = 7
W2V_EPOCH = 32
W2V_MIN_COUNT = 10

# KERAS
SEQUENCE_LENGTH = 300
EPOCHS = 8
BATCH_SIZE = 1024

# SENTIMENT
POSITIVE = "POSITIVE"
NEGATIVE = "NEGATIVE"
NEUTRAL = "NEUTRAL"
SENTIMENT_THRESHOLDS = (0.4, 0.7)

# EXPORT
KERAS_MODEL = "model.h5"
WORD2VEC_MODEL = "model.w2v"
TOKENIZER_MODEL = "tokenizer.pkl"
ENCODER_MODEL = "encoder.pkl"

cap=0

app=Flask(__name__)

@app.route('/')
def index():
 return flask.render_template('index.html')

 
def decode_sentiment(score, include_neutral=True):
    if include_neutral:        
        label = NEUTRAL
        if score <= SENTIMENT_THRESHOLDS[0]:
            label = NEGATIVE
        elif score >= SENTIMENT_THRESHOLDS[1]:
            label = POSITIVE

        return label
    else:
        return NEGATIVE if score < 0.5 else POSITIVE 

def predict(text_input_args, include_neutral=True):
    model = tensorflow.keras.models.load_model("model.h5")
    encoder = pickle.load(open("encoder.pkl","rb"))
    tokenizer = pickle.load(open("tokenizer.pkl","rb"))
    w2v_model = pickle.load(open("w2v_model_ML.pkl","rb"))  
    start_at = time.time()
    text_input=str(text_input_args)
    # Tokenize text
    x_test = pad_sequences(tokenizer.texts_to_sequences([text_input]), maxlen=SEQUENCE_LENGTH)
    # Predict
    score = model.predict([x_test])[0]
    # Decode sentiment
    label = decode_sentiment(score, include_neutral=include_neutral)
    return {"label": label, "score": float(score),
       "elapsed_time": time.time()-start_at}  

def get_key(value):
    dictionary={'joy':0,'anger':1,'love':2,'sadness':3,'fear':4,'surprise':5}
    for key,val in dictionary.items():
          if (val==value):
            return key

def predict_text(sentence):
    text_model = tensorflow.keras.models.load_model("text_model.h5")
    tokenizer2 = pickle.load(open("tokenizer2.pkl","rb"))
    sentence_lst=[]
    sentence_lst.append(sentence)
    sentence_seq=tokenizer2.texts_to_sequences(sentence_lst)
    sentence_padded=pad_sequences(sentence_seq,maxlen=80,padding='post')
    predict_x=text_model.predict(sentence_padded) 
    classes_x=np.argmax(predict_x,axis=1)
    ans=get_key(classes_x)
    return ans
    
@app.route('/real_time',methods = ['POST'])
def real_time():
    if request.method == 'POST':
        face_classifier=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        classifier = load_model('ferNet.h5')

        class_labels=['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']
        cap=cv2.VideoCapture(0)

        while True:
            ret,frame=cap.read()
            labels=[]
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            faces=face_classifier.detectMultiScale(gray,1.3,5)

            for (x,y,w,h) in faces:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                roi_gray=gray[y:y+h,x:x+w]
                roi_gray=cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)

                if np.sum([roi_gray])!=0:
                    roi=roi_gray.astype('float')/255.0
                    roi=img_to_array(roi)
                    roi=np.expand_dims(roi,axis=0)

                    preds=classifier.predict(roi)[0]
                    label=class_labels[preds.argmax()]
                    label_position=(x,y)
                    cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
                else:
                    cv2.putText(frame,'No Face Found',(20,20),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
    
            cv2.imshow('Emotion Detector',frame)
            if cv2.waitKey(1) & 0xFF == ord('x'):
                break
        cap.release()
        cv2.destroyAllWindows()
        return render_template("real_time.html")

@app.route('/face_analysis',methods = ['POST'])
def frame_analysis():
    if request.method == 'POST':
        cap = cv2.VideoCapture("cam_video_01.avi")
        ret, frame = cap.read()
        count = 0
        length_f = int(cap.get(cv2. CAP_PROP_FRAME_COUNT))
        for num in range(1,length_f,3):
            cv2.imwrite("static/frames/frame%d.jpg" % num, frame)     # save frame as JPEG file      
            ret,frame = cap.read()
        filename = 'cam_video_01.avi'
        vid = imageio.get_reader(filename,  'ffmpeg')
        face_classifier=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        classifier = tensorflow.keras.models.load_model('ferNet.h5')

        class_labels=['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']
        labels=[0]
        freq={'Angry': 0, 'Disgust': 0, 'Fear': 0, 'Happy': 0, 'Neutral': 0, 'Sad': 0, 'Surprise': 0}
        for num in range(1,length_f):
            image = vid.get_data(num)
            gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            faces=face_classifier.detectMultiScale(gray,1.3,5)
            for (x,y,w,h) in faces:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                roi_gray=gray[y:y+h,x:x+w]
                roi_gray=cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
                if np.sum([roi_gray])!=0:
                    roi=roi_gray.astype('float')/255.0
                    roi=img_to_array(roi)
                    roi=np.expand_dims(roi,axis=0)
                    preds=classifier.predict(roi)[0]
                    label=class_labels[preds.argmax()]
                    labels.append(label)
                    freq[label]= freq.get(label, 0)+1
                else:
                    continue
        total_em = freq['Angry'] + freq['Disgust'] + freq['Fear'] + freq['Happy'] + freq['Neutral'] +  freq['Sad'] +  freq['Surprise']
        percent=[0,0,0,0,0,0,0]
        percent[0]=(freq['Angry']/total_em)*100
        percent[1]=(freq['Disgust']/total_em)*100
        percent[2]=(freq['Fear']/total_em)*100
        percent[3]=(freq['Happy']/total_em)*100
        percent[4]=(freq['Neutral']/total_em)*100
        percent[5]=(freq['Sad']/total_em)*100
        percent[6]=(freq['Surprise']/total_em)*100
        images = os.listdir(os.path.join(app.static_folder, "frames"))
        return render_template("face_analysis.html", per0=round(percent[0],2), per1=round(percent[1],2), per2=round(percent[2],2), per3=round(percent[3],2), per4=round(percent[4],2), per5=round(percent[5],2), per6=round(percent[6],2), images=images, labels=labels)
        
def label(image):
        img = image.load_img(image,target_size = (48,48),color_mode = "grayscale")
        img = np.array(img)
        label_dict = {0:'Angry',1:'Disgust',2:'Fear',3:'Happy',4:'Neutral',5:'Sad',6:'Surprise'}
        img = np.expand_dims(img,axis = 0) #makes image shape (1,48,48)
        img = img.reshape(1,48,48,1)
        result = model.predict(img)
        result = list(result[0])
        img_index = result.index(max(result))
        labl = label_dict[img_index]
        return labl

@app.route('/predict',methods = ['POST'])
def result():
 if request.method == 'POST':
    text = request.form['sentiment']
    result = predict(text)
    prediction = str(result)
    import nltk
    nltk.download('vader_lexicon')
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    analyser = SentimentIntensityAnalyzer()
    sentiment_score=analyser.polarity_scores(text)
    return render_template("predict.html",prediction=prediction), sentiment_score

@app.route('/record',methods = ['POST'])
def record():
    if request.method == 'POST':
        #Capture video from webcam
        vid_capture = cv2.VideoCapture(0)
        vid_cod = cv2.VideoWriter_fourcc(*'XVID')
        output = cv2.VideoWriter('cam_video_01.avi', vid_cod, 30, (640,480))
    
 
        # Record in chunks of 1532 samples
        chunk = 1532
 
        # 16 bits per sample
        sample_format = pyaudio.paInt16 
        chanels = 2
 
        # Record at 44400 samples per second
        smpl_rt = 44400 
        filename = "cam_video_01.wav"
 
        # Create an interface to PortAudio
        pa = pyaudio.PyAudio()
        frames = []
        while(True):
            # Capture each frame of webcam video
            ret, frame = vid_capture.read()
            stream = pa.open(format=sample_format, channels=chanels,
                        rate=smpl_rt, input=True,
                        frames_per_buffer=chunk)
 
            cv2.imshow('My cam video', frame)
            output.write(frame) 
            data = stream.read(chunk)
            frames.append(data)
            # Close and break the loop after pressing "x" key
            if cv2.waitKey(1) &0XFF == ord('x'):
                    break
        # close the already opened camera
        vid_capture.release()
        # close the already opened file
        output.release()
        # close the window and de-allocate any associated memory usage
        cv2.destroyAllWindows()
        stream.stop_stream()
        stream.close()
        pa.terminate()

        sf = wave.open(filename, 'wb')
        sf.setnchannels(chanels)
        sf.setsampwidth(pa.get_sample_size(sample_format))
        sf.setframerate(smpl_rt)
        sf.writeframes(b''.join(frames))
        sf.close()
        return render_template("record.html")

def read_file(filename):
 with open(filename, 'rb') as _file:
  while True:
   data = _file.read(1532)
   if not data:
    break
   yield data

@app.route('/transcribe',methods = ['POST'])
def transcribe():
    auth_key = "b77df5fa261544098e221a343f5127f9"
    save_location = "cam_video_01.wav"
    ## AssemblyAI endpoints and headers
    transcript_endpoint = "https://api.assemblyai.com/v2/transcript"
    upload_endpoint = 'https://api.assemblyai.com/v2/upload'
    headers_auth_only = {'authorization': auth_key}
    headers = {
    "authorization": auth_key,
    "content-type": "application/json"
    }
    ## Upload audio to AssemblyAI
    upload_response = requests.post(
    upload_endpoint,
    headers=headers_auth_only, data=read_file(save_location)
    )
    audio_url = upload_response.json()['upload_url']
    ## Start transcription job of audio file
    data = {
        'audio_url': audio_url,
        'sentiment_analysis': 'False',
        }
    transcript_response = requests.post(transcript_endpoint, json=data, headers=headers)
    transcript_id = transcript_response.json()['id']
    polling_endpoint = transcript_endpoint + "/" + transcript_id
    ## Waiting for transcription to be done
    status = 'submitted'
    while status != 'completed':
        sleep(1)
        polling_response = requests.get(polling_endpoint, headers=headers)
        transcript = polling_response.json()['text']
        status = polling_response.json()['status']
    # Display transcript
    ## Sentiment analysis response 
    sar = polling_response.json()['sentiment_analysis_results']
    ## Save to a dataframe for ease of visualization
    sen_df = pd.DataFrame(sar)
    text = sen_df['text'].to_string(index = False)
    return render_template("transcription.html",text=text)

if __name__ == "__main__":
    app.run(debug=True)