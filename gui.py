import tkinter as tk
from tkinter import filedialog
from tkinter import *
import time
import cv2
import numpy as np
import librosa
from tensorflow.keras.models import model_from_json
from PIL import Image, ImageTk
import pygame
from PIL import Image, ImageDraw, ImageTk
import pyaudio
import wave
import tempfile
import os
import threading

# Define emotion labels
EMOTIONS_LIST = ["Angry","Disgust","Fear","Happy","Neutral","Sad","Surprise"]


# LOADING AUDIO THROUGH MICROPHONE INTO A PATH AND THEN PREDICTING THE EMOTION USING PYAUDIO----------------------------
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

def start_recording():
    record_button.config(state=tk.DISABLED)
    stop_button.config(state=tk.NORMAL)
    global audio_data
    audio_data = []
    threading.Thread(target=record_audio).start()

def stop_recording():
    global recording
    recording = False
    record_button.config(state=tk.NORMAL)
    stop_button.config(state=tk.DISABLED)

def record_audio():
    global recording
    recording = True

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    while recording:
        data = stream.read(CHUNK)
        audio_data.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    process_recorded_audio()

def process_recorded_audio():
    combined_audio = b''.join(audio_data)
    temp_path = os.path.join(tempfile.gettempdir(), "recorded_audio.wav")

    with wave.open(temp_path, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(RATE)
        wf.writeframes(combined_audio)

    print("Recording complete. Audio saved to:", temp_path)
    predict_through_microphone(temp_path)
 

def predict_through_microphone(file_path):
    audio_features = extract_audio_features(file_path=file_path)
    audio_emotion = voice_model.predict(audio_features)
    pred = EMOTIONS_LIST[np.argmax(audio_emotion)]
    # pred = np.argmax(audio_emotion)
    print("Predicted Emotion is" + pred)
    label1.configure(foreground="#011638",text = pred)


recording = False
audio_data = []
#--------------------------------------------------------------------------------------------------------


top = tk.Tk()
top.geometry('1000x800')
top.title('Emotion Detector')
top.configure(background='#CDCDCD')

label1 = Label(top, background='#CDCDCD', font=('arial',15,'bold'))
sign_image = Label(top)

def SpeechExpressionModel(json_file, weights_file):
    with open(json_file,"r") as file:
        loaded_model_json = file.read()
        model = model_from_json(loaded_model_json)

    model.load_weights(weights_file)
    model.compile(optimizer ='adam', loss='sparse_categorical_crossentropy', metrics = ['accuracy'])

    return model


#Extracting the audio features using the path of the audio
def extract_audio_features(file_path, num_mfcc=13, max_frames=504):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=num_mfcc)
    
    if mfccs.shape[1] < max_frames:
        pad_width = max_frames - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)))
    else:
        mfccs = mfccs[:, :max_frames]
    
    # Reshape audio features to match expected input shape
    mfccs = mfccs.reshape((1, num_mfcc, max_frames, 1))
    
    return mfccs


# Load pre-trained models for facial expression recognition and voice tone analysis
voice_model = SpeechExpressionModel("Speechmodel_a1.json","Speechmodel_weights1.h5")


#Function to detect the emotion of the audio uploaded
def Detect(file_path):
    global Label_packed
    try:
        audio_features = extract_audio_features(file_path)
        # audio_features = np.array(audio_features)
        print(len(audio_features))

        audio_emotion = voice_model.predict(audio_features)
        print(audio_emotion)
        pred = EMOTIONS_LIST[np.argmax(audio_emotion)]
        print("Predicted Emotion is" + pred)
        label1.configure(foreground="#011638",text = pred)
    except:
        label1.configure(foreground="#011638",text = "Unable to detect")


#Function to show "detect emotion" button only when the audio is uploaded
def show_Detect_button(file_path):
    detect_b = Button(top,text="Detect Emotion", command= lambda: Detect(file_path),padx=10,pady=5)
    detect_b.configure(background="#364156",foreground='white',font=('arial',10,'bold'))
    detect_b.place(relx =0.79,rely=0.46)


# DESIGNING THE PLAY BUTTON--------------------------------------------------
# Create a triangle image using PIL
triangle_size = 50
triangle_img = Image.new('RGBA', (triangle_size, triangle_size))
draw = ImageDraw.Draw(triangle_img)
triangle_color = '#AA0000'  # red color
draw.polygon([(0, triangle_size), (triangle_size // 2, 0), (triangle_size, triangle_size)], fill=triangle_color)

# Rotate the triangle image 90 degrees to the right
triangle_img = triangle_img.rotate(-90, expand=True)

# Convert the triangle image to PhotoImage
triangle_photo = ImageTk.PhotoImage(triangle_img)
#-----------------------------------------------------------------------------


#Function to show play button only when the audio is uploaded...........
def show_play_btn(file_path):
    play_button = Button(top)
    play_button.configure(background="#CDCDCD")
    # play_button.pack(side='bottom',pady=50)
    play_button.place(x=475, y=360)

    # Configure the "Play" button with the triangle image
    play_button.config(image=triangle_photo, command= lambda: play_audio(file_path),padx=10,pady=5, compound=tk.CENTER)
    # play_button.pack()

    # Create and configure a label
    info_label = tk.Label(top, text="Play the Audio file", fg="black", background=None)
    info_label.place(x=452, y=420)



#Function to upload the audio.............
def upload_audio():
    # global file_path
    try:
        file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.mp3 *.wav *.ogg")])
        if file_path:
            filename = file_path.split("/")[-1]
            label1.config(text="Audio File: " + filename)

        show_Detect_button(file_path)
        show_play_btn(file_path)
    except:
        pass


#Function to play the audio............
def play_audio(file_path):
    if file_path:
        pygame.mixer.init()
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()


#BUTTONS-----------------------------------------------------------------------------------------
upload1 = Button(top, text="Upload Audio", command=upload_audio, padx=10, pady=5)
upload1.configure(background="#364156",foreground='white',font=('arial',20,'bold'))
upload1.pack(side='bottom',pady=50)
upload1.place(x=25, y=100)

record_button = tk.Button(top, text="Record", command=start_recording)
record_button.pack(pady=10)
record_button.configure(background="#364156",foreground='white',font=('arial',20,'bold'))
record_button.pack(side='bottom',pady=50)
record_button.place(x=25, y=300)

stop_button = tk.Button(top, text="Stop", command=stop_recording, state=tk.DISABLED)
stop_button.pack(pady=5)
stop_button.configure(background="#364156",foreground='white',font=('arial',20,'bold'))
stop_button.pack(side='bottom',pady=50)
stop_button.place(x=25, y=400)

#----------------------------------------------------------------------------------------------------


sign_image.pack(side='bottom', expand='True')
label1.pack(side='bottom', expand='True')
heading = Label(top,text='Emotion Detector',pady=20,font=('arial',25,'bold'))
heading.configure(background='#CDCDCD',foreground="#364156")
heading.pack()

top.mainloop()