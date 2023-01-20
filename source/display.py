import os, os.path

import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from PIL import Image,ImageTk
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

model.load_weights('test_model.h5')

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
   


open_camera_face_recognition:bool = False

def choose_single_image_option():
    print("jehehe")
    global main, root
    main.destroy()
    main=Frame(root)
    main.pack()

    root.filename = filedialog.askopenfilename(initialdir="/", title="Select A File", filetypes=(("png files", "*.png"),("all files", "*.*")))

    if root.filename is not None:	        
        image = cv2.imread(root.filename)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        color = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
       
        face_cascade = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml'))
        faces = face_cascade.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(color, (x, y-50), (x+w, y+h+10), (255, 255, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))

        img= Image.fromarray(gray)
        color_img =  Image.fromarray(color)
        test = ImageTk.PhotoImage(color_img)

        text_label = Label(main, text=f"{emotion_dict[maxindex]}", pady=50,font=("Arial Bold", 70))
        text_label.pack(side=TOP)
        label_bottom= Label(image=test)
        label_bottom.image = test
        label_bottom.pack(side=BOTTOM, padx=10, pady=10)




def choose_camera_option():
    main.destroy()
    global open_camera_face_recognition
    open_camera_face_recognition = True
    
    root.destroy()


root = Tk()
root.title("Emotion Recognition Application")
root.geometry("1000x900")
main=Frame(root)
main.pack()

one_image_option_button = Button(main, text="Single Image Emotion Recognition", padx= 100, pady=100, command=choose_single_image_option)
camera_option_button = Button(main, text="Camera Emotion Recognition", padx=100, pady=100, command=choose_camera_option)


my_label = Label(main, text="Emotion Recognition Application", pady=300,font=("Arial Bold", 40))
my_label.pack(side=TOP)
one_image_option_button.pack(anchor=S, side=LEFT)
camera_option_button.pack(anchor=S, side=RIGHT)

root.mainloop()


if open_camera_face_recognition:
    cap = cv2.VideoCapture(0)
    while True:

        ret, frame = cap.read()

        face_cascade = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml'))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 255, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Video', frame)
        if cv2.waitKey(1) == ord('q'):
            break
        
    cv2.destroyAllWindows()


