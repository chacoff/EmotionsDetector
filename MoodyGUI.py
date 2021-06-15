import PySimpleGUI as sg
from tensorflow.keras.models import model_from_json
import numpy as np
import cv2
from numpy import random, moveaxis
import os
import pyautogui
from win32api import GetSystemMetrics
from Emotions_UI_final import load_model, predict_emotion
sg.theme('Reddit')


def standbyVideo(f3, f4):
    img = np.full((f4, f3), 240)
    imgbytes = cv2.imencode('.png', img)[1].tobytes()  # this is faster, shorter and needs less includes
    window['image'].update(data=imgbytes)


if __name__ == '__main__':

    # ---------- LAYOUT ---------- #
    layout = [[sg.Image(filename='', key='image')],

        # [sg.Text('Face Detector', size=(14, 1), auto_size_text=False, justification='left'),
        #  sg.InputText('', size=(14, 1), enable_events=True, key="-FACE_MOD-"),
        #  sg.FolderBrowse(size=(10, 1), font='Helvetica 8')],

        # [sg.Text('Emotion\'s model', size=(14, 1), auto_size_text=False, justification='left'),
        #  sg.InputText('', size=(14, 1), enable_events=True, key="-EMO_MOD-"),
        # sg.FolderBrowse(size=(10, 1), font='Helvetica 8')],

        [sg.Button('Process', size=(10, 1), font='Helvetica 8'),
         sg.Button('Stop', size=(10, 1), font='Helvetica 8'),
         sg.Button('Exit', size=(10, 1), font='Helvetica 8'), ]]

    # create the window and show it without the plot
    window = sg.Window('MoodyNet', layout, location=(800, 400))
    # ---------- LAYOUT ---------- #

    # ---------- PARAMETERS SETTINGS ---------- #
    CAM = False  # if True, the webcam/video will be capture if False, the primary screen will be capture
    VID = 0  # 0 for webcam or video address, example 'vidk.mp4'
    REC = False  # to record the webcam or the screen, must be True for single video processing
    REC_file = 'jup06_4classes.avi'
    FACTOR = 2.5  # factor to reduce the size of the screen in within the UI
    SingleChannel = True  # currently only supporting 1 channel

    # face detector
    protoPath = os.path.join('models', 'deploy.prototxt')  # face detector based on a res net
    modelPath = os.path.join('models', 'res10_300x300_ssd_iter_140000.caffemodel')  # face detector
    detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
    threshold = 0.45  # to filter out week face detections
    px_filter = 10  # size filter to ensure a sufficiently large face

    # emotions classifier
    path = "models/jup09/"
    model = load_model(path)
    # emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}
    emotion_dict = {0: "upset", 1: "happy", 2: "neutral"}

    # color_cycle = [[random.randint(10, 255) for _ in range(3)] for _ in range(len(emotion_dict))]
    color_cycle = [(30, 30, 250), (0, 210, 0), (0, 220, 220)]
    # ---------- PARAMETERS SETTINGS ---------- #

    # -------- Event LOOP Read and display frames, operate the GUI -------- #
    if CAM:
        webcam = cv2.VideoCapture(VID)
        f3 = int(webcam.get(3))  # width
        f4 = int(webcam.get(4))  # height
        recording = False
    else:
        f3 = int(GetSystemMetrics(0) / FACTOR)  # width
        f4 = int(GetSystemMetrics(1) / FACTOR)  # height
        recording = False

    if REC is True:
        out = cv2.VideoWriter(REC_file, cv2.VideoWriter_fourcc(*'XVID'), 12, (f3, f4))

    while True:
        event, values = window.read(timeout=20)  # pysimpleGUI
        standbyVideo(f3, f4)

        if event == 'Exit' or event == sg.WIN_CLOSED:
            break

        elif event == 'Process':
            recording = True

        elif event == 'Stop':
            recording = False
            standbyVideo(f3, f4)

        if recording:

            if CAM:
                ret, frame = webcam.read()
            else:
                img = pyautogui.screenshot()  # PIL
                frame = np.array(img)
                frame = cv2.resize(frame, (f3, f4))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                ret = True

            if ret is False:
                break

            frame = cv2.normalize(frame, None, 10, 245, cv2.NORM_MINMAX)  # normalize
            (h_frame, w_frame) = frame.shape[:2]

            # construct a blob from the image
            imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 187.0, 123.),
                                              swapRB=False, crop=False)
            detector.setInput(imageBlob)  # OpenCV's face detector to localize faces in the input image
            detections = detector.forward()

            for i in range(0, detections.shape[2]):  # loop over all the detections

                confidence = detections[0, 0, i, 2]  # extract the confidence associated with the prediction

                if confidence > threshold:  # filter out weak detections
                    box = detections[0, 0, i, 3:7] * np.array([w_frame, h_frame, w_frame, h_frame])
                    (startX, startY, endX, endY) = box.astype("int")  # x, y coordinates of the bounding box for the face
                    (startX, startY, endX, endY) = (5 + startX, 5 + startY, 5 + endX, 5 + endY)  # offset for classification
                    detect_width = endX - startX
                    detect_height = endY - startY

                    face = frame[startY:endY, startX:endX]  # extract the face ROI
                    face = cv2.resize(face, (300, 300), interpolation=cv2.INTER_AREA)
                    (fH, fW) = face.shape[:2]
                    # cv2.imwrite('text1.jpg', face)

                    if fW < px_filter or fH < px_filter:  # ensure the face width and height are sufficiently large
                        continue

                    if SingleChannel:
                        gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
                        # cv2.imwrite('text2.jpg', gray)
                    else:
                        gray = face  # because of 3 channel model

                    emotion_id, proba = predict_emotion(gray, startX, startY,
                                                        detect_width, detect_height, SC=True, mod=model)
                    emotion = emotion_dict[emotion_id]

                    text1 = '{}: {:.2f}%'.format(emotion, proba)  # draw the face's bounding box along with the probability
                    text2 = 'face{} with {:.2f}%'.format(i + 1, confidence)
                    # text2 = "face #" + str(i+1)
                    y_shift1 = startY - 10 if startY - 10 > 10 else startY + 10
                    y_shift2 = startY + detect_height + 15
                    cv2.rectangle(frame, (startX, startY), (endX, endY), color_cycle[emotion_id], 2)
                    cv2.putText(frame, text1, (startX, y_shift1), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_cycle[emotion_id], 2)
                    cv2.putText(frame, text2, (startX, y_shift2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_cycle[emotion_id], 2)

            if REC is True:
                out.write(frame)

            imgbytes = cv2.imencode('.png', frame)[1].tobytes()  # ditto
            window['image'].update(data=imgbytes)
