import PySimpleGUI as sg
from tensorflow.keras.models import model_from_json
import numpy as np
import cv2
from numpy import random, moveaxis
import os, sys
import pyautogui
from win32api import GetSystemMetrics
import xml.etree.ElementTree as ET
from Emotions_UI_final import load_model, predict_emotion
from MoodyGraphBase64 import *


def standbyVideo(f3, f4):
    img = np.full((f4, f3), 240)
    imgbytes = cv2.imencode('.png', img)[1].tobytes()  # this is faster, shorter and needs less includes
    window['image'].update(data=imgbytes)


def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)


def list_parameters(l):
    list_param = []
    for i, data in enumerate(l):
        list_param.append(data)
    return list_param


def UIlayout_A():
    sg.theme('Reddit')  # THEME currently not implemented

    menu_def = [['&Help', ['&About', '&Exit']],
                ]

    layout = [
        [sg.MenubarCustom(menu_def, pad=(0, 0), k='-CUST MENUBAR-')],
        [sg.Image(filename='', key='image')],
        [sg.Button('', image_data=image_check,
                   button_color=('white', sg.theme_background_color()),
                   border_width=0, key='Process'),
         sg.Button('', image_data=image_remove,
                   button_color=('white', sg.theme_background_color()),
                   border_width=0, key='Stop'),
         sg.Button('', image_data=image_settings,
                   button_color=('white', sg.theme_background_color()),
                   border_width=0, key='Parameters'),
         ]
    ]

    # create the window and show it without the plot
    window = sg.Window('Moody.AI', layout,
                       location=(800, 400),
                       alpha_channel=HALO,
                       use_custom_titlebar=True,
                       titlebar_icon=mainWindowIcon_32,
                       keep_on_top=True,
                       resizable=True,
                       )
    return window


def UIlayout_B(PARAMS):

    lyt = [
        [sg.Text('Detection:', font=("Helvetica", 12))],
        [sg.Text('Detector Threshold', size=(15, 1)), sg.Combo(values=[round(x/100, 3) for x in range(10, 105, 5)], default_value=PARAMS[10], key='thrs', size=(9, 1))],
        [sg.Text('Models Folder', size=(15, 1)), sg.InputText(PARAMS[7], key='mod', size=(11, 1))],
        [sg.Text('Model Name', size=(15, 1)), sg.InputText(PARAMS[11], key='emoc', size=(11, 1))],


        [sg.Text('Options:', font=("Helvetica", 12))],
        [sg.Text('Webcam', size=(15, 1)), sg.Combo(values=['YES', 'NO'], default_value=PARAMS[2], key='webcam', size=(9, 1))],
        [sg.Text('Webcam Source', size=(15, 1)), sg.Combo(values=[y for y in range(0, 6, 1)], default_value=PARAMS[3], key='websource', size=(9, 1))],
        [sg.Text('Recording', size=(15, 1)), sg.Combo(values=['YES', 'NO'], default_value=PARAMS[4], key='rec', size=(9, 1))],
        [sg.Text('Recording file', size=(15, 1)), sg.InputText(PARAMS[5], key='avifile', size=(11, 1))],
        [sg.Text('FPS', size=(15, 1)), sg.Combo(values=[y for y in range(8, 50, 1)], default_value=PARAMS[12], key='fps_rec', size=(9, 1))],

        [sg.Text('Misc:', font=("Helvetica", 12))],
        [sg.Text('UI Transparency', size=(15, 1)), sg.Combo(values=[round(x/100, 3) for x in range(11, 105, 5)], default_value=PARAMS[1], key='halo', size=(9, 1))],
        [sg.Button('Save'), sg.Button('Exit')]
           ]

    win = sg.Window('Parameters', lyt,
                    no_titlebar=False,
                    grab_anywhere=False,
                    keep_on_top=True,
                    icon=image_settings)

    while True:
        eve, val = win.read()
        if eve in (sg.WINDOW_CLOSED, 'Exit'):
            break
        elif eve == 'Save':
            continue

    win.close()


def main_XML():
    tree = ET.parse(resource_path('params.xml'))
    root = tree.getroot()
    # fullXML = ET.tostring(root, encoding='utf8').decode('utf8')
    Xlist = [t.text for t in root.iter('item')]  # list of the parameters
    PARAMS = list_parameters(Xlist)

    THEME = root[0][0].text
    HALO = root[0][1].text
    CAM = True if root[0][2].text == 'YES' else False  # if True, the webcam/video will be capture if False, the primary screen will be capture
    VID = root[0][3].text  # 0 for webcam or video address, example 'vidk.mp4'
    REC = True if root[0][4].text == 'YES' else False  # to record the webcam or the screen
    REC_file = root[0][5].text
    fps_xml = int(root[0][12].text)
    FACTOR = float(root[0][6].text)  # factor to reduce the size of the screen in within the UI

    # face detector
    protoPath = os.path.join(root[0][7].text, root[0][8].text)  # face detector based on a res net
    modelPath = os.path.join(root[0][7].text, root[0][9].text)  # face detector
    detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
    threshold = float(root[0][10].text)  # to filter out weak face detections

    # emotions classifier
    emo_model = os.path.join(root[0][7].text, root[0][11].text)
    model = load_model(emo_model)  # emotions classifier model

    return PARAMS, HALO, CAM, VID, REC, REC_file, fps_xml, FACTOR, detector, threshold, model


if __name__ == '__main__':
    sg.popup_quick_message('Loading Moody.AI', background_color='white', text_color='red')
    PARAMS, HALO, CAM, VID, REC, REC_file, fps_xml, FACTOR, detector, threshold, model = main_XML()
    window = UIlayout_A()

    # emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}
    emotion_dict = {0: "upset",
                    1: "happy",
                    2: "neutral"}

    # color_cycle = [[random.randint(10, 255) for _ in range(3)] for _ in range(len(emotion_dict))]
    color_cycle = [(30, 30, 250),
                   (0, 210, 0),
                   (0, 220, 220)]

    # -------- Event LOOP Read and display frames, operate the GUI -------- #
    if CAM:
        VID = int(VID) if VID == '0' else VID
        webcam = cv2.VideoCapture(VID)
        f3 = int(webcam.get(3))  # width
        f4 = int(webcam.get(4))  # height
        recording = False
    else:
        f3 = int(GetSystemMetrics(0) / float(FACTOR))  # width
        f4 = int(GetSystemMetrics(1) / float(FACTOR))  # height
        recording = False

    if REC is True:
        out = cv2.VideoWriter(REC_file, cv2.VideoWriter_fourcc(*'XVID'), fps_xml, (f3, f4))

    # ----- Main Loop -------
    while True:
        event, values = window.read(timeout=20)  # pysimpleGUI
        standbyVideo(f3, f4)

        if event == 'Exit' or event == sg.WIN_CLOSED:
            break

        elif event == 'About':
            # window.disappear()
            sg.popup(sg.get_versions(),
                     grab_anywhere=True,
                     keep_on_top=True,
                     icon=mainWindowIcon_32)
            # window.reappear()

        elif event == 'Parameters':
            recording = False
            standbyVideo(f3, f4)
            UIlayout_B(PARAMS)
            #sg.popup('Parameters',
            #         *PARAMS,
            #         no_titlebar = False,
            #         grab_anywhere=True,
            #         keep_on_top=True,
            #         icon=mainWindowIcon_32)

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
                    of = 8
                    (startX, startY, endX, endY) = (startX - of, startY - of, endX + of, endY + of)  # offset for classification
                    detect_width = endX - startX
                    detect_height = endY - startY

                    face = frame[startY:endY, startX:endX]  # extract the face ROI
                    try:
                        face = cv2.resize(face, (300, 300), interpolation=cv2.INTER_AREA)
                    except:
                        continue
                    (fH, fW) = face.shape[:2]
                    # cv2.imwrite('text1.jpg', face)

                    if fW < 10 or fH < 10:  # ensure the face width and height are sufficiently large
                        continue

                    SingleChannel = True  # currently only supporting 1 channel
                    if SingleChannel:
                        gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
                        # cv2.imwrite('text2.jpg', gray)
                    else:
                        gray = face  # because of 3 channel model

                    emotion_id, proba = predict_emotion(gray, startX, startY,
                                                        detect_width, detect_height, SC=True, mod=model)

                    # Post processing
                    if emotion_id == 0:
                        if proba <= 85.00:
                            emotion_id = 2

                    emotion = emotion_dict[emotion_id]

                    text1 = '{}: {:.1f}%'.format(emotion, proba)  # draw the face's bounding box along with the probability
                    text2 = 'face{} with {:.1f}%'.format(i + 1, 100*confidence)
                    # text2 = "face #" + str(i+1)
                    y_shift1 = startY - 10 if startY - 10 > 10 else startY + 10
                    y_shift2 = startY + detect_height + 15
                    cv2.rectangle(frame, (startX, startY), (endX, endY), color_cycle[emotion_id], 2)
                    cv2.putText(frame, text1, (startX, y_shift1), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_cycle[emotion_id], 1)
                    cv2.putText(frame, text2, (startX, y_shift2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_cycle[emotion_id], 1)

            if REC is True:
                out.write(frame)

            imgbytes = cv2.imencode('.png', frame)[1].tobytes()  # ditto
            window['image'].update(data=imgbytes)
