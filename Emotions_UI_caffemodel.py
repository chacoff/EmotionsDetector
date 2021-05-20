from tensorflow.keras.models import model_from_json
import numpy as np
import cv2
from numpy import random
import os


def load_model(path):
	json_file = open(path + 'model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	model = model_from_json(loaded_model_json)
	model.load_weights(path + "model.h5")
	print("Loaded model from disk")
	return model


def predict_emotion(gray, x, y, w, h):
	face = np.expand_dims(np.expand_dims(np.resize(gray[y:y+w, x:x+h]/255.0, (48, 48)), -1), 0)
	prediction = model.predict([face])

	return int(np.argmax(prediction)), round(max(prediction[0])*100, 2)


# face detector
protoPath = os.path.join('models', 'deploy.prototxt')  # face detector based on a res net
modelPath = os.path.join('models', 'res10_300x300_ssd_iter_140000.caffemodel')  # face detector
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# emotions classifier
path = "models/exp2/"
model = load_model(path)
emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}

color_cycle = [[random.randint(0, 255) for _ in range(3)] for _ in range(10)]
webcam = cv2.VideoCapture(0)
while True:
	ret, frame = webcam.read()
	frame = cv2.normalize(frame, None, 10, 230, cv2.NORM_MINMAX)  # normalize
	(h_frame, w_frame) = frame.shape[:2]

	# construct a blob from the image
	imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 187.0, 123.),
									  swapRB=False, crop=False)
	detector.setInput(imageBlob)  # OpenCV's deep learning-based face detector to localize faces in the input image
	detections = detector.forward()

	for i in range(0, detections.shape[2]):  # loop over all the detections

		confidence = detections[0, 0, i, 2]  # extract the confidence associated with the prediction

		if confidence > 0.5:  # filter out weak detections
			box = detections[0, 0, i, 3:7] * np.array([w_frame, h_frame, w_frame, h_frame])
			(startX, startY, endX, endY) = box.astype("int")  # x, y coordinates of the bounding box for the face
			detect_width = endX - startX
			detect_height = endY - startY

			face = frame[startY:endY, startX:endX]  # extract the face ROI
			(fH, fW) = face.shape[:2]

			if fW < 20 or fH < 20:  # ensure the face width and height are sufficiently large
				continue

			gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
			emotion_id, proba = predict_emotion(gray, startX, startY, detect_width, detect_height)
			emotion = emotion_dict[emotion_id]

			text1 = "{}: {:.2f}%".format(emotion, proba)  # draw the face's bounding box along with the probability
			text2 = "face #" + str(i+1)
			y_shift1 = startY - 10 if startY - 10 > 10 else startY + 10
			y_shift2 = startY + detect_height + 15
			cv2.rectangle(frame, (startX, startY), (endX, endY), color_cycle[i], 2)
			cv2.putText(frame, text1, (startX, y_shift1), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_cycle[i], 2)
			cv2.putText(frame, text2, (startX, y_shift2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_cycle[i], 2)

	cv2.imshow('Emotion Recognition - Press q to exit.', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

webcam.release()
cv2.destroyAllWindows()