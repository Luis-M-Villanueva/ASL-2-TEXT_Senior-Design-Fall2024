from config_prep import DIR
import csv
import os
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.models import load_model		# type: ignore
from config_prep import mp_holistic, mp_drawing, mediapipe_detectionConversion, apply_landmarks, get_all_keypoints, TOTAL_NUM_FRAMES

print("Getting .keras models...")

cwd = os.getcwd() 			# current working directory. duplicate of DIR from config_prep

model_files = [f for f in os.listdir(os.path.join(cwd, "models")) if f.endswith('.keras')]
for index, file in enumerate(model_files):
	print(f"{index}: {file}")

choice = int(input("See terminal. Pick model: "))
modelName = model_files[choice]
print(f"\nUsing model: {modelName}\n")

# find csv file with same name as model
csv_file = None
for file in os.listdir(os.path.join(DIR, 'models')):
	if file.startswith(modelName[:-6]) and file.endswith('.csv'):
		csv_file = os.path.join(os.path.join(cwd, 'models'), file)		# full path to csv file

if csv_file is None:
	print("No corresponding csv file.")
else:
	# read csv file
	glosses = []
	with open(csv_file, 'r') as file:
		reader = csv.reader(file)
		for row in reader:
			glosses.append(row[0])

	print(f"Glosses in this model via csv: {glosses}\n")

glosses = np.array(glosses)

model = load_model(os.path.join(os.path.join(cwd, 'models'), modelName))

# this code collects 30 frames/keypoints and makes a prediciton. then moves on to the next one.
sequence = []
threshold = 0.4
frame_count = 0

cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.4, min_tracking_confidence=0.4) as holistic:
	predicted_gloss = ""	# initialize to blank

	while cap.isOpened():
		ret, frame = cap.read()
		image, results = mediapipe_detectionConversion(frame, holistic)

		apply_landmarks(image, results)

		keypoints = get_all_keypoints(results)
		sequence.append(keypoints)				# add to sequence
		sequence = sequence[-TOTAL_NUM_FRAMES:]				# only keep last 30 frames

		#frame_count = (frame_count + 1) % TOTAL_NUM_FRAMES
		#cv2.putText(image, f"{frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

		# if we're at 30th frame, make prediction
		if len(sequence) == TOTAL_NUM_FRAMES:

			# add new axis to sequence. shape = 1, 30, num_keypoints
			# 	get 0th prediction
			# LSTM model takes (batch_size, timesteps, features)
			# batch size is 1? i don't fully understand why we need only the 0th.
				# something to do with batch size = 1
			pred_arry = model.predict(np.expand_dims(sequence, axis=0))[0]

			# console output
			# for num, prob in enumerate(pred_arry):
			# 	print(f'{num}: {prob:.2f}')

			# display on feed
			if pred_arry[np.argmax(pred_arry)] > threshold:
				predicted_gloss = glosses[np.argmax(pred_arry)]
				# moving this to outside the if statement
				# cv2.putText(image, predicted_gloss, (10, image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

			# reset sequence
			sequence.clear()
		
		# moved down here
		cv2.putText(image, predicted_gloss, (10, image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

		# output to feed
		cv2.imshow('RUNNING PREDICTIONS', image)

		if cv2.waitKey(10) == 27:  # 27 is the esc key
			print("exit safely")
			break

	cap.release()
	cv2.destroyAllWindows()