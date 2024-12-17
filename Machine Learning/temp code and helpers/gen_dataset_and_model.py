# %%
import mediapipe as mp
import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential          # type: ignore
from tensorflow.keras.layers import Input, LSTM, Dense         # type: ignore
# from tensorflow.keras.callbacks import TensorBoard      # type: ignore
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical   	# type: ignore
from tensorflow.keras.models import load_model			# type: ignore

# %%
# DECLARE MEDIAPIPE FUNCS
mp_holistic = mp.solutions.holistic 	# using holistic for pose face and hands
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detectionConversion(image, model):
	"""
	Performs mediapipe detection on a frame of video. Does color conversion and sets image to be non-writeable for performance.

	Args:
		image (numpy array): The frame of video to be processed
		model (mediapipe.solutions.holistic.Holistic): The mediapipe model object

	Returns:
		tuple: A tuple containing the processed frame and the mediapipe results
	"""
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image.flags.writeable = False
	results = model.process(image)				# pipinig into mediapipe
	image.flags.writeable = True				# saves memory? idk their github did it
	image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
	return image, results

def apply_landmarks(image, results):
	"""
	Applies landmarks to an image using mediapipe. Draws pose, face, and hand landmarks onto the image.

	Args:
		image (numpy array): The image to be processed
		results (mediapipe.solutions.holistic.Holistic): The mediapipe results object

	Returns:
		numpy array: The processed image
	"""
	mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
							mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2),
							mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2)
							)
	mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
							mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
							mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2)
							)
	mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
							mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
							mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
							)
	mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
							mp_drawing.DrawingSpec(color=(245,117,66), thickness=1, circle_radius=1), 
							mp_drawing.DrawingSpec(color=(245,66,230), thickness=1, circle_radius=1)
							)

def get_all_keypoints(results):
	
	"""
	Takes mediapipe results and returns a single numpy array of all keypoints extracted from the image.
	
	Keypoints are ordered as follows:
		1. Pose keypoints
		2. Face keypoints
		3. Left hand keypoints
		4. Right hand keypoints
	
	Each keypoint is stored as a numpy array with columns [x, y, z, visibility] for pose and [x, y, z] for face and hand.
	
	Args:
		results (mediapipe.solutions.holistic.Holistic): The mediapipe results object
	
	Returns:
		numpy array: The concatenated numpy array of keypoints
	"""
	if results.pose_landmarks:
		pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten()
	else:
		pose = np.zeros(33 * 4)

	# Extract face landmarks
	if results.face_landmarks:
		face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten()
	else:
		face = np.zeros(468 * 3)

	# TODO check if that 1404 matches up with face tesselation/contours

	# Extract left hand landmarks
	if results.left_hand_landmarks:
		lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten()
	else:
		lh = np.zeros(21 * 3)

	# Extract right hand landmarks
	if results.right_hand_landmarks:
		rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()
	else:
		rh = np.zeros(21 * 3)
	
	return np.concatenate([pose, face, lh, rh])


# %% KEYPOINT DRAWING TEST
cap = cv2.VideoCapture(0)
# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.25, min_tracking_confidence=0.25) as holistic:
	while cap.isOpened():
		ret, frame = cap.read()
		image, results = mediapipe_detectionConversion(frame, holistic)
		
		apply_landmarks(image, results)

		cv2.imshow('Testing if mediapipe works. Esc to exit', image)
		if cv2.waitKey(10) == 27:  # 27 is the esc key
			break
	cap.release()
	cv2.destroyAllWindows()


# %%
# MAKE DATASET FOLDER (SINGULAR)
file_path = os.path.join('D:\\senior_design', input("Folder name for dataset: "))
if os.path.exists(file_path):
	print("Folder exists, THIS WILL OVERWRITE...")
else:
	os.makedirs(file_path)

totalNum_frames = int(input("Total number of frames for each sequence: "))
num_sequences = int(input("Sequences for each gloss: "))

# %%
# DATASET CAPTURE
glosses = []
def capture_gloss_data(base_path, totalNum_frames, glosses):
	cap = cv2.VideoCapture(0)
	mp_holistic = mp.solutions.holistic
	with mp_holistic.Holistic(min_detection_confidence=0.25, min_tracking_confidence=0.25) as holistic:
		while True:
			# get gloss, remove whitespace
			gloss = input("Gloss name (press ESC to exit): ").strip()
			if gloss.lower() == "esc":
				print("Exiting...")
				break
			
			glosses.append(gloss)  # add gloss to the list
			
			# make folder. if exists, overwrite. if not, make it
			gloss_path = os.path.join(base_path, gloss)
			if os.path.exists(gloss_path):
				print(f"'{gloss}' folder exists, overwriting...")
			else:
				os.makedirs(gloss_path)

			for sequence in range(0, num_sequences):
				# make folder. if exists, overwrite. if not, make it
				sequence_path = os.path.join(gloss_path, str(sequence))
				os.makedirs(sequence_path, exist_ok=True)
				
				for frame_num in range(totalNum_frames):
					ret, frame = cap.read()
					if not ret:
						print("camera error. exiting...")
						cap.release()
						cv2.destroyAllWindows()
						return
					
					image, results = mediapipe_detectionConversion(frame, holistic)
					apply_landmarks(image, results)
					
					# Show instructions
					if frame_num == 0:
						cv2.putText(image, f'Sign {gloss}!', (15, image.shape[0] - 10),
									cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
						cv2.imshow('Perform action after message', image)
						cv2.waitKey(1250)
					else:
						cv2.putText(image, f'Collecting frames for {gloss} Video {sequence}/30', (15, 12),
									cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
						cv2.imshow('Perform action after message', image)
					
					# Capture keypoints
					keypoints = get_all_keypoints(results)
					npy_path = os.path.join(sequence_path, f'{frame_num}.npy')
					np.save(npy_path, keypoints)
					
					if cv2.waitKey(10) == 27:  # ESC to exit
						print("Exit early...")
						cap.release()
						cv2.destroyAllWindows()
						return
				
			# End of sequence, prompt to continue
			print(f"Press SPACE to continue to the next gloss, or ESC to exit.")
			while True:
				key = cv2.waitKey(0)
				if key == 32:  # Spacebar to continue
					break
				elif key == 27:  # ESC to exit
					print("Exiting...")
					cap.release()
					cv2.destroyAllWindows()
					return
	
	cap.release()
	cv2.destroyAllWindows()

capture_gloss_data(file_path, totalNum_frames, glosses)
# glosses must be np array bc can't use glosses.shape[0] w/o it
glosses = np.array(glosses)								

# %% CLOSE DATASET COLLECTION EARLY
cap.release()
cv2.destroyAllWindows()


# %%
# generate label map
label_map = {label:num for num, label in enumerate(glosses)}

# %%
# possible issue here with out of bounds picking 30.
categories, labels = [], []
for gloss in glosses:
	for sequence in np.array(os.listdir(os.path.join(file_path, gloss))).astype(int):
		window = []
		for frame_num in range(totalNum_frames):
			res = np.load(os.path.join(file_path, gloss, str(sequence), "{}.npy".format(frame_num)))
			window.append(res)
		categories.append(window)
		labels.append(label_map[gloss])

# %%
# prep data
x = np.array(categories)

y = to_categorical(labels).astype(int) # one hot encoding
										# one between _, _, _

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

#log_dir = os.path.join('Logs')
#tb_callback = TensorBoard(log_dir=log_dir)


# %%
# create model

# this model made invite.h5. it was more accurate than anything and handled 12 inputs. but it can't tell hand signs well.

model = Sequential([
    Input(shape=(30, 1662)),  # Specify the input shape here
    LSTM(64, return_sequences=True, activation='tanh', dropout=0.2),
    LSTM(128, return_sequences=True, activation='tanh', dropout=0.2),
    LSTM(64, return_sequences=False, activation='tanh', dropout=0.2),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(glosses.shape[0], activation='softmax')  # Ensure glosses.shape[0] is the correct output size
])

# Optimizer with gradient clipping
optimizer = tf.keras.optimizers.Adam(clipnorm=1.0)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])


# this one made invite 2.h5.
from tensorflow.keras.layers import Bidirectional #	type: ignore

# model = Sequential()
# model.add(Bidirectional(LSTM(128, return_sequences=True, input_shape=(30, 1662), activation='tanh', dropout=0.2)))
# model.add(Bidirectional(LSTM(256, return_sequences=True, activation='tanh', dropout=0.2)))
# model.add(Bidirectional(LSTM(128, return_sequences=False, activation='tanh', dropout=0.2)))
# model.add(Dense(256, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(glosses.shape[0], activation='softmax'))  # Adjust for gloss count

# optimizer = tf.keras.optimizers.Adam(clipnorm=1.0)  # Gradient clipping
# model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# let's not talk about invite3.h5

# %%
# train model
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='categorical_accuracy', patience=15, restore_best_weights=True)
model.fit(x_train, y_train, epochs=200, callbacks=[early_stopping])

model.summary()

model_name = input("Name model: ")

model.save('{}.h5'.format(model_name))

# %%
# JANK SECTION
# TEST TE MODEL - JANKY AF TODO: make new file for this

print("Run code for libraries, functions, before this code.")

modelName = input("Enter model name under test: ")

glosses = input("Enter the glosses in the model separated by commas: ").split(',')
glosses = np.array([action.strip() for action in glosses])

model = load_model('{}.h5'.format(modelName))


# %%
# i am not all the way there but let's do this

# this code collects 30 frames/keypoints and makes a prediciton. then moves on to the next one.
sequence = []
threshold = 0.6

cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.35, min_tracking_confidence=0.35) as holistic:
	while cap.isOpened():
		ret, frame = cap.read()
		image, results = mediapipe_detectionConversion(frame, holistic)

		apply_landmarks(image, results)

		keypoints = get_all_keypoints(results)
		sequence.append(keypoints)				# add to sequence
		sequence = sequence[-30:]				# only keep last 30 frames

		# if we're at 30th frame, make prediction
		if len(sequence) == 30:
			# add new axis to sequence. shape = 1, 30, num_keypoints
			# 	get 0th prediction
			# LSTM model takes (batch_size, timesteps, features)
			# batch size is 1? i don't fully understand why we need only the 0th.
				# something to do with batch size = 1
			pred_arry = model.predict(np.expand_dims(sequence, axis=0))[0]

			# console output
			for num, prob in enumerate(pred_arry):
				print(f'{num}: {prob:.2f}')

			# display on feed
			if pred_arry[np.argmax(pred_arry)] > threshold:
				predicted_gloss = glosses[np.argmax(pred_arry)]
				cv2.putText(image, predicted_gloss, (10, image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

		# output to feed
		cv2.imshow('RUNNING PREDICTIONS', image)

		if cv2.waitKey(10) == 27:  # 27 is the esc key
			print("exit safely")
			break

	cap.release()
	cv2.destroyAllWindows()
# %%
