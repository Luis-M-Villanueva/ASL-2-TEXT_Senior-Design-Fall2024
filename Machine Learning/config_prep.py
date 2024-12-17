'''
config.py
- Includes all libraries and functions and some variables
- as well as test mediapipe and opencv
'''

# %%
import mediapipe as mp
import cv2
import numpy as np
import os
import tensorflow as tf
# from tensorflow.keras.models import Sequential          # type: ignore
# from tensorflow.keras.layers import Input, LSTM, Dense         # type: ignore
# from tensorflow.keras.callbacks import TensorBoard      # type: ignore
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.utils import to_categorical   	# type: ignore
# from tensorflow.keras.models import load_model			# type: ignore

print(f"\n-------------------------\nPython version: {__import__('sys').version}")
print(f"TensorFlow version: {tf.__version__}")
print(f"MediaPipe version: {mp.__version__}")
print(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")

print()
print("When working with WSL, remember these:")
print("PS: usbipd list; usbipd attach --wsl --busid 1-8")
print("WSL: ls -al /dev/video*\n-------------------------\n")


# %%
# constants
DIR = os.getcwd()
TOTAL_NUM_FRAMES = 15
NUM_SEQUENCES = 35
INPUT_SHAPE_FACES = 1662
INPUT_SHAPE_NO_FACES = 258

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
	# mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
	# 						mp_drawing.DrawingSpec(color=(245,117,66), thickness=1, circle_radius=1), 
	# 						mp_drawing.DrawingSpec(color=(245,66,230), thickness=1, circle_radius=1)
	# 						)

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
	# if results.face_landmarks:
	# 	face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten()
	# 	print(face.shape)
	# else:
	# 	face = np.zeros(468 * 3)

	# check if that 1404 matches up with face tesselation/contours
	# done. it is 1404.

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
	
	# return np.concatenate([pose, face, lh, rh])
	return np.concatenate([pose, lh, rh])


# %%
def keypoint_capture_test():
	cap = cv2.VideoCapture(0)
	# Set mediapipe model 
	with mp_holistic.Holistic(min_detection_confidence=0.25, min_tracking_confidence=0.25) as holistic:
		while cap.isOpened():
			ret, frame = cap.read()
			image, results = mediapipe_detectionConversion(frame, holistic)
			
			apply_landmarks(image, results)

			# keypoints = get_all_keypoints(results)

			cv2.imshow('Testing if mediapipe works. Esc to exit', image)
			if cv2.waitKey(10) == 27:  # 27 is the esc key
				break
		cap.release()
		cv2.destroyAllWindows()

# keypoint_capture_test()
