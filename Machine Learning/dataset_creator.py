import cv2
import numpy as np
import mediapipe as mp
import os
import csv
from config_prep import DIR, TOTAL_NUM_FRAMES, NUM_SEQUENCES, mediapipe_detectionConversion, apply_landmarks, get_all_keypoints

# %%
# MAKE DATASET FOLDER (SINGULAR)
file_path = os.path.join(os.path.join(DIR, 'datasets'), input("Folder name for dataset: "))
if os.path.exists(file_path):
	print("Folder exists already!")
else:
	os.makedirs(file_path, exist_ok=True)

totalNum_frames = TOTAL_NUM_FRAMES
num_sequences = NUM_SEQUENCES

# %%
# DATASET CAPTURE
glosses = []
def capture_gloss_data(base_path, totalNum_frames, glosses):
	cap = cv2.VideoCapture(0)
	mp_holistic = mp.solutions.holistic
	with mp_holistic.Holistic(min_detection_confidence=0.35, min_tracking_confidence=0.35) as holistic:
		while True:
			# get gloss, remove whitespace
			gloss = input("Gloss name (press ESC to exit): ").strip()
			if gloss.lower() == "esc":
				print("Exiting...")
				break
			
			glosses.append(gloss)  # add gloss to the list
			
			# make folder. if exists, overwrite. if not, make it
			gloss_path = os.path.join(base_path, f"{len(glosses)-1}_{gloss}")
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
						cv2.putText(image, f'Collecting frames for {gloss} Video {sequence+1}/{num_sequences}', (15, 12),
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

# write to csv.
with open('latest_glosses.csv', 'w', newline='') as file:
	writer = csv.writer(file)
	for gloss in glosses:
		writer.writerow([gloss])

print("Done! don't forget to rename csv.")

# glosses must be np array bc can't use glosses.shape[0] w/o it
# glosses = np.array(glosses)