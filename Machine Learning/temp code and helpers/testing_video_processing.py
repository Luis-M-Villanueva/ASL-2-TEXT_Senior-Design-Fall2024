import pathlib
import random
import tensorflow as tf
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Define your dataset directory
data_dir = pathlib.Path(r'D:\senior_design\temp_test_vids')

# Gather all video file paths and their corresponding labels
def get_video_paths_and_labels(data_dir):
	video_paths = list(data_dir.glob('*/*.mp4'))  # Get all mp4 video files
	labels = [p.parent.name for p in video_paths]  # Labels are the folder names
	return video_paths, labels

video_paths, labels = get_video_paths_and_labels(data_dir)

# Split the data into train, validation, and test using sklearn's train_test_split
train_paths, test_paths, train_labels, test_labels = train_test_split(
	video_paths, labels, test_size=0.2, stratify=labels, random_state=42
)

train_paths, val_paths, train_labels, val_labels = train_test_split(
	train_paths, train_labels, test_size=0.25, stratify=train_labels, random_state=42
)  # 0.25 * 0.8 = 0.2

print(f"Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")


# Function to extract frames from video
def format_frames(frame, output_size):
	"""
	Pad and resize an image from a video.
	Args:
		frame: Image that needs to resized and padded. 
		output_size: Pixel size of the output frame image.
	Return:
		Formatted frame with padding of specified output size.
	"""
	frame = tf.image.convert_image_dtype(frame, tf.float32)
	frame = tf.image.resize_with_pad(frame, *output_size)
	return frame

def frames_from_video_file(video_path, n_frames, output_size=(224,224), frame_step=15):
	"""
	Creates frames from each video file present for each category.
	Args:
		video_path: File path to the video.
		n_frames: Number of frames to be created per video file.
		output_size: Pixel size of the output frame image.
	Return:
		A NumPy array of frames in the shape of (n_frames, height, width, channels).
	"""
	result = []
	src = cv2.VideoCapture(str(video_path))
	video_length = src.get(cv2.CAP_PROP_FRAME_COUNT)
	need_length = 1 + (n_frames - 1) * frame_step

	if need_length > video_length:
		start = 0
	else:
		max_start = video_length - need_length
		start = random.randint(0, max_start + 1)

	src.set(cv2.CAP_PROP_POS_FRAMES, start)
	ret, frame = src.read()
	result.append(format_frames(frame, output_size))

	for _ in range(n_frames - 1):
		for _ in range(frame_step):
			ret, frame = src.read()
		if ret:
			frame = format_frames(frame, output_size)
			result.append(frame)
		else:
			result.append(np.zeros_like(result[0]))
	src.release()
	result = np.array(result)[..., [2, 1, 0]]  # Convert BGR to RGB
	return result


# Create a generator function for TensorFlow datasets
# Step 1: Create a label to integer mapping
label_to_index = {label: idx for idx, label in enumerate(sorted(set(labels)))}
index_to_label = {idx: label for label, idx in label_to_index.items()}  # Reverse mapping

# Update the FrameGenerator to use label_to_index
class FrameGenerator:
	def __init__(self, video_paths, labels, n_frames, training=False):
		self.video_paths = video_paths
		self.labels = labels
		self.n_frames = n_frames
		self.training = training

	def __call__(self):
		pairs = list(zip(self.video_paths, self.labels))

		if self.training:
			random.shuffle(pairs)

		for path, label in pairs:
			video_frames = frames_from_video_file(path, self.n_frames)
			yield video_frames, label_to_index[label]  # Map label to its corresponding integer

# Define the output signature for the dataset
output_signature = (
	tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),  # Shape of frames
	tf.TensorSpec(shape=(), dtype=tf.int32)  # Label
)

# Create TensorFlow datasets
train_ds = tf.data.Dataset.from_generator(
	FrameGenerator(train_paths, train_labels, n_frames=10, training=True),
	output_signature=output_signature
)

val_ds = tf.data.Dataset.from_generator(
	FrameGenerator(val_paths, val_labels, n_frames=10),
	output_signature=output_signature
)

test_ds = tf.data.Dataset.from_generator(
	FrameGenerator(test_paths, test_labels, n_frames=10),
	output_signature=output_signature
)

# Configure and batch the datasets for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE).batch(2)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE).batch(2)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE).batch(2)


# Initialize the model (EfficientNetB0)
base_model = tf.keras.applications.EfficientNetB0(include_top=False, input_shape=(224, 224, 3))
base_model.trainable = True

# Build the full model
model = tf.keras.Sequential([
	tf.keras.layers.TimeDistributed(base_model),  # Apply EfficientNet over the time dimension
	tf.keras.layers.GlobalAveragePooling3D(),  # Global pooling across the time and spatial dimensions
	tf.keras.layers.Dense(10)  # Dense layer for classification (assuming 10 classes)
])

# Compile the model
model.compile(
	optimizer='adam',
	loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
	metrics=['accuracy']
)

# Train the model
model.fit(
	train_ds,
	epochs=10,
	validation_data=val_ds,
	callbacks=[tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss')]
)

# Evaluate on the test dataset
test_loss, test_acc = model.evaluate(test_ds)
print(f"Test accuracy: {test_acc:.2f}")
