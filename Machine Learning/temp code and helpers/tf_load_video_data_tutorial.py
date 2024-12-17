# Imports and setup
import tqdm
import random
import pathlib
import collections
import os
import cv2
import numpy as np
import remotezip as rz
import tensorflow as tf
import imageio
from tensorflow_docs.vis import embed

# Define URL for UCF101 dataset
URL = 'https://storage.googleapis.com/thumos14_files/UCF101_videos.zip'

# Function to list files from the remote ZIP
def list_files_from_zip_url(zip_url):
	""" List the files in each class of the dataset given a URL with the zip file.
	Args:
		zip_url: A URL from which the files can be extracted from.
	Returns:
		List of files in each of the classes.
	"""
	files = []
	with rz.RemoteZip(zip_url) as zip:
		for zip_info in zip.infolist():
			files.append(zip_info.filename)
	return files

# Retrieve and filter video files
files = list_files_from_zip_url(URL)
files = [f for f in files if f.endswith('.avi')]
print(files[:10])

# Get class name from filename
def get_class(fname):
	""" Retrieve the name of the class given a filename.
	Args:
		fname: Name of the file in the UCF101 dataset.
	Returns:
		Class that the file belongs to.
	"""
	return fname.split('_')[-3]

# Group files by class
def get_files_per_class(files):
	""" Retrieve the files that belong to each class.
	Args:
		files: List of files in the dataset.
	Returns:
		Dictionary of class names (key) and files (values). 
	"""
	files_for_class = collections.defaultdict(list)
	for fname in files:
		class_name = get_class(fname)
		files_for_class[class_name].append(fname)
	return files_for_class

# Specify number of classes and videos
NUM_CLASSES = 10
FILES_PER_CLASS = 50

files_for_class = get_files_per_class(files)
classes = list(files_for_class.keys())

print('Num classes:', len(classes))
print('Num videos for class[0]:', len(files_for_class[classes[0]]))

# Select a subset of classes and files
def select_subset_of_classes(files_for_class, classes, files_per_class):
	""" Create a dictionary with the class name and a subset of the files in that class.
	Args:
		files_for_class: Dictionary of class names (key) and files (values).
		classes: List of classes.
		files_per_class: Number of files per class of interest.
	Returns:
		Dictionary with class as key and list of specified number of video files in that class.
	"""
	files_subset = dict()
	for class_name in classes:
		class_files = files_for_class[class_name]
		files_subset[class_name] = class_files[:files_per_class]
	return files_subset

files_subset = select_subset_of_classes(files_for_class, classes[:NUM_CLASSES], FILES_PER_CLASS)
print(list(files_subset.keys()))

# Download videos and organize them into subsets
def download_from_zip(zip_url, to_dir, file_names):
	""" Download the contents of the zip file from the zip URL.
	Args:
		zip_url: A URL with a zip file containing data.
		to_dir: A directory to download data to.
		file_names: Names of files to download.
	"""
	with rz.RemoteZip(zip_url) as zip:
		for fn in tqdm.tqdm(file_names):
			class_name = get_class(fn)
			zip.extract(fn, str(to_dir / class_name))
			unzipped_file = to_dir / class_name / fn

			fn = pathlib.Path(fn).parts[-1]
			output_file = to_dir / class_name / fn
			unzipped_file.rename(output_file)

# Split files into training, validation, and test sets
def split_class_lists(files_for_class, count):
	""" Returns the list of files belonging to a subset of data as well as the remainder of
		files that need to be downloaded.
	Args:
		files_for_class: Files belonging to a particular class of data.
		count: Number of files to download.
	Returns:
		Files belonging to the subset of data and dictionary of the remainder of files that need to be downloaded.
	"""
	split_files = []
	remainder = {}
	for cls in files_for_class:
		split_files.extend(files_for_class[cls][:count])
		remainder[cls] = files_for_class[cls][count:]
	return split_files, remainder

# Function to download a UCF-101 subset and split it into training, validation, and test sets
def download_ucf_101_subset(zip_url, num_classes, splits, download_dir):
	""" Download a subset of the UCF101 dataset and split them into various parts, such as
		training, validation, and test.
	Args:
		zip_url: A URL with a ZIP file with the data.
		num_classes: Number of labels.
		splits: Dictionary specifying the training, validation, test, etc. (key) division of data 
				(value is number of files per split).
		download_dir: Directory to download data to.
	Returns:
		Mapping of the directories containing the subsections of data.
	"""
	files = list_files_from_zip_url(zip_url)
	files_for_class = get_files_per_class(files)
	classes = list(files_for_class.keys())[:num_classes]

	for cls in classes:
		random.shuffle(files_for_class[cls])
	
	files_for_class = {x: files_for_class[x] for x in classes}

	dirs = {}
	for split_name, split_count in splits.items():
		print(split_name, ":")
		split_dir = download_dir / split_name
		split_files, files_for_class = split_class_lists(files_for_class, split_count)
		download_from_zip(zip_url, split_dir, split_files)
		dirs[split_name] = split_dir

	return dirs

# Download data subsets
download_dir = pathlib.Path('./UCF101_subset/')
subset_paths = download_ucf_101_subset(URL, num_classes=NUM_CLASSES, splits={"train": 30, "val": 10, "test": 10}, download_dir=download_dir)

# Count videos
video_count_train = len(list(download_dir.glob('train/*/*.avi')))
video_count_val = len(list(download_dir.glob('val/*/*.avi')))
video_count_test = len(list(download_dir.glob('test/*/*.avi')))
video_total = video_count_train + video_count_val + video_count_test
print(f"Total videos: {video_total}")

# Create frames from each video file
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
	result = np.array(result)[..., [2, 1, 0]]
	return result

# Visualize video data
def to_gif(images):
	converted_images = np.clip(images * 255, 0, 255).astype(np.uint8)
	imageio.mimsave('./animation.gif', converted_images, fps=10)
	return embed.embed_file('./animation.gif')

# Create frame generator class
class FrameGenerator:
	def __init__(self, path, n_frames, training=False):
		""" Returns a set of frames with their associated label. 
		Args:
			path: Video file paths.
			n_frames: Number of frames. 
			training: Boolean to determine if training dataset is being created.
		"""
		self.path = path
		self.n_frames = n_frames
		self.training = training
		self.class_names = sorted(set(p.name for p in self.path.iterdir() if p.is_dir()))
		self.class_ids_for_name = dict((name, idx) for idx, name in enumerate(self.class_names))

	def get_files_and_class_names(self):
		video_paths = list(self.path.glob('*/*.avi'))
		classes = [p.parent.name for p in video_paths]
		return video_paths, classes

	def __call__(self):
		video_paths, classes = self.get_files_and_class_names()
		pairs = list(zip(video_paths, classes))

		if self.training:
			random.shuffle(pairs)

		for path, name in pairs:
			video_frames = frames_from_video_file(path, self.n_frames)
			label = self.class_ids_for_name[name]
			yield video_frames, label



# Test out the FrameGenerator object before wrapping it as a TensorFlow Dataset object
# For the training dataset, ensure training mode is enabled so the data will be shuffled.

fg = FrameGenerator(subset_paths['train'], n_frames=10, training=True)
frames, label = next(fg())

print(f"Shape: {frames.shape}")
print(f"Label: {label}")

# Now create a TensorFlow data input pipeline
# This pipeline allows you to feed the data into a deep learning model.
# Each element is a set of frames and its associated label.

# Define the output signature for the dataset
output_signature = (
    tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.float32),
    tf.TensorSpec(shape=(), dtype=tf.int16)
)

# Create the training dataset
train_ds = tf.data.Dataset.from_generator(
    FrameGenerator(subset_paths['train'], 10, training=True),
    output_signature=output_signature
)

# Check if the labels are shuffled
for frames, labels in train_ds.take(10):
    print(labels)

# Create the validation dataset
val_ds = tf.data.Dataset.from_generator(
    FrameGenerator(subset_paths['val'], 10),
    output_signature=output_signature
)

# Print the shapes of the data
train_frames, train_labels = next(iter(train_ds))
print(f'Shape of training frames: {train_frames.shape}')
print(f'Shape of training labels: {train_labels.shape}')

val_frames, val_labels = next(iter(val_ds))
print(f'Shape of validation frames: {val_frames.shape}')
print(f'Shape of validation labels: {val_labels.shape}')

# Configure the dataset for performance
# Use buffered prefetching to yield data from disk without blocking I/O
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

# Batch the data
# When working with video data, ensure it's shaped as [batch_size, num_frames, height, width, channels]
train_ds = train_ds.batch(2)
val_ds = val_ds.batch(2)

train_frames, train_labels = next(iter(train_ds))
print(f'Shape of batched training frames: {train_frames.shape}')
print(f'Shape of batched training labels: {train_labels.shape}')

val_frames, val_labels = next(iter(val_ds))
print(f'Shape of batched validation frames: {val_frames.shape}')
print(f'Shape of batched validation labels: {val_labels.shape}')

# Now proceed to define the model using a pre-trained EfficientNetB0
net = tf.keras.applications.EfficientNetB0(include_top=False)
net.trainable = False

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(scale=255),  # Rescale input images
    tf.keras.layers.TimeDistributed(net),  # Distribute the EfficientNet model over time steps
    tf.keras.layers.Dense(10),  # Dense layer for classification
    tf.keras.layers.GlobalAveragePooling3D()  # Global average pooling over the 3D output
])

# Compile the model with the Adam optimizer and sparse categorical cross-entropy loss
model.compile(optimizer='adam',
				loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
				metrics=['accuracy'])

# Train the model, with early stopping based on validation loss
model.fit(train_ds,
			epochs=10,
			validation_data=val_ds,
			callbacks=[tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss')])
