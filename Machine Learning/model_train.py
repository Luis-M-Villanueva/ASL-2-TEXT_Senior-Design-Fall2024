# %%
import os
import numpy as np
import csv
import tensorflow as tf
from sklearn.model_selection import train_test_split
from config_prep import DIR, TOTAL_NUM_FRAMES, INPUT_SHAPE_FACES, INPUT_SHAPE_NO_FACES
from tensorflow.keras.models import Sequential			# type: ignore
from tensorflow.keras.layers import Input, LSTM, Dense	# type: ignore
from tensorflow.keras.utils import to_categorical		# type: ignore

# %%
# get dataset folder
# file_path = os.path.join(DIR, input("Folder name for dataset: "))

print("Choose dataset folder...")
subfolders = [f for f in os.listdir(os.path.join(DIR, 'datasets')) if os.path.isdir(os.path.join(DIR, 'datasets', f))]
print(subfolders)

for index, folder in enumerate(subfolders):
	print(f"{index}: {folder}")

choice = int(input("See terminal. Pick subfolder that is the dataset folder: "))

file_path = os.path.join(DIR, 'datasets', subfolders[choice])
print(f"\nUsing dataset folder: {file_path}\n")

# %%
# get glosses, label map
print("Ensure that glosses are in order. Enumerate them.")
glosses = [f for f in os.listdir(file_path) if os.path.isdir(os.path.join(file_path, f))]
print(f"Glosses in this dataset: {glosses}")

# write to csv in cwd (github)
with open(os.path.join(os.path.join(DIR, 'models'), f'renameMe_{subfolders[choice]}.csv'), 'w', newline='') as file:
	writer = csv.writer(file)
	for gloss in glosses:
		writer.writerow([gloss])
print("CSV written.")

# convert to np array for use of shape[0]
glosses = np.array(glosses)

# enumerate the glosses
label_map = {label:num for num, label in enumerate(glosses)}

# %%
# extract sequences of frames for each gloss and apply labels
# i.e.
# catogories is 2d array of sequence1, frame 0, 1, 2, 3...
# labels is 1d array corresponding to the sequences, so it could be 0, 0, 1... for hello, hello, thanks...
categories, labels = [], []
for gloss in glosses:
	for sequence in np.array(os.listdir(os.path.join(file_path, gloss))).astype(int):
		window = []
		for frame_num in range(TOTAL_NUM_FRAMES):
			res = np.load(os.path.join(file_path, gloss, str(sequence), "{}.npy".format(frame_num)))
			window.append(res)
		categories.append(window)
		labels.append(label_map[gloss])

# %%
# prep data
x = np.array(categories)
y = to_categorical(labels).astype(int) # one hot encoding. one between _, _, _

testSplit = 0.2

print(f"Using test size of {testSplit}")
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=testSplit)

# log_dir = os.path.join('Logs')
# tb_callback = TensorBoard(log_dir=log_dir)

# %%
# creating model and compile
# from tensorflow.keras.models import Sequential
import tensorflow as tf
from tensorflow.keras.layers import (	# type: ignore
    Conv1D,
    Dropout,
    Input,
    LSTM,
    Dense,
    Attention,
    Add,
    LayerNormalization,
    BatchNormalization
)
from tensorflow.keras.models import Model 	# type: ignore

# input layer
input_layer = Input(shape=(TOTAL_NUM_FRAMES, INPUT_SHAPE_NO_FACES))

# 1d convolutional layers to extract spatial features
conv1 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(input_layer)
conv1_norm = BatchNormalization()(conv1)  # batch normalization for regularization
conv2 = Conv1D(filters=128, kernel_size=5, activation='relu', padding='same')(conv1_norm)
conv2_norm = BatchNormalization()(conv2)
conv_dropout = Dropout(0.2)(conv2_norm)  # reduced dropout rate

# lstm layers for temporal processing
lstm1 = LSTM(128, return_sequences=True, activation='tanh', dropout=0.2, recurrent_dropout=0.2)(conv_dropout)
lstm2 = LSTM(256, return_sequences=True, activation='tanh', dropout=0.2, recurrent_dropout=0.2)(lstm1)

# attention mechanism
attention_scores = Attention()([lstm2, lstm2])  # attention applied to the lstm outputs
attention_add = Add()([attention_scores, lstm2])  # add residual connection for better gradient flow
attention_norm = LayerNormalization()(attention_add)  # normalize the attention outputs

# additional lstm layers after attention
lstm3 = LSTM(128, return_sequences=True, activation='tanh', dropout=0.2, recurrent_dropout=0.2)(attention_norm)
lstm4 = LSTM(64, return_sequences=False, activation='tanh', dropout=0.2, recurrent_dropout=0.2)(lstm3)

# fully connected layers for classification
dense1 = Dense(128, activation='relu')(lstm4)
dense1_norm = BatchNormalization()(dense1)  # batch normalization for dense layers
dense2 = Dense(64, activation='relu')(dense1_norm)
dense2_norm = BatchNormalization()(dense2)
output_layer = Dense(glosses.shape[0], activation='softmax')(dense2_norm)

# define the model
model = Model(inputs=input_layer, outputs=output_layer)

# compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['categorical_accuracy']
)

# define callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_categorical_accuracy',  # monitor validation accuracy
    patience=10,
    restore_best_weights=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_categorical_accuracy', 
    factor=0.5, 
    patience=5, 
    min_lr=1e-6  # prevent lr from reducing too much
)

# train the model
history = model.fit(
    x_train, y_train,
    validation_split=0.2,  # use 20% of the data for validation
    epochs=150,
    batch_size=32,
    callbacks=[early_stopping, reduce_lr]
)

# print the model summary
model.summary()

# model_name = input("Name model: ")
model_name = f"renameMe_{subfolders[choice]}"
model.save(os.path.join(os.path.join(DIR, 'models'), f"renameMe_{subfolders[choice]}.keras"))
# %%
