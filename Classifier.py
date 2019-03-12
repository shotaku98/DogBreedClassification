import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from PIL import Image
import os
from random import shuffle

DIR="C:/Projects/DATASETS/DOGBREED/train"

def get_size_statistics():
	heights=[]
	width=[]
	imgcount=0
	for img in os.listdir(DIR):
		path = os.path.join(DIR,img)
		if "DS_Store" not in path:
			data=np.array(Image.open(path))
			heights.append(data.shape[0])
			width.append(data.shape[1])
			imgcount += 1
	aheight=sum(heights)/len(heights)
	awidth=sum(width)/imgcount
	print(imgcount)

def label_img(name):
	word_label = name.split('-')[0]
	if word_label== 'golden_retriever':return np.array([1,0])
	elif word_label== 'shetland_sheepdog': return np.array([0,1])

def load_training_data():
	train_data=[]
	for img in os.listdir(DIR):
		label=label_img(img)
		path = os.path.join(DIR,img)
		if "DS_Store" not in path:
			img=Image.open(path)
			img=img.convert('L')
			img=img.resize((100,100),Image.ANTIALIAS)
			train_data.append([np.array(img),label])

	shuffle(train_data)
	return train_data

train_data=load_training_data()
plt.imshow(train_data[43][0])

trainImages = np.array([i[0] for i in train_data]).reshape(-1,100,100,1)
trainLabels= np.array([i[1] for i in train_data])

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers. normalization import BatchNormalization

model = Sequential()
model.add(Conv2D(32,kernel_size = (3,3), activation = 'relu', input_shape=(100,100,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(96, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(96, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(2, activation = 'softmax'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(trainImages,trainLabels, batch_size = 75 , epochs = 1 , verbose=1 )



TEST_DIR='C:/Projects/DATASETS/DOGBREED/test'

def load_test_data():
	test_data=[]
	for img in os.listdir(TEST_DIR):
		label = label_img (img)
		path =  os.path.join (TEST_DIR,img)
		if DS_Store not in path:
			img = Image.open(path)
			img = img.convert('L')
			img = img.resize((100,100), Image.ANTIALIAS)
			test_data.append([np.array(img),label])
	shuffle(test_data)
	return test_data

test_data = load_test_data()

testImages = np.array([i[0] for i in test_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
testLabels = np.array([i[1] for i in test_data])

loss, acc = model.evaluate(testImages, testLabels, verbose = 0)
print(acc * 100)

