#!/usr/bin/env python
# coding: utf-8

# In[7]:


import os 
import keras
from six.moves import cPickle as pickle
import numpy as np
import tensorflow as tensor_flow
import matplotlib.image as mpimg
from scipy import misc


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.utils import np_utils, plot_model
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam


# In[8]:


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


# In[9]:


def load_images(folder, min_num_images):
    """Load the data for a single letter label."""
    image_files = os.listdir(folder)
    dataset = np.ndarray(shape=(len(image_files), image_width, image_height),
            dtype=np.float32)
    labels = np.ndarray(shape=(len(image_files)),
                        dtype=np.int32)
    print(folder)
    num_images = 0
    for image in image_files:
        try:
            image_file = os.path.join(folder, image)
            img = mpimg.imread(image_file)
            gray = rgb2gray(img)
            gray_scaled = misc.imresize(gray, (image_width, image_height))
            image_data = (gray_scaled.astype(float) - pixel_depth / 2) / pixel_depth
            if image_data.shape != (image_width, image_height):
                raise Exception('Unexpected image shape: %s' % str(image_data.shape))
            dataset[num_images, :, :] = image_data
            if str(image_file[-5]) == "1":
                labels[num_images] = 1
            else:
                labels[num_images] = 0
            num_images = num_images + 1
            print(num_images)
        except IOError as e:
            print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')

    dataset = dataset[0:num_images, :, :]
    labels = labels[0:num_images]
    if num_images < min_num_images:
        raise Exception('Many fewer images than expected: %d < %d' %
                (num_images, min_num_images))

    print('Full dataset tensor:', dataset.shape)
    print('Mean:', np.mean(dataset))
    print('Standard deviation:', np.std(dataset))
    return dataset, labels


# In[10]:


def get_dataset():
    set_filename = "CXR_png.pickle"

    if not os.path.isfile(set_filename) :
        dataset , labels = load_images("CXRpng",4)
        data = { "dataset" : dataset, 
            "labels"  : labels }
        try:
            with open(set_filename, 'wb') as f:
                pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print('Unable to save data to', set_filename, ':', e)

    with open(set_filename, 'rb') as f:
        data = pickle.load(f)
        dataset = data["dataset"]
        labels = data["labels"]
        # print "===LABELS==="
        # print labels
        
        test_dataset = np.ndarray(shape=(202, image_width, image_height),
                            dtype=np.float32)
        test_labels = np.ndarray(shape=(202),
                            dtype=np.int32)
        train_dataset = np.ndarray(shape=(460, image_width, image_height),
                            dtype=np.float32)
        train_labels = np.ndarray(shape=(460),
                            dtype=np.int32)
        
        #30% test
        num_test = 0
        for i in np.random.randint(low=0, high=662, size=202):
            test_dataset[num_test,:,:] = dataset[i,:,:] 
            test_labels[num_test] = labels[i]
            num_test = num_test + 1
        
        #70% train
        num_train = 0
        for i in np.random.randint(low=0, high=662, size=460):
            train_dataset[num_train,:,:] = dataset[i,:,:] 
            train_labels[num_train] = labels[i]
            num_train = num_train + 1

        del dataset  # hint to help gc free up memory
        del labels
        del data
        print('Training set', train_dataset.shape, train_labels.shape)
        #print('Validation set', valid_dataset.shape, valid_labels.shape)
        print('Test set', test_dataset.shape, test_labels.shape)
    #return train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels
    return train_dataset, train_labels, test_dataset, test_labels


# In[11]:


num_labels = 2
image_width = 640
image_height = 480
pixel_depth = 255.0
def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_width, image_height, 1)).astype(np.float32)
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return dataset, labels


# In[6]:


train_dataset, train_labels, test_dataset, test_labels = get_dataset()

train_dataset, train_labels = reformat(train_dataset, train_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

batch_size = 16
patch_size = 5
depth = 16
num_hidden = 64
num_channels = 1

graph1 = tensor_flow.Graph()
with graph1.as_default():
    # Input data.
    tf_train_dataset = tensor_flow.placeholder(tensor_flow.float32, shape=(batch_size, image_width, image_height, num_channels),name="input")
    tensor_flow.summary.image('train_input', tf_train_dataset, 3)
    tf_train_labels  = tensor_flow.placeholder(tensor_flow.float32, shape=(batch_size, num_labels),name="labels")


# In[ ]:


xsize = (640,480,1)

#Normalize image bytes (0-255) to a scale of 0-1
train_dataset = train_dataset / 255
test_dataset = test_dataset / 255

#Reshape x for tensorflow (channels last)
train_dataset = train_dataset.reshape(train_dataset.shape[0], xsize[0], xsize[1], xsize[2])
test_dataset = test_dataset.reshape(test_dataset.shape[0], xsize[0], xsize[1], xsize[2])

#Convert y to a categorical variable
#train_labels = np_utils.to_categorical(train_labels)
#test_labels = np_utils.to_categorical(test_labels)

#Double check shape
print(train_labels.shape)
print(test_labels.shape)
classes = test_labels.shape[1]


# In[ ]:


#Model configurations
epochs = 15
batchsize = 2000

#Create model
cnn = Sequential()

#Start with a convolutional layer with 50 filters and a 4x4 kernel
cnn.add(Conv2D(48, kernel_size=(4,4), activation='relu', input_shape=xsize, padding='valid'))
cnn.add(MaxPooling2D(pool_size=(2,2)))

#Add two more convolutions and a pooling layer
cnn.add(Conv2D(12, kernel_size=(3,3), activation='relu', input_shape=xsize, padding='valid'))
cnn.add(Conv2D(6, kernel_size = (2,2), activation='relu', input_shape=xsize, padding='valid'))
cnn.add(MaxPooling2D(pool_size=(2,2)))

#Add a flatten
cnn.add(Flatten())

#Finish with two fully connected layers to improve accuracy then convert to visible output
cnn.add(Dense(128, activation='relu'))
cnn.add(Dense(classes, activation='softmax'))

#Compile model using log loss function as objective function
cnn.compile(loss=categorical_crossentropy, optimizer=Adam(), metrics=['accuracy'])

#Train model
trained = cnn.fit(train_dataset, train_labels, batch_size=batchsize, epochs=epochs, verbose=1, validation_data=(test_dataset, test_labels))


# In[ ]:


#Visualise what was ran, start with model.summary() and pydot.Graph:
print(cnn.summary())
display(SVG(model_to_dot(cnn).create(prog='dot', format='svg')))

#Show model performance using code from Keras site: https://keras.io/visualization/
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

# Plot training & validation accuracy values
plt.plot(trained.history['acc'])
plt.plot(trained.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(trained.history['loss'])
plt.plot(trained.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# In[ ]:





# In[ ]:




