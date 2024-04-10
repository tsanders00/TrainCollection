import glob
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.models import Sequential
from PIL import Image
from sklearn.model_selection import train_test_split
import numpy as np
from matplotlib import pyplot as plt


# load data function
def load_data(path: str):
    """
    Load images
    :param path: path to images
    :return: images as numpy array
    """
    images = []
    files = glob.glob(path)

    for file in files:
        img = Image.open(file)
        images.append(np.array(img))

    return images


def pad_images(list):
    """
    function to pad images to the same size
    :param list: images you want to pad
    :return: padded images
    """
    max_shape = np.max([arr.shape for arr in list], axis=0)
    padded_images = []

    for arr in list:
        pad_width = [(0, max_dim - cur_dim) for max_dim, cur_dim in zip(max_shape, arr.shape)]
        padded_arr = np.pad(arr, pad_width, mode='minimum')
        padded_images.append(padded_arr)

    return padded_images

# model
model = Sequential()
model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', data_format='channels_last'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# load data
train_cat = load_data('./cats_and_dogs/train/cats/*.jpg')
train_dog = load_data('./cats_and_dogs/train/dogs/*.jpg')
validation_cat = load_data('./cats_and_dogs/validation/cats/*.jpg')
validation_dog = load_data('./cats_and_dogs/validation/dogs/*.jpg')
test_data = load_data('./cats_and_dogs/test/*.jpg')

# concat
train_images = train_cat + train_dog + validation_cat + validation_dog
train_labels = [0] * len(train_cat) + [1] * len(train_dog) + [0] * len(validation_cat) + [1] * len(validation_dog)

# pad
train_images = pad_images(train_images)
test_data = pad_images(test_data)

# convert
train_images = np.array(train_images)
train_labels = np.array(train_labels)
test_images = np.array(test_data)

# Normalize images
print('Data type: %s' % train_images.dtype)
print('Min: %.3f, Max: %.3f' % (train_images.min(), train_images.max()))

# Calculate the minimum and maximum pixel values in your dataset
min_val = np.min(train_images)
max_val = np.max(train_images)

# Normalize the data using Min-Max scaling
train_images_normalized = (train_images - min_val) / (max_val - min_val)

# print('Data type: %s' % train_images_normalized.dtype)
# print('Min: %.3f, Max: %.3f' % (train_images_normalized.min(), train_images_normalized.max()))

# split
x_train, x_val, y_train, y_val = train_test_split(train_images_normalized, train_labels, test_size=0.2, random_state=42)

# train
# laptop cannot handle amount of data unfortunately
# rest of code is actually not tested
model.fit(train_images, train_labels, validation_data=(x_val, y_val), verbose=3)

# prediction task
# var pred contains probability for label for each test image
predictions = model.predict(test_images)
# var pred_prob contains converted labels for the test images, either 0 or 1, 0=cat 1=dog
predictions_prob = (predictions > 0.5).astype(np.uint8)

imgplot = plt.imshow(test_images[0])
plt.show()
print(f'Predicted label for test image: {predictions_prob[0]}')