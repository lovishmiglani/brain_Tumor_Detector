import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import tensorflow as tf
from PIL import Image
import imutils
from keras.models import load_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import normalize, to_categorical
from tensorflow.keras.preprocessing import image
from keras.applications.inception_resnet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Flatten, Dropout, Dense, Conv2D, MaxPooling2D, Activation

def crop_brain_contour(image):
    image_width = 240
    image_height = 240

    # Convert the image to grayscale, and blur it slightly
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold the image, then perform a series of erosions + dilations to remove any small regions of noise
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find contours in thresholded image, then grab the largest one
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)

    # Find the extreme points
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])

    # crop new image out of the original image using the four extreme points (left, right, top, bottom)
    new_image = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]

    resized_image = cv2.resize(new_image, dsize=(image_width, image_height), interpolation=cv2.INTER_CUBIC)
    # normalize values
    normalized_image = resized_image / 255.

    return normalized_image

# Load the data
image_directory = '/Users/lovishmiglani/Desktop/brain_tumor_detection/brain_img'
no_tumor_images = os.listdir(os.path.join(image_directory, 'no'))
yes_tumor_images = os.listdir(os.path.join(image_directory, 'yes'))

dataset = []
label = []

# Creating label for brain not having tumor
for image_name in no_tumor_images:
    if image_name.split('.')[-1].lower() == 'jpg':
        image_path = os.path.join(image_directory, 'no', image_name)
        if os.path.exists(image_path):
            image = cv2.imread(image_path)
            if image is not None:
                # Apply preprocessing to enhance the image
                preprocessed_image = crop_brain_contour(image)
                # Resize the image
                INPUT_SIZE = (64, 64)
                preprocessed_image = cv2.resize(preprocessed_image, dsize=INPUT_SIZE, interpolation=cv2.INTER_CUBIC)
                dataset.append(np.array(preprocessed_image))
            else:
                print(f"Failed to read image: {image_path}")
        else:
            print(f"Image not found: {image_path}")
        label.append(0)

# Creating label for brain not having tumor
for image_name in yes_tumor_images:
    if image_name.split('.')[-1].lower() == 'jpg':
        image_path = os.path.join(image_directory, 'yes', image_name)
        if os.path.exists(image_path):
            image = cv2.imread(image_path)
            if image is not None:
                # Apply preprocessing to enhance the image
                preprocessed_image = crop_brain_contour(image)
                # Resize the image
                INPUT_SIZE = (64, 64)
                preprocessed_image = cv2.resize(preprocessed_image, dsize=INPUT_SIZE, interpolation=cv2.INTER_CUBIC)
                dataset.append(np.array(preprocessed_image))
            else:
                print(f"Failed to read image: {image_path}")
        else:
            print(f"Image not found: {image_path}")
        label.append(1)

# Convert labels to a numpy array
label = np.array(label)
dataset = np.array(dataset)

print('Dataset: ', len(dataset))
print('Label: ', len(label))

X_train, X_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state=2023)

X_train = normalize(X_train, axis=0)
X_test = normalize(X_test, axis=0)

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# model.fit(X_train, y_train, 
# batch_size=32, 
# verbose=1, epochs=100, 
# validation_data=(X_test, y_test),
# shuffle=False)

history = model.fit(
    X_train, y_train,
    batch_size=32,
    verbose=1,
    epochs=100,
    validation_data=(X_test, y_test),
    shuffle=False
)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training and validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

model.save('/Users/lovishmiglani/Desktop/brain_tumor_detection/BrainTumorDetec.h5')
model = load_model('/Users/lovishmiglani/Desktop/brain_tumor_detection/BrainTumorDetec.h5')
def make_prediction(img):
    
    input_img = np.expand_dims(img, axis=0)
    
    res = (model.predict(input_img) > 0.5).astype("int32")
    return res

def show_result(img):
    img_path = os.path.join(image_directory, 'pred', img)
    
    # Check if the image file exists
    if os.path.exists(img_path):
        # Read the image using cv2
        image = cv2.imread(img_path)
        
        # Check if the image is successfully read
        if image is not None:
            # Convert the image to an RGB PIL Image
            img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Resize the image
            img = img.resize((64, 64))
            
            # Convert the image back to a numpy array if needed
            img = np.array(img)
            
            plt.imshow(img)
            plt.show()
    
            pred = make_prediction(img)
            if pred:
                print("Tumor Detected")
            else:
                print("No Tumor")
show_result('pred4.jpg')