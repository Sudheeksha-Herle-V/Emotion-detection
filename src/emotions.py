import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Command line argument
ap = argparse.ArgumentParser()
ap.add_argument("--mode", help="train/display")
mode = ap.parse_args().mode

# Plot accuracy and loss curves
def plot_model_history(model_history):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    
    # Calculate ticks for x-axis
    num_epochs = len(model_history.history['accuracy'])
    tick_step = max(1, num_epochs // 10)  # Ensure tick_step is at least 1
    
    axs[0].plot(range(1, num_epochs + 1), model_history.history['accuracy'])
    axs[0].plot(range(1, num_epochs + 1), model_history.history['val_accuracy'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1, num_epochs + 1, tick_step))
    axs[0].legend(['train', 'val'], loc='best')
    
    axs[1].plot(range(1, num_epochs + 1), model_history.history['loss'])
    axs[1].plot(range(1, num_epochs + 1), model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1, num_epochs + 1, tick_step))
    axs[1].legend(['train', 'val'], loc='best')
    
    fig.savefig('plot.png')
    print("Before plotting")
    plt.show(block=False)
    print("After plotting")
    plt.close()

# Define data generators
train_dir = 'data/train'
val_dir = 'data/test'
batch_size = 64
num_epoch = 50

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

# Creating TensorFlow datasets from generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode='categorical'
)

validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(48, 48),
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode='categorical'
)

# Convert to tf.data.Dataset
train_dataset = tf.data.Dataset.from_generator(
    lambda: train_generator,
    output_signature=(
        tf.TensorSpec(shape=(None, 48, 48, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 7), dtype=tf.float32)
    )
).repeat()

validation_dataset = tf.data.Dataset.from_generator(
    lambda: validation_generator,
    output_signature=(
        tf.TensorSpec(shape=(None, 48, 48, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 7), dtype=tf.float32)
    )
).repeat()

# Calculate steps per epoch
steps_per_epoch = train_generator.samples // batch_size
validation_steps = validation_generator.samples // batch_size

print(f"Training steps per epoch: {steps_per_epoch}")
print(f"Validation steps per epoch: {validation_steps}")

# Train the model
if mode == "train":
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Conv2D(128, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(1024, activation='relu'),
        Dropout(0.5),
        Dense(7, activation='softmax')
    ])

    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=0.0001),
        metrics=['accuracy']
    )
    model_info = model.fit(
        train_dataset,
        steps_per_epoch=steps_per_epoch,
        epochs=num_epoch,
        validation_data=validation_dataset,
        validation_steps=validation_steps
    )
    plot_model_history(model_info)

    try:
        model.save('model.h5')  # This saves the whole model in a single file
        print("Model saved successfully as model.h5.")
    except Exception as e:
        print(f"Error saving model: {e}")

# Save model architecture separately if needed
    try:
        model_json = model.to_json()
        with open('model.json', 'w') as json_file:
            json_file.write(model_json)
        print("Model architecture saved successfully as model.json.")
    except Exception as e:
        print(f"Error saving model architecture: {e}")

# Save model weights to a separate file with the expected suffix
    try:
        model.save_weights('model.weights.h5')  # Custom suffix if required by your system
        print("Model weights saved successfully as model.weights.h5.")
    except Exception as e:
        print(f"Error saving model weights: {e}")

# Emotions will be displayed on your face from the webcam feed
elif mode == "display":
    from tensorflow.keras.models import model_from_json

    # Load model architecture from JSON file
    with open('model.json', 'r') as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json)

    # Load model weights from H5 file
    model.load_weights('model.h5')

    # Prevents OpenCL usage and unnecessary logging messages
    cv2.ocl.setUseOpenCL(False)

    # Dictionary which assigns each label an emotion (alphabetical order)
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

    # Start the webcam feed
    cap = cv2.VideoCapture(0)
    while True:
        # Find haar cascade to draw bounding box around face
        ret, frame = cap.read()
        if not ret:
            break
        facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            cv2.putText(frame, emotion_dict[maxindex], (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Video', cv2.resize(frame, (1600, 960), interpolation=cv2.INTER_CUBIC))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
