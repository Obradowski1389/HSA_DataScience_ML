import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense
from keras.models import Model
from PIL import Image
from keras.models import load_model
import os
from .color_classes import color_classes_rev


global model
global train_in_slo, test_in_slo, train_out_slo, test_out_slo


def build_nn(save, epoch, batch, predict_path, out_name):
    global model
    global train_in_slo, test_in_slo, train_out_slo, test_out_slo

    cur = os.path.dirname(os.path.abspath(__file__))
    parent = os.path.normpath(os.path.join(cur, '..'))
    dir_in_path = os.path.join(parent, "npy", "input")
    dir_out_path = os.path.join(parent, "npy", "out")
    in_path = os.path.join(dir_in_path, "in_slo.npy")
    out_path = os.path.join(dir_out_path, "out_slo.npy")

    # Load the input and output matrices
    in_slo = np.load(in_path)
    out_slo = np.load(out_path)

    MODELS_DIR = os.path.normpath(os.path.join(cur, '..', 'Model'))
    MODEL_TRAIN_PATH = os.path.join(MODELS_DIR, 'model_learn.hdf5')
    MODEL_VALID_PATH = os.path.join(MODELS_DIR, 'model_valid.hdf5')
    MODEL_TRAIN_MONITOR = os.path.join(MODELS_DIR, 'accuracy')
    MODEL_VALID_MONITOR = os.path.join(MODELS_DIR, 'val_accuracy')
    MODEL_MODE = os.path.join(MODELS_DIR, 'max')
    HISTORY_PATH = os.path.join(MODELS_DIR, 'history.csv')

    if save != 1:
        model = load_model(os.path.join(MODELS_DIR, 'model.h5'))
        in_ns = np.load(predict_path)

        # Perform prediction on input data
        predicted_out_ns = model.predict(in_ns.reshape(-1, in_ns.shape[2]))

        # Convert predictions to class indices
        predicted_classes = np.argmax(predicted_out_ns, axis=1)

        out_data = predicted_classes
        img1c_color = Image.new("RGB", (in_ns.shape[1], in_ns.shape[0]))
        out_data = out_data.reshape((in_ns.shape[0], in_ns.shape[1]))

        for i in range(in_ns.shape[0]):
            for j in range(in_ns.shape[1]):
                mask = out_data[i][j]
                rgb = color_classes_rev[mask]
                img1c_color.putpixel((j, i), rgb)

        # Save the image
        output_path = os.path.join(cur, "..", "img", "output", out_name)
        img1c_color.save(output_path)  # Rotate the image 270 degrees clockwise
    else:
        # Reshape the input array to match the number of samples
        in_slo_reshaped = in_slo.reshape((-1, in_slo.shape[2]))
        out_slo_reshaped = out_slo.flatten()

        # Split the data into train and test sets
        train_in_slo, test_in_slo, train_out_slo, test_out_slo = train_test_split(
            in_slo_reshaped, out_slo_reshaped, test_size=0.2, random_state=42)

        # Define the input shape
        input_shape = (in_slo.shape[2],)

        # Build the model
        inputs = Input(shape=input_shape)
        x = Dense(32, activation='relu')(inputs)
        outputs = Dense(11, activation='softmax')(x)
        model = Model(inputs=inputs, outputs=outputs)

        model.compile(optimizer=tf.keras.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Callbacks
        # chkpt_lrn_callback = tf.keras.callbacks.ModelCheckpoint(filepath = MODEL_TRAIN_PATH, save_weights_only = False, monitor = MODEL_TRAIN_MONITOR, 
        #                             mode = MODEL_MODE, save_best_only = True)
        # chkpt_val_callback = tf.keras.callbacks.ModelCheckpoint(filepath = MODEL_VALID_PATH, save_weights_only = False, monitor = MODEL_VALID_MONITOR,
        #                             mode = MODEL_MODE, save_best_only = True)
        hist_log_callback = tf.keras.callbacks.CSVLogger(filename = HISTORY_PATH, append = False)

        # Train the model
        history = model.fit(train_in_slo, train_out_slo, epochs=epoch, batch_size=batch, validation_data=(test_in_slo, test_out_slo), 
                    callbacks = [hist_log_callback])
                    # callbacks = [chkpt_lrn_callback, chkpt_val_callback, hist_log_callback])
        
        # Save the model
        model.save(os.path.join(MODELS_DIR, 'model.h5'))
        
        calculate()

    
def calculate():
    
    global model
    global train_in_slo, test_in_slo, train_out_slo, test_out_slo
    # Evaluate the model on the test set
    test_loss, test_acc = model.evaluate(test_in_slo, test_out_slo)
    print("Test Loss:", test_loss)
    print("Test Accuracy:", test_acc)

    # Evaluate the model on the training set
    train_loss, train_acc = model.evaluate(train_in_slo, train_out_slo)
    print("Train Loss:", train_loss)
    print("Train Accuracy:", train_acc)

    # Make predictions on the training set
    predicted_out_slo = model.predict(train_in_slo)
    predicted_out_slo = np.argmax(predicted_out_slo, axis=1)

    # Calculate confusion matrix
    cf_matrix = confusion_matrix(train_out_slo, predicted_out_slo)

    # Calculate F1-scores for each class
    f1_scores = f1_score(train_out_slo, predicted_out_slo, average=None)

    print('Confusion Matrix:')
    print(cf_matrix)
    print('F1-scores:')
    print(f1_scores)
