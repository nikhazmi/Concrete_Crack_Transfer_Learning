#%%
import numpy as np
import pandas as pd
import tensorflow as tf
import datetime
import os 
import pathlib
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras.utils import plot_model
from tensorflow.keras import layers, optimizers, losses, callbacks, applications

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


#%%
"""
    DATA DOCUMENTATION (Explanation for each number section)

       1. Data Loading: This step is responsible for loading the image dataset from the specified directory. 
           The dataset used in this code is the "Concrete Crack Images for Classification" dataset.
       2. Defining the file path to the dataset: The code sets the file path variable to the location of the dataset on the local machine.
       3. Prepared the data: This step is responsible for preparing the dataset for use in training the model. 
           The code sets a seed value for reproducibility, and the image size is set to (160,160). 
           The train and validation datasets are created using the 'tf.keras.utils.image_dataset_from_directory' function.
       4. Create class names to display some images as examples: The class names for the dataset are created and used to 
          display a sample of images from the training dataset.
       5. Further split the validation dataset into validation-test split: This step splits the validation dataset into a 
          validation set and a test set. The validation set is used to evaluate the model during training, 
          and the test set is used to evaluate the final model.
       6. Convert the BatchDataset into PrefectDataset: This step converts the BatchDataset into a PrefectDataset, 
          which is a dataset optimized for performance.
       7. Create a small pipeline for data augmentation: This step creates a pipeline for data augmentation, 
          which is used to artificially expand the dataset and improve the model's robustness.
              7.1 Apply the data augmentaion to test it out: This step applies the data augmentation pipeline to a test image to check its effect.
                  Prepared the layer for preprocessing: This step prepares the layer for preprocessing input images before they are fed into the model.
       8. Prepared the layer for preprocessing: This step prepares the layer for preprocessing input images before they are fed into the model.
       9. Apply transfer learning: This step involves using a pre-trained model, such as MobileNetV2, to extract features from the input images, and using these                 
       features as input to a new model that will be trained to classify the images.
              9.1 Disable the training for the feature extractor (freeze the layers): To improve the performance of the new model, 
              it is important to prevent the pre-trained layers from being updated during training. This step is done by setting the trainable attribute of the pre-                 
              trained model to False.
       10. Create the classification layers: A new set of layers is added to the model that will be used to classify the images based on the features 
           extracted by the pre-trained model. This step typically involves adding a GlobalAveragePooling2D layer and a Dense layer with a softmax activation function.
       11. Use functional API to link all the modules together: The functional API is used to create the final model by linking the pre-trained model and the new                  
       classification layers together. The pre-trained model's output is used as input to the classification layers.
       12. Compile the model and train: The model is compiled by defining the loss function, optimizer, and evaluation metric. Then, the model is trained by fitting              
       the training data to it.
              12.1 Evaluate the model before model training: The model's performance is evaluated on the validation set before the training process.
              12.2 Callback funtion: Additional callbacks can be defined, such as saving the model after each epoch or early stopping the training if there is no                          
              improvement in the validation set.
              12.3 Perform model training: The training is performed by fitting the data to the model.
              12.4 Plot Training, Validation Accuracy, Validation Loss: The training and validation accuracy and loss are plotted over the training process to                            
              visualize the model's performance during training.
       13. Apply the next transfer learning strategy: The next strategy is applied by defining new configurations to the model such as unfreezing some layers, or
           adding more layers.
       14. Freeze the earlier layers: Freezing the earlier layers means making them non-trainable to preserve their learned weights.
              14.1 Compile the model: The model is recompiled with the new configurations.
       15. Continue the training with this new set of configuration: The training is continued with the new configurations.
              15.1 Follow up from the previous model training: The training is resumed from the last checkpoint.
       16. Evaluate the final model: The final model's performance is evaluated using the test dataset, which was set aside earlier.
              16.1 Evaluating the model on the test dataset: The model is evaluated using the test dataset in order to get a sense of its generalization performance.
              16.2 Predict: The model is used to predict the class of new images by calling the predict function.
              16.3 Label VS Predict: The true labels of the images are compared with the predicted labels to measure the accuracy of the predictions.
              16.4 Plot Training, Validation Accuracy, Validation Loss: The accuracy and loss during training and validation is plotted in order to see how well the                      
              model was trained and its performance on the validation dataset.
       17. Show some predictions: A few predictions are made using the model and the images are displayed alongside their predicted labels.
       18. Model analysis using classfication report and confusion matrix
              18.1 Display the report
       19. Model Save: The final model is saved to a file so that it can be reused in the future without the need to retrain the model.
       """

#%%
#1. Data Loading
file_path = r"C:\Users\Nik Hazmi\Desktop\Concrete_Crack_Images\Datasets\Concrete Crack Images for Classification"

#2. Defining the file path to the dataset
data_dir = pathlib.Path(file_path)

#%%
#3. Prepared the data
SEED = 32
IMG_SIZE = (160,160)
train_dataset = tf.keras.utils.image_dataset_from_directory(data_dir, shuffle = True, validation_split = 0.2, subset="training", seed = SEED, image_size = IMG_SIZE , batch_size = 10)
val_dataset = tf.keras.utils.image_dataset_from_directory(data_dir, shuffle = True, validation_split = 0.2, subset="validation", seed = SEED, image_size = IMG_SIZE , batch_size = 10)

# %%
#4. Create class names to display some images as examples
class_names = train_dataset.class_names

plt.figure(figsize = (10, 10))
for images, labels in train_dataset.take(1):
    for i in range(9):
        plt.subplot(3,3, i+1)
        plt.imshow(images[i].numpy().astype('uint8'))
        plt.title(class_names[labels[i]])
        plt.axis('off')

# %%
#5. Further split the validation dataset into validation-test split
val_batches = tf.data.experimental.cardinality(val_dataset)
test_dataset = val_dataset.take(val_batches//5)
validation_dataset = val_dataset.skip(val_batches//5)

# %%
#6. Convert the BatchDataset into PrefectDataset
AUTOTUNE = tf.data.AUTOTUNE
pf_train = train_dataset.prefetch(buffer_size = AUTOTUNE)
pf_val = validation_dataset.prefetch(buffer_size = AUTOTUNE)
pf_test = test_dataset.prefetch(buffer_size = AUTOTUNE)

# %%
#7. Create a small pipeline for data augmentation
data_augmentation = keras.Sequential()
data_augmentation.add(layers.RandomFlip('horizontal'))
data_augmentation.add(layers.RandomRotation(0.2))

# %%
#7.1 Apply the data augmentaion to test it out
for images, labels in pf_train.take(1):
    first_images = images[0]
    plt.figure(figsize = (10, 10))
    for i in range(9):
        plt.subplot(3,3, i+1)
        augmented_image = data_augmentation(tf.expand_dims(first_images, axis = 0))
        plt.imshow(augmented_image[0]/255.0)
        plt.axis('off')

# %%
#8. Prepared the layer for preprocessing
preprocess_input = applications.mobilenet_v2.preprocess_input

#9. Apply transfer learning
IMG_SHAPE = IMG_SIZE + (3,)
feature_extractor = applications.MobileNetV2(input_shape = IMG_SHAPE, include_top = False, weights = 'imagenet')

#9.1 Disable the training for the feature extractor (freeze the layers)
feature_extractor.trainable = False
feature_extractor.summary()
keras.utils.plot_model(feature_extractor, show_shapes = True)

# %%
#10. Create the classification layers
global_AVG = layers.GlobalAveragePooling2D()
output_layer = layers.Dense(len(class_names), activation = 'softmax')

# %%
#11. Use functional API to link all the modules together
inputs = keras.Input(shape = IMG_SHAPE)
x = data_augmentation(inputs)
x = preprocess_input(x)
x = feature_extractor(x)
x = global_AVG(x)
x = layers.Dropout(0.3)(x)
outputs = output_layer(x)

model = keras.Model(inputs = inputs, outputs = outputs)
model.summary()

# %%
#12. Compile the model and train
optimizer = optimizers.Adam(learning_rate = 0.0001)
loss = losses.SparseCategoricalCrossentropy()
model.compile(optimizer = optimizer, loss= loss, metrics = ['accuracy'])
plot_model(model,show_shapes=True,show_layer_names=True)

#%%
#12.1 Evaluate the model before model training
loss0, accuracy0 = model.evaluate(pf_val)
print('loss =', loss0)
print('acc =', accuracy0)

# %%
#12.2 Callback funtion 
log_path = os.path.join('log_dir', 'tl', datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
tb = callbacks.TensorBoard(log_dir = log_path)

#%% 
#12.3 Perform model training
EPOCHS = 10
history = model.fit(pf_train, validation_data = pf_val, epochs = EPOCHS, callbacks=[tb])

#%% 
#12.4 Plot Training, Validation Accuracy, Validation Loss 
train_loss = history.history["loss"]
val_loss = history.history["val_loss"]
train_acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
epochs = history.epoch

plt.plot(epochs, train_loss, label="Training loss")
plt.plot(epochs, val_loss, label="Validation loss")
plt.title("Training vs Validation loss")
plt.legend()
plt.figure()

plt.plot(epochs, train_acc, label="Training accuracy")
plt.plot(epochs, val_acc, label="Validation accuracy")
plt.title("Training vs Validation accuracy")
plt.legend()
plt.figure()

plt.show()
#%%
#13. Apply the next transfer learning strategy
feature_extractor.trainable = True

#14. Freeze the earlier layers
for layer in feature_extractor.layers[:100]:
    layer.trainable = False

feature_extractor.summary()

# %%
#14.1 Compile the model
optimizer = optimizers.RMSprop(learning_rate = 0.00001)
plot_model(model,show_shapes=True,show_layer_names=True)
model.compile(optimizer = optimizer, loss = loss, metrics= ['accuracy'])

# %%
#15. Continue the training with this new set of configuration
fine_tune_epoch = 10
total_epoch = fine_tune_epoch + EPOCHS

#15.1 Follow up from the previous model training
history_fine = model.fit(pf_train, validation_data= pf_val, epochs = total_epoch, initial_epoch = history.epoch[-1],callbacks= [tb])

# %%
#16. evaluate the final model
#16.1 Evaluating the model on the test dataset.
test_loss, test_acc = model.evaluate(pf_test)
print('loss =', test_loss)
print('acc = ', test_acc)

# %%
#16.2 Predict
image_batch, label_batch, = pf_test.as_numpy_iterator().next()
predictions = np.argmax(model.predict(image_batch), axis = 1)

#%%
#16.3 Label VS Predict
label_vs_prediction = np.transpose(np.vstack((label_batch, predictions)))

#%%
#16.4 Plot Training, Validation Accuracy, Validation Loss
train_loss = history.history["loss"]
val_loss = history.history["val_loss"]
train_acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
epochs = history.epoch

plt.plot(epochs, train_loss, label="Training loss")
plt.plot(epochs, val_loss, label="Validation loss")
plt.title("Training vs Validation loss")
plt.legend()
plt.figure()

plt.plot(epochs, train_acc, label="Training accuracy")
plt.plot(epochs, val_acc, label="Validation accuracy")
plt.title("Training vs Validation accuracy")
plt.legend()
plt.figure()

plt.show()

# %%
#17. Show some predictions
plt.figure(figsize=(10,10))

for i in range(9):
    axs = plt.subplot(3,3,i+1)
    plt.imshow(image_batch[i].astype("uint8"))
    plt.title(class_names[predictions[i]])
    plt.axis("off")

#%%
#18. Model Analysis
print(classification_report(label_batch, predictions))
cm = confusion_matrix(label_batch, predictions)

#%%
#18.1 Display the reports
disp= ConfusionMatrixDisplay(cm)
disp.plot()

# %%
#19. Model Save
model.save('Models\model.h5')

# %%
