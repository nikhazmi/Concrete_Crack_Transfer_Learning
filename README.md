# Concrete_Crack_Transfer_Learning
Image detection using transfer learning

## Badges

![Windows 11](https://img.shields.io/badge/Windows%2011-%230079d5.svg?style=for-the-badge&logo=Windows%2011&logoColor=white)
![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white)
![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)

## Explanation
The code provided is a Python script that demonstrates how to use transfer learning to train a deep learning model for image classification 
using the Tensorflow library. The code is divided into multiple sections, each of which is executed using the Jupyter Notebook's "cell" execution feature.

The script is written for classifying concrete crack images, it's divided into several sections (cells) as following:

## Documentation
DATA DOCUMENTATION (Explanation for each number section)

       1. Importing the required libraries such as numpy, pandas, tensorflow, pathlib, and matplotlib.
       2. Defining the file path to the dataset location.
       3. Using the tf.keras.utils.image_dataset_from_directory to load the dataset, shuffle it and split the data into training and validation sets.
       4. Print the class names.
       5. Further split the validation dataset into validation-test split.
       6. Convert the BatchDataset into PrefectDataset using the tf.data.experimental.cardinality and tf.data.AUTOTUNE .
       7. Create a small pipeline for data augmentation using RandomFlip and RandomRotation layers from keras.
       8. Create a preprocessing layer using applications.mobilenet_v2.preprocess_input.
       9. Apply Transfer learning using MobileNetV2 pre-trained model with the imagenet weights and disabling the training for the feature extractor.
       10. Create the classification layers, where the last layers output is the number of classes and activation as softmax
       11. Create a model and adding the feature_extractor, global_AVG and output_layer to the model.
       12. Compile the model using the Adam optimizer, categorical_crossentropy loss, and accuracy metric.
       13. Fit the model using the fit_generator method which is using the train_datagen.flow function to provide the data.


This script provides a good starting point for using transfer learning to train a deep learning model for image classification, 
but it should be noted that the parameters and architectures used in this script may not be optimal for all datasets and use cases. 
And depending on the size and complexity of your dataset and the power of your machine, you may need to adjust the parameters accordingly to avoid 
overfitting or long training times.
