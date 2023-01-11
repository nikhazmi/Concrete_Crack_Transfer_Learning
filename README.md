# Concrete_Crack_Transfer_Learning
Welcome to the Concrete_Crack_Transfer_Learning project! This project is a demonstration of how to use transfer learning to train a deep learning model for image classification using the Tensorflow library. The code is designed to classify images of concrete cracks, providing a binary classification between cracked and not cracked images.

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
       9. Apply transfer learning: This step involves using a pre-trained model, such as MobileNetV2, to extract features from the input images, and using these                 features as input to a new model that will be trained to classify the images.
              9.1 Disable the training for the feature extractor (freeze the layers): To improve the performance of the new model, 
              it is important to prevent the pre-trained layers from being updated during training. This step is done by setting the trainable attribute of the pre-                 trained model to False.
       10. Create the classification layers: A new set of layers is added to the model that will be used to classify the images based on the features 
           extracted by the pre-trained model. This step typically involves adding a GlobalAveragePooling2D layer and a Dense layer with a softmax activation function.
       11. Use functional API to link all the modules together: The functional API is used to create the final model by linking the pre-trained model and the new                  classification layers together. The pre-trained model's output is used as input to the classification layers.
       12. Compile the model and train: The model is compiled by defining the loss function, optimizer, and evaluation metric. Then, the model is trained by fitting              the training data to it.
              12.1 Evaluate the model before model training: The model's performance is evaluated on the validation set before the training process.
              12.2 Callback funtion: Additional callbacks can be defined, such as saving the model after each epoch or early stopping the training if there is no                          improvement in the validation set.
              12.3 Perform model training: The training is performed by fitting the data to the model.
              12.4 Plot Training, Validation Accuracy, Validation Loss: The training and validation accuracy and loss are plotted over the training process to                            visualize the model's performance during training.
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
              16.4 Plot Training, Validation Accuracy, Validation Loss: The accuracy and loss during training and validation is plotted in order to see how well the                      model was trained and its performance on the validation dataset.
       17. Show some predictions: A few predictions are made using the model and the images are displayed alongside their predicted labels.
       18. Model Save: The final model is saved to a file so that it can be reused in the future without the need to retrain the model.

This script provides a good starting point for using transfer learning to train a deep learning model for image classification, 
but it should be noted that the parameters and architectures used in this script may not be optimal for all datasets and use cases. 
And depending on the size and complexity of your dataset and the power of your machine, you may need to adjust the parameters accordingly to avoid 
overfitting or long training times.
## Graph
![Training vs Validation](https://user-images.githubusercontent.com/82282919/211732878-3d2ad857-62ee-496f-9fe9-80eb83ffde81.png)

## Architecture of the model
![Architecture 1](https://user-images.githubusercontent.com/82282919/211738936-7a80d1e7-61aa-4354-a3db-b12ebaf5d202.png)

## Acknowledgment 
We would like to acknowledge the use of the dataset provided by Mendeley Data in this project. The dataset contains a set of images of concrete cracks for classification. The availability of this dataset greatly helped in the development and evaluation of our image classification model. We express our gratitude to the creators of this dataset for making it publicly available and hope that it continues to be a valuable resource for the research community.

https://data.mendeley.com/datasets/5y9wdsg2zt/2
