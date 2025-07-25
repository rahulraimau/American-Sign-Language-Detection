ASL Alphabet Image Classification
This project demonstrates the development of a Convolutional Neural Network (CNN) model to classify images of the American Sign Language (ASL) alphabet.

live demo:https://opulent-space-meme-jj5xgxx5rx625v9r-8501.app.github.dev/
https://american-sign-language-detection-4ylu.onrender.com/
Project Steps
Setup Kaggle Credentials and Download Dataset:

Configured Kaggle API credentials to download the ASL alphabet dataset.
Downloaded and unzipped the dataset to the Colab environment.
Data Loading and Preprocessing:

Loaded images from the dataset directories.
Resized images to 64x64 pixels and normalized pixel values.
Converted labels to one-hot encoded format.
Limited the number of images per class to 300 for faster processing during development (can be adjusted for the final model).
Data Splitting:

Split the dataset into training and testing sets (80% train, 20% test) using stratified sampling to maintain class distribution.
Model Architecture:

Built a Sequential CNN model using TensorFlow/Keras.
The model consists of convolutional layers, max pooling layers, a flatten layer, and dense layers with a dropout layer for regularization.
The final layer has 29 units with a softmax activation for classification into 29 classes (A-Z, del, nothing, space).
Model Compilation and Training:

Compiled the model with the Adam optimizer, categorical crossentropy loss, and accuracy as the evaluation metric.
Trained the model on the training data with a validation split of 20%.
Used Early Stopping to prevent overfitting by monitoring validation loss.
Model Evaluation:

Evaluated the trained model on the test set to determine its performance.
Calculated and printed the test accuracy.
Generated a classification report showing precision, recall, and F1-score for each class.
Plotted the training history (accuracy and loss) to visualize model performance during training.
Plotted a confusion matrix to visualize the model's predictions against the true labels.
Model Saving:

Saved the trained model to an HDF5 file (asl_classifier_model.h5) for future use.
Making Predictions on Test Images:

Loaded individual test images from the test folder.
Preprocessed each image (resize and normalize).
Used the trained model to predict the class of each test image.
Stored the original file names and the predicted labels.
Saving and Downloading Predictions:

Created a Pandas DataFrame with the image names and their corresponding predicted labels.
Saved the predictions to a CSV file (asl_test_predictions.csv).
Provided a link to download the predictions CSV file.
Results
The model achieved a test accuracy of 93.10%.
The classification report and confusion matrix provide a detailed breakdown of the model's performance across different ASL signs, highlighting classes where the model performed well and areas for potential improvement.
Future Work
Experiment with different CNN architectures.
Use data augmentation techniques to increase the size and diversity of the training data.
Fine-tune hyperparameters for better performance.
Explore transfer learning with pre-trained models.
