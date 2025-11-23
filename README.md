# **Multiclass Classification of Chest X-ray Images Using Convolutional Neural Networks**

## **1. Project Title**
Multiclass Classification of Chest X-ray Images Using Convolutional Neural Networks

## **2. Project Overview**
This project builds a VGG16-inspired CNN to classify chest X-ray images into Normal Lung, Viral Pneumonia, and Lung Opacity. It aims to assist medical practitioners by providing faster, automated lung disease screening.
The dataset, collected from Mendeley Data (2025) and Kaggle, contains 3,475 images across the three classes. All images were preprocessed and organized into train, validation, and test sets. The project also includes a systematic hyperparameter comparison to evaluate how different configurations affect model accuracy and stability.

## **3. Features**
This project includes several useful functionalities:
- Automated disease classification from chest X-ray images.
- Model training with hyperparameter tuning options.
- Visualization support for accuracy curves, loss curves, and confusion matrices.
- Model checkpoint saving to preserve the best weights.
- Support for image preprocessing and real-time prediction.
- Easily expandable to new disease categories or architectures.

## **4. Installation Instructions**
The project requires Python and several deep-learning libraries such as TensorFlow, Keras, NumPy, Pandas, Matplotlib, and scikit-learn.  
All required packages are listed inside a requirements.txt file.

Install dependencies using the command:

```bash
pip install -r requirements.txt

## **5. Dataset Details**
- The dataset contains chest X-ray images from three categories: Normal, Viral Pneumonia, and Lung Opacity.
- It is obtained from Mendeley Data.
- The dataset is organized into separate folders for training, validation, and testing.
- Each folder contains three subfolders — one for each class.
Dataset link: *(Add link here)*

## **6. How to Run the Project**
To train the model, run the training script.  
It loads the dataset, preprocesses the images, trains the VGG16-style CNN, and saves the best model checkpoint.

You may also run the prediction script by providing a chest X-ray image and receiving a predicted disease label with probability.


## **7. Model Details**
### Model Architecture
A VGG16-inspired CNN designed with:
- 5 Convolutional Blocks using 3×3 kernels  
- Filter sizes: 64 → 128 → 256 → 512 → 512  
- MaxPooling after each block  
- Fully Connected Layers:
  - Dense(4096) → Dropout(0.5)
  - Dense(4096) → Dropout(0.5)
  - Dense(3) with Softmax output

### Hyperparameter Comparison
- Different learning rates (0.1, 0.000001)
- Batch sizes (8, 16)
- Early stopping patience (2, 5)
- Input image sizes (128×128, 256×256)
- Dropout rates (0.2, 0.7)
- Optimizers (RMSprop, SGD with momentum)
- Padding (Same, Valid)
- Stride (3, 2)

## **8. Results**
- The model achieves around 98% accuracy on validation and testing.
- Accuracy and loss curves show smooth and stable learning.
- Confusion matrix indicates strong classification across all three classes.
- Additional visualizations such as curves, charts, and sample predictions may be included for support.

## **9. Contributors**
Sumaiya Khanam  
Email: sumaiya.khanam01@gmail.com

Sirazum Munira  
Email: 1133sirazum.munira@gmail.com

Riya Saha  
Email: riyasaha1124@gmail.com

## **10. Demo Section**


