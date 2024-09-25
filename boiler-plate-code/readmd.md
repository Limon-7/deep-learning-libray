# MNIST Image Classification using PyTorch

This project demonstrates the implementation of an image classification model using the MNIST dataset. The project is built with PyTorch and includes features such as custom validation, early stopping, and the ability to save and load checkpoints. The notebook focuses on training a deep neural network to classify handwritten digits with high accuracy.

## Table of Contents
- [MNIST Image Classification using PyTorch](#mnist-image-classification-using-pytorch)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
  - [Requirements](#requirements)
      - [The dependencies are listed in the requirements.txt file, and include:](#the-dependencies-are-listed-in-the-requirementstxt-file-and-include)
  - [Features](#features)
  - [Usage](#usage)
  - [Training Details](#training-details)
      - [Key components:](#key-components)
  - [Model Checkpointing](#model-checkpointing)
      - [Loading a Pretrained Model](#loading-a-pretrained-model)
  - [Results](#results)

## Project Overview
The objective of this project is to classify handwritten digits from the MNIST dataset. The model is trained using a neural network, and includes mechanisms for early stopping and dynamic model checkpointing, which helps save the best-performing model during training.

The notebook demonstrates key concepts like:
- Data loading and augmentation using `torchvision`
- Model training loop with validation
- Custom accuracy metrics
- Model checkpointing
- Loading models from checkpoints for further training or evaluation

## Requirements
To run this project, install the required packages by running the following command:

```bash
pip install -r requirements.txt
```
#### The dependencies are listed in the requirements.txt file, and include:

- Python 3.x
- PyTorch
- Torchvision
- Numpy
- Matplotlib
- Tqdm

## Features
1. Early Stopping: Stops training when validation performance stops improving.
2. Custom Validation: Validation is performed after each epoch with accuracy, precision, recall, and F1 score.
3. Checkpointing: Saves the best model based on validation loss.
4. Data Augmentation: Includes transformations like normalization.
5. Accuracy Calculation: A utility function to calculate accuracy during training and validation.

## Usage
1. Clone Repository: 
    ```bash
    git clone https://github.com/Limon-s-AI-Zone/Computer-Vision.git
    ```
2. Navigate to the project directory and open the Jupyter notebook:
    ```bash
    cd Computer-Vision/ImageClassification
    jupyter notebook image-classification.ipynb
    ```
3. Run the cells in the notebook to train the model, visualize results, and evaluate performance.

## Training Details
The training process involves iterating through batches of the MNIST dataset, calculating the loss using CrossEntropy, and optimizing the model using the Adam optimizer.

#### Key components:
1. Model Architecture: A neural network with convolutional and fully connected layers.
2. Optimizer: Adam optimizer is used for faster convergence.
3. Loss Function: CrossEntropy loss is used for multi-class classification.
4. Metrics: Accuracy, precision, recall, and F1 score are calculated for performance evaluation.

## Model Checkpointing
During training, the model is saved dynamically if the validation loss decreases. This ensures the best model is saved for future evaluation or resuming training.

#### Loading a Pretrained Model
If you want to load a saved model and continue training or evaluate, the following code snippet can be used:
```bash
model, optimizer, checkpoint, start_epoch, best_acc, valid_loss_min = load_model(filepath="path/to/checkpoint.pth.tar", model=model, optimizer=optimizer)
```

## Results
After training the model for several epochs, the model achieves an accuracy of around `95%` on the test set. Validation performance is monitored throughout the training, and the model with the lowest validation loss is saved as the best-performing model.