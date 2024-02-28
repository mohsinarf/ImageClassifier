# ImageClassifier

This repository contains code for training and evaluating a Convolutional Neural Network (CNN) on the Fashion MNIST dataset using PyTorch.

## Introduction
This codebase demonstrates the following:
- Loading Fashion MNIST dataset using PyTorch's `torchvision.datasets`
- Creating train and test dataloaders using `torch.utils.data.DataLoader`
- Building a CNN architecture for image classification
- Training the model using Stochastic Gradient Descent (SGD) optimizer
- Evaluating the model's performance on the test dataset

## Requirements
- Python 3.x
- PyTorch
- torchvision
- matplotlib
- tqdm

## Usage
1. Clone the repository:
    ```
    git clone https://github.com/mohsinarf/ImageClassifier.git
    ```
2. Install the required dependencies:
    ```
    pip install -r requirements.txt
    ```
3. Run the code:
    ```
    python App.py
    ```

## Dataset
Fashion MNIST is a dataset of Zalando article images, consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes.

## Model Architecture
The model architecture is based on TinyVGG, consisting of two convolutional blocks followed by a fully connected classifier. Each convolutional block consists of two convolutional layers followed by ReLU activation and max pooling.

## Results
The model achieves an accuracy of approximately 90% on the test dataset after XX epochs of training.

## License
This project is licensed under the MIT License.

