# XOR Neural Network From Scratch

This project demonstrates how a basic feed-forward neural network can be implemented without using machine learning frameworks. Every step of the learning process, including forward propagation, backpropagation, and parameter updates, is written manually using Python and NumPy.

The model is trained on the XOR dataset, a classic machine learning problem that requires a hidden layer because it is not linearly separable.

## Overview

The implementation includes:

* Random weight and bias initialization
* Sigmoid activation function
* Forward propagation
* Error calculation
* Backpropagation
* Gradient descent optimization
* Prediction after training

## Network Architecture

* **Input Layer:** 2 neurons
* **Hidden Layer:** 2 neurons
* **Output Layer:** 1 neuron

## Training Settings

| Parameter           |   Value |
| ------------------- | ------: |
| Learning Rate       |    0.01 |
| Epochs              |    1000 |
| Hidden Neurons      |       2 |
| Activation Function | Sigmoid |

## Dataset

| Input  | Target |
| ------ | ------ |
| (0, 0) | 0      |
| (0, 1) | 1      |
| (1, 0) | 1      |
| (1, 1) | 0      |

## Requirements

```bash
pip install numpy
```

## Run

```bash
python neural_network.py
```

After training, the program prints the predicted output for each XOR input.

## Project Goals

This project was created to strengthen my understanding of the mathematical principles behind neural networks instead of relying on high-level deep learning libraries.

## Topics Covered

* Artificial Neural Networks
* Feed-Forward Networks
* Backpropagation
* Gradient Descent
* Sigmoid Activation
* Binary Classification
* NumPy
* Matrix Operations

## Possible Improvements

* Add support for configurable network sizes
* Implement ReLU and Tanh activation functions
* Plot training loss over time
* Add different optimization algorithms
* Support custom datasets
* Save and load trained models

## License

This project is intended for educational purposes and learning the fundamentals of neural networks through a complete implementation from scratch.
