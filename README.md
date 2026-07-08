# Neural Network From Scratch

This project is a simple implementation of a feed-forward neural network built from scratch using only **Python** and **NumPy**. The main goal is to understand how neural networks work internally by implementing every step manually instead of relying on machine learning frameworks.

The network is trained on the XOR dataset, a classic example that cannot be solved using a single-layer perceptron. By adding a hidden layer and training with backpropagation, the model learns the correct mapping between the inputs and outputs.

## Features

* Built completely from scratch
* Uses only Python and NumPy
* Implements forward propagation manually
* Implements backpropagation manually
* Uses gradient descent for training
* Sigmoid activation function
* Learns the XOR problem without external ML libraries

## Project Structure

```text
.
├── neural_network.py
├── README.md
└── requirements.txt
```

## How It Works

The network consists of:

* Input layer with 2 neurons
* Hidden layer with 2 neurons
* Output layer with 1 neuron

During training, the model performs the following steps:

1. Initialize weights and biases randomly.
2. Perform forward propagation.
3. Calculate the prediction error.
4. Compute gradients using backpropagation.
5. Update weights and biases.
6. Repeat for multiple epochs.

## Dataset

The network is trained on the XOR truth table.

| Input | Output |
| ----- | ------ |
| 0, 0  | 0      |
| 0, 1  | 1      |
| 1, 0  | 1      |
| 1, 1  | 0      |

## Requirements

* Python 3.x
* NumPy

Install the dependency:

```bash
pip install numpy
```

## Run

```bash
python neural_network.py
```

After training, the model prints the predicted output for each input in the XOR dataset.

## What I Learned

This project helped me understand:

* How neurons process input data
* Matrix operations in neural networks
* Forward propagation
* Backpropagation
* Gradient descent
* Weight and bias updates
* Why hidden layers are necessary for non-linear problems like XOR

## Future Improvements

Some ideas for extending this project:

* Add support for multiple hidden layers
* Implement different activation functions such as ReLU and Tanh
* Track and visualize training loss
* Make the network configurable for different datasets
* Add support for mini-batch training
* Save and load trained weights

## Notes

The project is intended for educational purposes and focuses on understanding the core concepts behind neural networks rather than building a production-ready machine learning framework.
