import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(output):
    return output * (1 - output)
learning_rate = 0.01
num_epochs = 1000
input_size = 2
hidden_size = 2
output_size = 1

W_ih = np.random.uniform(-1, 1, (input_size, hidden_size))
b_h = np.random.uniform(-1, 1, hidden_size)
W_ho = np.random.uniform(-1, 1, (hidden_size, output_size))
b_o = np.random.uniform(-1, 1, output_size)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])
for epoch in range(num_epochs):
    for i in range(len(X)):

        input_layer = X[i]
        hidden_layer_input = np.dot(input_layer, W_ih) + b_h
        hidden_layer_output = sigmoid(hidden_layer_input)
        output_layer_input = np.dot(hidden_layer_output, W_ho) + b_o
        predicted_output = sigmoid(output_layer_input)
        error = Y[i] - predicted_output
        delta_output = error * sigmoid_derivative(predicted_output)
        error_hidden_layer = delta_output.dot(W_ho.T)
        delta_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)
        W_ho += hidden_layer_output.reshape(-1, 1).dot(delta_output.reshape(1, -1)) * learning_rate
        b_o += delta_output * learning_rate
        W_ih += input_layer.reshape(-1, 1).dot(delta_hidden_layer.reshape(1, -1)) * learning_rate
        b_h += delta_hidden_layer * learning_rate

print("Final predicted outputs:")
for x in X:
    h_input = sigmoid(np.dot(x, W_ih) + b_h)
    o_input = sigmoid(np.dot(h_input, W_ho) + b_o)
    print(f"Input: {x}, Predicted Output: {o_input}")
