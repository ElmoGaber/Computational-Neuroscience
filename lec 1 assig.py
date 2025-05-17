import math
import random
def tanh(x):
    return math.tanh(x)
def initialize_weights():
    return {
        "w1": random.uniform(-0.5, 0.5), "w2": random.uniform(-0.5, 0.5),
        "w3": random.uniform(-0.5, 0.5), "w4": random.uniform(-0.5, 0.5),
        "w5": random.uniform(-0.5, 0.5), "w6": random.uniform(-0.5, 0.5),
        "w7": random.uniform(-0.5, 0.5), "w8": random.uniform(-0.5, 0.5)
    }
def forward_propagation(i1, i2, b1, b2, weights):
    h1_input = i1 * weights["w1"] + i2 * weights["w3"] + b1
    h2_input = i1 * weights["w2"] + i2 * weights["w4"] + b1
    h1_output = tanh(h1_input)
    h2_output = tanh(h2_input)
    o1_input = h1_output * weights["w5"] + h2_output * weights["w7"] + b2
    o2_input = h1_output * weights["w6"] + h2_output * weights["w8"] + b2
    o1_output = tanh(o1_input)
    o2_output = tanh(o2_input)
    return h1_output, h2_output, o1_output, o2_output
i1, i2 = 0.05, 0.10
b1, b2 = 0.35, 0.60
weights = initialize_weights()
h1, h2, o1, o2 = forward_propagation(i1, i2, b1, b2, weights)
print("Hidden Layer Outputs:")
print(f"h1 = {h1:.4f}, h2 = {h2:.4f}")
print("Output Layer Outputs:")
print(f"o1 = {o1:.4f}, o2 = {o2:.4f}")
