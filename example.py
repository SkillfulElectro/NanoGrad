from nanograd import *
import math
import numpy as np
import matplotlib.pyplot as plt

xs = [Value(i) for i in np.linspace(-1, 1, 100)]
ys = [Value(i.data**2) for i in xs]

#MLP
hidden_layer = [Neuron(1) for _ in range(16)]

output_neuron = Neuron(16)

def forward(x):
    hidden_outs = [n.forward([x]).tanh() for n in hidden_layer]
    
    output = output_neuron.forward(hidden_outs)
    return output


learning_rate = 0.001 # Increased learning rate for faster overfitting


all_neurons = hidden_layer + [output_neuron]

for epoch in range(1000):
    total_loss = 0
    for i, j in zip(xs, ys):
        # Forward pass
        pred = forward(i)
        loss = (pred - j)**2
        total_loss += loss.data

        loss.zerograd()
        loss.backward()


        # Update parameters
        for n in all_neurons:
            for w in n.weights:
                w.data -= learning_rate * w.grad
            n.bias.data -= learning_rate * n.bias.grad

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss/len(xs):.6f}")


test_xs = np.linspace(-4, 4, 100)
predictions = [forward(Value(x)).data for x in test_xs]
actuals = [x**2 for x in test_xs]

plt.plot(test_xs, predictions, label='Model Prediction')
plt.plot(test_xs, actuals, label='Actual x^2', linestyle='--')
plt.axvspan(-1, 1, alpha=0.2, color='green', label='Training Range')
plt.legend()
plt.title("Model Fit: Inside vs Outside Training Range")
plt.show()
