# NanoGrad

**A simple and slow Autograd system from scratch, built for educational purposes.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

NanoGrad is a minimal automatic differentiation (autograd) engine implemented in Python. This project is designed to help you understand the core concepts behind deep learning frameworks like PyTorch or TensorFlow.


## Quick Start

Here is a basic example of how to use the `Value` class to build a computation graph and perform backpropagation.

```python
from nanograd import Value

# Build a simple expression: f(x, y) = x * y + tanh(x)
x = Value(2.0)
y = Value(3.0)
z = x * y
out = z + x.tanh()

# Backward pass
out.backward()

# Gradients are stored in the .grad attribute
print(f"Gradient of x: {x.grad}")  # d(out)/dx = y + (1 - tanh(x)^2) = 3 + (1 - 0.93) ≈ 3.07
print(f"Gradient of y: {y.grad}")  # d(out)/dy = x = 2.0
```

### Building a Simple Neuron

```python
from nanograd import Neuron
import numpy as np

# A neuron with 2 inputs
n = Neuron(2)

# Input values
x = [Value(1.0), Value(-2.0)]

# Forward pass
out = n.forward(x).tanh()
out.backward()

# Accessing gradients
for i, w in enumerate(n.weights):
    print(f"Gradient of weight {i}: {w.grad}")
print(f"Gradient of bias: {n.bias.grad}")
```

## How It Works

### The `Value` Class

The core of the autograd engine is the `Value` class. Each `Value` object holds:

- `data`: The scalar value.
- `grad`: The gradient of the final output with respect to this value.
- `childern`: A tuple of tuples, where each inner tuple contains a child `Value` and a lambda function representing the local gradient of the operation.

When you perform an operation like `a + b`, a new `Value` object is created. Its `childern` attribute stores references to `a` and `b`, along with their respective local gradient functions:

- For `a`: `lambda x: x + b.data` (derivative of `a + b` w.r.t `a` is 1, but stored as the partial derivative function)
- For `b`: `lambda x: a.data + x` (derivative of `a + b` w.r.t `b` is 1)

The `backward()` method performs a **reverse-mode automatic differentiation**. It starts from the root `Value` and recursively computes gradients using the chain rule. It uses a numerical approximation function (`gradformula`) to compute the local gradients from the stored lambda functions.

> **Why numerical gradients?** For educational clarity, this implementation uses numerical differentiation (finite differences) instead of analytical derivatives. This makes the code easier to read and understand, but it is computationally expensive and imprecise, hence the "slow" in the title.
