import math
import numpy as np


class Value:
  def __init__(self , data , childern=() , grad=0.0):
    self.data = data
    self.childern = childern
    self.grad = 0.0

  def __repr__(self):
    return f"Value(data={self.data})"

  def __add__(self , other):
    if isinstance(other, Value): Value(other)
    
    return Value(self.data + other.data , childern=((self , lambda x: x + other.data ) , (other , lambda x: self.data + x)))

  def __sub__(self , other):
    if isinstance(other, Value): Value(other)

    return Value(self.data - other.data , childern=((self , lambda x: x - other.data ) , (other , lambda x: self.data - x)))

  def __mul__(self , other):
    if isinstance(other, Value): Value(other)

    return Value(self.data * other.data , childern=((self , lambda x: x * other.data ) , (other , lambda x: x * self.data)))

  def __pow__(self, other):
    if isinstance(other, Value): Value(other)

    return Value(self.data ** other, childern=((self, lambda x: x ** other),))

  def __truediv__(self , other):
    if isinstance(other, Value): Value(other)
    return Value(self.data / other.data , childern=((self , lambda x: x / other.data ) , (other , lambda x: self.data / x)))

  def __neg__(self):
    if isinstance(other, Value): Value(other)

    return Value(-1 * self.data , childern=((self , lambda x: -1 * x ),))

  def tanh(self):
    out_data = math.tanh(self.data)

    return Value(out_data, childern=((self, lambda x : math.tanh(x)),))

  def relu(self):
    out_data = max(0 , self.data)
    return Value(out_data, childern=((self, lambda x : max(0 , x)),))

  def gradformula(self , func , val , h=0.0001):
    return (func(val + h) - func(val)) / h


  def backward(self , prev_grad=1.0):
    self.grad += prev_grad

    for child , grad_fn in self.childern:
      grad = self.gradformula(grad_fn , child.data) * self.grad
      child.backward(grad)


    return

  def zerograd(self):
    self.grad = 0.0
    for child , _ in self.childern:
      child.zerograd()


class Neuron:
  def __init__(self , num_inputs):
    self.weights = [Value(np.random.randn()) for _ in range(num_inputs)]
    self.bias = Value(np.random.randn())

  def forward(self , x):
    return sum((wi * xi for wi , xi in zip(self.weights , x)) , self.bias)
