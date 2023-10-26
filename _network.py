import random

# maybe this is a stupid idea
class BasicLayer:
  def __init__(self, size):
    self.activations = [0] * size

class InputLayer(BasicLayer):
  def __init__(self, size):
    super().__init__(size)
    self.weights = [random.uniform(-1.0, 1.0) for i in range(size)]

class HiddenLayer(BasicLayer):
  def __init__(self, size):
    super().__init__(size)
    self.weights = [random.uniform(-1.0, 1.0) for i in range(size)]
    self.biases = [random.uniform(-1.0, 1.0) for i in range(size)]

class OutputLayer(BasicLayer):
  def __init__(self, size):
    super().__init__(size)
    self.biases = [random.uniform(-1.0, 1.0) for i in range(size)]

class Network:
  def __init__(self, layers_sizes):
    self.Layers = []
    self.Layers.append(InputLayer(layers_sizes[0]))
    for i in range(1, len(layers_sizes) - 1):
      self.Layers.append(HiddenLayer(layers_sizes[i]))
    self.Layers.append(OutputLayer(layers_sizes[-1]))

  def feed_forward(self, L = 0):
    current_layer = self.Layers[L]
    if current_layer is self.Layers[-1]:
      return  # we're done

    aw = 0
    for i in range(len(current_layer.activations)):
      aw += current_layer.activations[i] * current_layer.weights[i]

    next_layer = self.Layers[L+1]
    for i in range(len(next_layer.activations)):
      next_layer.activations[i] = sigmoid(aw + next_layer.biases[i])

    self.feed_forward(L+1)

  def prop_backwards(self, y, L=-1):
    current_layer = self.Layers[L]
    if current_layer is self.Layers[0]:
      return

    prev_layer = self.Layers[L-1]
    a = current_layer.activations

    # never EVER sleep in calculus classes
    dC_da = sum(2*(a[i] - y) for i in range(len(a)))/len(a)  # C = sum((a[i] - y[i])**2)/n  -->  dC/da = sum(2*(a[i] - y[i]))/n
    dz_dw = sum(prev_layer.activations)  # z = sum(a[i] * w[i]) + b  -->  dz/dw = sum(a[i])

    # this term is only here for completeness' sake
    dz_db = 1  # z = sum(a[i] * w[i]) + b  -->  dz/db = 1

    for i in range(len(a)):
      da_dz = sigmoid_prime(a[i])
      # chain rule
      dC_dz = dC_da * da_dz
      dC_dw = dC_dz * dz_dw
      dC_db = dC_dz * dz_db
      # record nudges
      print(L)
      self.delta_w[L][i] += dC_dw
      self.delta_b[L][i] += dC_db

      self.prop_backwards(a[i] - dC_da, L-1)

  def train(self, data, batch_size, learning_rate, num_epochs, tests = None):
    for i in range(num_epochs):
      random.shuffle(data)
      # yada yada... slice training data into batches and feed inputs of each
      batches = [data[k : k + batch_size] for k in range(0, len(data), batch_size)]
      average_cost = 0
      for batch in batches:
        self.delta_w = [[0] * len(self.Layers[i].weights) for i in range(len(self.Layers) - 1)]
        self.delta_b = [[0] * len(self.Layers[i].biases) for i in range(1, len(self.Layers))]
        for (x, expected_output) in batch:
          for i in range(len(self.Layers[0].activations)):
            self.Layers[0].activations[i] = x[i]
          self.feed_forward()
          a = self.Layers[-1].activations  # output layer activations
          y = [0] * len(a)
          y[expected_output] = 1.0
          for i in range(len(a)):
            average_cost += (a[i] - y[i])**2  # mean square error
          average_cost /= len(a)
          self.prop_backwards(expected_output)  # from the last layer

        eta = learning_rate / len(batch)
        # remember: input layers don't have biases, output layers don't have weights
        for i in range(len(self.Layers) - 1):
          for j in range(len(self.Layers[i].activations)):
            self.Layers[i].weights[j] -= (delta_w[i][j]) * eta
          for j in range(len(self.Layers[i+1].biases)):
            self.Layers[i+1].biases[j] -= (delta_b[i][j]) * eta

      del self.delta_w
      del self.delta_b

      # at this point, the network has seen all the data, let's test!
      correct_guesses = 0
      if tests is not None:
        num_tests = len(tests)
        for i in range(len(test)):
          for j in range(len(self.Layers[0].activations)):
            self.Layers[0].activations[i] = test[i].inp
          self.feed_forward()
          if max(self.Layers[-1].activations) == test[i].out:
            correct_guesses += 1
        print("epoch: %d/%d\tcorrect: %d/%d\tAverage cost: %f" % (i, num_epochs, correct_guesses, num_tests, average_cost))

from math import exp

def sigmoid(z):
  return 1/(1+exp(-z))

# sigmoid derivative, assumes the input is already a sigmoid
def sigmoid_prime(a):
  return a * (1 - a)