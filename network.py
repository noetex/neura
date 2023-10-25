# input_layer  -> (Values, Weights)
# hidden_layer -> (Weights, Biases, Activation Function)
# output_layer -> (Values, Biases)
import random
import math

def activation_sigmoid(X):
	return 1/(1 + math.exp(-X))

def random_list(Length):
	return [random.uniform(-1, 1) for _ in range(Length)]

class input_layer:
	def __init__(self, Inputs):
		self.Values  = Inputs
		self.Weights = random_list(len(Inputs))

class hidden_layer:
	def __init__(self, Size):
		self.Weights = random_list(Size)
		self.Biases  = random_list(Size)
		self.Activation = 0

class output_layer:
	def __init__(self, Outputs):
		self.Values = Outputs
		self.Biases = random_list(len(Outputs))

class neural_network:
	def __init__(self, Inputs, DesiredOutputs, NumHiddenLayers, NeuronsPerLayer):
		self.Layers  = [input_layer(Inputs)]
		self.Layers += [hidden_layer(NeuronsPerLayer) for _ in range(NumHiddenLayers)]
		self.Layers += [output_layer(DesiredOutputs)]

	def cost(self):
		pass

	def feed_forward(self):
		for Layer in self.Layers:
			if(isinstance(Layer, output_layer)): return
		pass

	def backpropagate(self):
		for Layer in self.Layers:
			if(isinstance(Layer, input_layer)): return
		pass

	def train(self, inputs, desired_outputs):
		pass

	def test(self, inputs):
		pass

	def __repr__(self):
		pass

	def __str__(self):
		pass
