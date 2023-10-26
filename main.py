import random
from network import neural_network

train_dataset = [
	[0, 1, 2, 3, 4],
	[0, 2, 4, 6, 8],
]

def cost(weight):
	loss = 0
	for (the_input, output) in train_dataset:
		estimate = the_input * weight
		error = output - estimate
		loss += error * error
	result = loss / len(train_dataset)
	return result

epsilon = 0.01
loss_minimum = 0.000001
learning_rate = 0.01

def main():
	Network = neural_network(train_dataset[0], train_dataset[1], 2, 4)
	return
	Network.train()
	weight = random.uniform(0, 10)
	iterations = 0
	while(True):
		the_cost = cost(weight)
		print(f"Weight: {weight}, Loss: {the_cost}")
		if(cost(weight) < loss_minimum): break
		iterations += 1
		dw = (cost(weight + epsilon) - the_cost)/epsilon
		weight -= learning_rate * dw
	print(f"\niterations: {iterations}")

if(__name__ == "__main__"):
	main()
