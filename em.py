import numpy as np
import matplotlib.pyplot as plt
import random as rand


size = 500
data = np.random.randint(size, size=size)
weights = np.ones(size)
newweights = np.random.randint(2, size=size)

## Functions
def w_avg(data, weights):
	return sum(data[i] * weights[i] for i in range(size)) / float(sum(weights))

def w_std(data, weights):
	return np.sqrt(w_avg((data-w_avg(data, weights))**2, weights))

def prob(x, mean, std):
	return np.exp( -((x-mean)/std) ** 2 / 2) / (std*np.sqrt(2*np.pi))

plt.scatter(data, np.ones(size), c=newweights, s=40)
plt.show()

while (not np.array_equal(weights, newweights)):
	weights = newweights
	mean1 = w_avg(data, weights)
	std1 = w_std(data, weights)

	mean2 = w_avg(data, 1-weights)
	std2 = w_std(data, 1-weights)


	for x in range(0, len(data)):
		if prob(data[x], mean1, std1)>prob(data[x], mean2, std2):
			newweights[x] = 1
		else:
			newweights[x] = 0

	plt.scatter(data, np.ones(size), c=weights, s=40)
	plt.show()