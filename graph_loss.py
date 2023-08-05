import matplotlib.pyplot as plt
import sys
import numpy as np

# epochs = int(sys.argv[2])

def best_fit(train, epochs, n=2):
	xtrain = np.array([1 + i / len(train) * (epochs-1) for i in range(len(train))])

	plt.scatter(xtrain, train, alpha=.5, linewidths=.5, s= 4)

	p = np.polyfit(xtrain, train, n)

	liney = np.zeros(xtrain.shape)
	for i in range(0,n+1):
		liney += p[i] * xtrain ** (n-i)

	plt.plot(xtrain, liney, color='red')   

	plt.show()

if __name__ == '__main__':

	data = None
	with open(sys.argv[1], "r") as f:
		data = f.read().split('\n')

	test = []
	train = []

	training = True
	for i in range(1,len(data)):
		if (data[i] == 'testing'):
				training = False
				continue
		elif (data[i] == ''):
			continue
		if (training):
			train.append(float(data[i]))
		else:
			test.append(float(data[i]))

	test = np.array(test)
	train = np.array(train)

	epochs = len(test) + 1

	print(test)
	# print(train)

	best_fit(train, epochs, 10)
