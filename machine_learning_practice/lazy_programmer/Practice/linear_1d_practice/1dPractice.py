import numpy as np
import matplotlib.pyplot as plt

X = []
Y = []

#loading data

for line in open('data_1d.csv'):
	x, y = line.split(',')
	X.append(float(x))
	Y.append(float(y))

#turning the arrays into numpy arrays
X = np.array(X)
Y = np.array(Y)

#plot data to see what it looks like
# plt.scatter(X,Y)
# plt.show()

#apply equations for best fit (a and b)

denominator = X.dot(X) - X.mean() * X.sum()
a = (X.dot(Y) - Y.mean() * X.sum() ) / denominator
b = (Y.mean() * X.dot(X) - X.mean() * X.dot(Y)) / denominator

# calculate predicted y
Yhat = a*X + b

#plot it
# plt.scatter(X,Y)
# plt.plot(X,Yhat)
# plt.show()

# calculate R squared
d1 = Y - Yhat #vector
d2 = Y - Y.mean() #vector
r2 = 1 - d1.dot(d1) / d2.dot(d2) #scalar
print "the r-squared is", r2







