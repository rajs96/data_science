import re
import numpy as np
import matplotlib.pyplot as plt

X = [] 
Y = []
#not really sure what this does, regex stuff
non_decimal = re.compile(r'[^\d]+')

for line in open('moore.csv'):
	r = line.split('\t')

	x = int(non_decimal.sub('', r[2].split('[')[0]))
	y = int(non_decimal.sub('', r[1].split('[')[0]))

	X.append(x)
	Y.append(y)

X = np.array(X)
Y = np.array(Y)

plt.scatter(X,Y)
plt.show()

Y = np.log(Y)
plt.scatter(X,Y)
plt.show()