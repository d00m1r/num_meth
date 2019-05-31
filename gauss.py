import numpy as np
import scipy.linalg as lg
import matplotlib.pyplot as plt
import time
import numpy.random as nr

def mx_create(size):
	f = nr.rand(size)
	Mx = nr.rand(size, size)
	for i in range(size):
		Sum = 0
		for j in range(size):
			Sum = Sum + abs(Mx[i][j])
		Mx[i][i] = Sum
	return Mx, f
	
def direct(Mx, f, size):
    for k in range(size):
        t = Mx[k][k]
        Mx[k] = Mx[k] / t
        f[k] = f[k] / t
        for i in range(k + 1, size):
            Mx[i] = Mx[i] - Mx[k] * Mx[i][k]
            f[i] = f[i] - f[k] * Mx[i][k]
            Mx[i][k] = 0
    return Mx, f
    
def reverse(Mx, f, size):
    x = [0] * size
    for i in range(size - 1, -1, -1):
        x[i] = f[i]
        for j in range(i + 1, size):
            x[i] = x[i] - Mx[i][j] * x[j]
    return np.array(x)

l = list(range(100, 301, 20))
lib, gauss = [0], [0]
for size in l:
	Mx, f = mx_create(size)
	
	t0 = time.time()
	x1 = lg.solve(Mx,f)
	t1 = time.time() - t0
	
	t0 = time.time()
	Mx, f = direct(Mx, f, size)
	x2 = reverse(Mx, f, size)
	t2 = time.time() - t0
	
	lib.append(t1)
	gauss.append(t2)
	print('Mx size:',size, '\tlib: ',round(t1,3),'s', 'usr: ',round(t2,3),'s')

x = [0] + l
plt.plot(x, lib, label = "Library")
plt.plot(x, gauss, label = "Gauss")
plt.legend(loc = "upper left")
plt.xlabel("Matrix size")
plt.ylabel("Time, s")
plt.axis([0, l[-1], 0, gauss[-1]])
plt.savefig('gauss.png', bbox_inches='tight')
plt.show()
