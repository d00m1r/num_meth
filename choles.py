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

def S_create(Mx,size):
	S = np.zeros((size, size))
	for i in range(size):
		Sum = 0.0
		for k in range(i):
			Sum += S[k, i] ** 2
		S[i, i] = (Mx[i, i] - Sum) ** 0.5
		for j in range(i + 1, size):
			S_sum = 0.0
			for k in range(i):
				S_sum += S[k, i] * S[k, j]
			S[i, j] = (Mx[i, j] - S_sum) / S[i, i]
	return S
  
            
def make(Mx, f, size):
    x = [0] * size
    for i in range(size):
        x[i] = f[i]
        for j in range(i):
            x[i] = x[i] - Mx[i][j] * x[j]
        x[i] = x[i] / Mx[i][i]
    return np.array(x)
    
l = list(range(100, 301, 20))
lib, cholesky = [0], [0]
for size in l:
	Mx, f = mx_create(size)
	
	t0 = time.time()
	c, low = lg.cho_factor(Mx)
	x1 = lg.cho_solve((c, low), f)
	t1 = time.time() - t0
	
	t0 = time.time()
	S = S_create(Mx, size)
	y = make(S, f, size)
	S.transpose()
	x2 = make(S, y, size)
	t2 = time.time() - t0
	
	lib.append(t1)
	cholesky.append(t2)
	print('Mx size:',size, ' 	lib: ',round(t1,3),'s', 'usr: ',round(t2,3),'s')

x = [0] + l
plt.plot(x, lib, label = "Library")
plt.plot(x, cholesky, label = "Cholesky")
plt.legend(loc = "upper left")
plt.xlabel("Matrix size")
plt.ylabel("Time, s")
plt.axis([0, l[-1], 0, cholesky[-1]])
plt.savefig('cholesky.png', bbox_inches='tight')
plt.show()
