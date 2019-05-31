import numpy as np
import scipy.linalg as lg
import matplotlib.pyplot as plt
import time
import numpy.random as nr

def mx_create(size):
	Mx = np.ones((size,size)) * 0
	a = nr.rand(size)
	b = nr.rand(size)
	c = nr.rand(size)
	f = nr.rand(size)
	a[0], c[size - 1] = 0, 0
	for i in range(size):
		b[i] = abs(a[i]) + abs(b[i]) + abs(c[i])
		Mx[i][i] = b[i]
		if i > 0:
			Mx[i][i - 1] = a[i]
		if i < size - 1:
			Mx[i][i + 1] = c[i])
	return Mx, a, b, c, f
	
def make(a, b, c, f, N):
    alpha = np.array([0.0] * (N + 1))
    beta = np.array([0.0] * (N + 1))
    for i in range(N):
        alpha[i + 1] = -c[i] / (a[i] * alpha[i] + b[i])
        beta[i + 1] = (f[i] - a[i] * beta[i]) / (a[i] * alpha[i] + b[i])
    x = np.array([0.0] * N)
    x[N - 1] = beta[N]
    for i in range(N - 2, -1, -1):
        x[i] = alpha[i + 1] * x[i + 1] + beta[i + 1]
    return x

def LibRun(mx, f):
	ud = np.insert(np.diag(mx, 1), 0, 0)
	d = np.diag(mx)
	ld = np.insert(np.diag(mx, -1), len(d)-1, 0)
	return lg.solve_banded((1, 1), np.matrix([ud, d, ld]), f)
	
l = list(range(1000, 12001, 1000))
lib, run = [0], [0]
for size in l:
	Mx, a, b, c, f = mx_create(size)
	
	t0 = time.time()
	x2 = LibRun(Mx, f)
	t1 = time.time() - t0
	
	t0 = time.time()
	x1 = make(a, b, c, f, size)
	t2 = time.time() - t0
	
	lib.append(t1)
	run.append(t2)
	print('Mx size:',size, '\tlib: ',round(t1,3),'s', 'usr: ',round(t2,3),'s')

x = [0] + l
plt.plot(x, lib, label = "Library")
plt.plot(x, run, label = "Run")
plt.legend(loc = "upper left")
plt.xlabel("Matrix size")
plt.ylabel("Time, s")
plt.axis([0, l[-1], 0, run[-1]])
plt.savefig('run.png', bbox_inches='tight')
plt.show()
