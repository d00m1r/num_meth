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

def Jacobi(Mx, f, x, size):
    new = [0] * size
    for i in range(size):
        Sum = 0
        for j in range(i - 1):
            Sum = Sum + Mx[i][j] * new[j]
        for j in range(i + 1, size):
            Sum = Sum + Mx[i][j] * new[j]
        new[i] = (f[i] - Sum) / Mx[i][i]
    return new
    
def solve(Mx, f, size, eps):
    new = [0] * size
    x = nr.rand(size)
    while norma(x, new, size) > eps:
        x = new
        new = Jacobi(Mx, f, x, size)
    return new
    
def norma(x1, x2, size):
	t=0
	k=0
	for i in range(size):
	    t = abs(x1[i]-x2[i])
	    if t>k:
		    k=t
	return k

l = list(range(100, 301, 20))
lib, jac = [0], [0]
for size in l:
	Mx, f = mx_create(size)
	
	t0 = time.time()
	x1 = lg.solve(Mx,f)
	t1 = time.time() - t0
	
	t0 = time.time()
	x0 = nr.rand(size)
	x2 = solve(Mx, f, size, 0.001)
	t2 = time.time() - t0

	lib.append(t1)
	jac.append(t2)
	print('Mx size:',size, '\tlib: ',round(t1,3),'s', 'usr: ',round(t2,3),'s')

x = [0] + l
plt.plot(x, lib, label = "Library")
plt.plot(x, jac, label = "Jacobi")
plt.legend(loc = "upper left")
plt.xlabel("Matrix size")
plt.ylabel("Time, s")
plt.axis([0, l[-1], 0, jac[-1]])
plt.savefig('jac.png', bbox_inches='tight')
plt.show()
