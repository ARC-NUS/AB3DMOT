from numba import jit, prange
import time


@jit(parallel=True)
def test_prange():
	for i in prange(5):
		for j in prange(5):
			print "here", i
			print "end", i

if __name__ == '__main__':
	test_prange()