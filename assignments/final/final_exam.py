import numpy as np
import scipy.signal

# Task 32
x = np.array([2, 1, -1, -2, -3])
y = np.array([1, 2, 1])
convolution_result = np.convolve(x, y, mode='full')

print(convolution_result)


# Task 33
x2 = np.array([[2, 1, -1, -2],
               [-3, -4, -5, -6],
               [3, 4, 5, 6],
               [7, 8, 9, 10]])

y2 = np.array([[1, 2],
               [2, 1]])

convolution_2d_result = scipy.signal.convolve2d(x2, y2, mode='valid')

print(convolution_2d_result)
