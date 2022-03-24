import sys

import numpy as np

input_file_1 = sys.argv[1]
input_file_2 = sys.argv[2]
output_file = sys.argv[3]

vec1 = np.load(input_file_1)
vec2 = np.load(input_file_2)
vec = np.vstack((vec1, vec2))
print(vec1.shape)
print(vec2.shape)
print(vec.shape)
print(vec1[0])
print(vec2[-1])
print(vec[0])
print(vec[-1])
np.save(output_file, vec)
