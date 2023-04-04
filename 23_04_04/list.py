import numpy as np
import pandas as pd
a = [[1,2,3], [4,5,6]]
b = np.array(a)

# print(b)
# [[1 2 3]
#  [4 5 6]]

c = [[1,2,3], [4,5]]
# print(c)
# [[1, 2, 3], [4, 5]]

d = np.array(c)
# print(d)
# [list([1, 2, 3]) list([4, 5])]
# List doesn't matter of different size.

e = [[1,2,3], ['boy', 'jini', 5, 6]]
# print(e)
# [[1, 2, 3], ['boy', 'jini', 5, 6]]
# List doesn't matter of different type.

f = np.array(e)
# print(f)
# [list([1, 2, 3]) list(['boy', 'jini', 5, 6])]