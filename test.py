import numpy as np
hoge = [True for i in range(10)]
huga = [True for i in range(10)]

piyo = [hoge, huga]
piyo = [np.array(value) for value in piyo]

print(sum(piyo))
