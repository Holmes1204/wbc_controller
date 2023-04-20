import numpy as np 
 
a = np.array([[1,2,3],[4,5,6]])
b = np.arange(0, 1.0, 0.1)
c = np.sin(b)
# c 使用了关键字参数 sin_array
np.savez("runoob.npz", sdfd=a, sdf=b, sin_array = c)
r = np.load("runoob.npz")  
print(r.files) # 查看各个数组名称
print(r["sdfd"]) # 数组 a
print(r["sdf"]) # 数组 b
print(r["sin_array"]) # 数组 c
