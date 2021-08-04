#numpy: hỗ trợ cho các phép tính trên mảng
import numpy as np

a=np.array([1,2,3,4,5,6])
c=np.array([1,2,3,4,5,6])

#hiển thị số chiều của mảng:
print("số chiều của mảng a: ",a.ndim)

b=np.array([[1,2,3,4],[5,6,7,8]])
print("số chiều của mảng a: ",b.ndim)

#hiển thị số hàng và cột:
print("số hàng và cột của mảng a: ", a.shape)

print("số hàng và cột của mảng b: ",b.shape)

#lấy độ dài mảng:
print('độ dài mảng a: ', len(a))

#các phép tính trên mảng:
print(a+c)