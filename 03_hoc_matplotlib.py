#matplotlib: hiển thị dữ liệu dưới dạng biểu đồ
import matplotlib.pyplot as plt
import numpy
import time

x=[5,6]
y=[9,10]

plt.plot(x,y)
#plt.show()

i=0
start_time = time.time()
end_time = time.time()
while(i<1000000000):
    i=i+1

    print(i)
    if(i==100000):
        break
print (i,(end_time - start_time) * 1000)