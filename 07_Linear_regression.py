#   Hồi quy tuyến tính
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv('heart_failure_clinical_records_dataset.csv')#đọc file data
print(data)

x=data.values[:,0]#lấy dữ liệu cột tuổi gán cho x
y=data.values[:,4]#lấy dữ liệu cột phân xuất tống máu gán cho y
# print(x)
# print(y)
print(len(x))

plt.xlabel('tuoi')
plt.ylabel('phan tram mau')
plt.title('mau chay vao tim')

#hiển thị dữ liệu lên đồ thị
#plt.scatter(x,y,marker='o')
plt.bar(x,y,color='b')
plt.show()

#hàm dự đoán
def predict (new_age,weight,bias):
    return weight*new_age+bias

#hàm chi phí tại thời điểm hiện tại mà nó học đc
def cost_function(x,y,weight,bias):
    n=len(x)
    sum_err=0
    for i in range(n):
        sum_err=(y[i]-(weight*x[i]+bias))**2
    return sum_err/n

def update_weight(x,y,weight,bias,learning_rate):
    n=len(x)
    weight_temp=0
    bias_temp=0
    for i in range(n):
        weight_temp+=-2*x[i]*(y[i]-(weight*x[i]+bias))
        bias_temp+=-2*(y[i]-(weight*x[i]+bias))
    weight-= (weight_temp/n)*learning_rate
    bias-= (bias_temp/n)*learning_rate
    return  weight,bias
#ham training
def train(x,y,weight,bias,learning_rate,iter):
    cos_his=[]
    for i in range(iter):
        weight,bias=update_weight(x,y,weight,bias,learning_rate)
        cost=cost_function(x,y,weight,bias)
        cos_his.append(cost)

    return weight,bias,cos_his

weight,bias,cos_his=train(x,y,0.03,0.0014,0.001,30)
print("ket qua:")
print(weight)
print(bias)
print(cos_his)
print("gia tri du doan",predict(50,weight,bias))

solanlap=[i for i in range(30)]
plt.plot(solanlap,cos_his)
plt.title('hien thi ket qua')
plt.show()