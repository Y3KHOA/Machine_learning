#hồi quy logictic
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data= pd.read_csv('heart_failure_clinical_records_dataset.csv')

true_x=[]
true_y=[]
false_x=[]
false_y=[]

print(data.values)
for item in data.values:
    if item[12]==1:
        true_x.append(item[2])
        true_y.append(item[11])
    else:
        false_x.append(item[2])
        false_y.append(item[11])

plt.scatter(true_x,true_y,marker='o',c='r')
plt.scatter(false_x,false_y,marker='s',c='b')
plt.show()

# np.exp: tính toán e^x cho mỗi giá trị x trong mảng đầu vào của bạn.
def sigmoid(z):
    return 1.0/( 1+np.exp(-z))

# hàm phân chia ranh giới
def phan_chia(p):#chuyền vào 1 xác xuất dự đoán
    if p>=0.5:
        return 1
    else:
        return 0

# hàm dự đoán
# np.dot: Để nhân 2 ma trận hoặc nhân vector với ma trận trong numpy, chúng ta sử dụng hàm
def predict(feature, weights):
    z= np.dot(feature,weights)
    return sigmoid(z)

# tính hàm chi phí:
def cost_function(feature,labels,weights):
    """
    :param feature: 1 cái mảng có kích thước 24(chưa tính nhãn: đã chết) tính thêm bias nữa là 25
    :param labels: trả về giá trị là sống hoặc chết
    :param weights: lấy số trạng thái nhân 1; 299x1
    :return: chi phí cost
    """
    n=len(labels)
    predictions=predict(feature,weights) # truyền vào đặc trưng và trọng số

    """
    kiểu trả về của prediction: mảng giá trị dự đoán.
    vd: 0.6, 0.7, 0.5
    bị lẫn lộn bởi những nhãn 1,0
    nên cần tính tổng 1 và 0 tách biệt ra
    """

    # tính chi phí theo y=1
    cost_class1=-labels*np.log(predictions)

    # tính chi phí theo y=0
    cost_class2=-(1-labels)*np.log(1-predictions)

    cost= cost_class1+cost_class2

    return cost.sum()/n

# cập nhập lại weight
def update_weight(feature,lable,weights, learning_rate):
    """
    :param feature: 299x25
    :param lable: 299x1
    :param weights: 25x1
    :param learning_rate: float
    :return:float
    """

    n=len(lable)

    #lấy giá trị dự đoán của tất cả các diểm;
    prediction=predict(feature,weights)

    # tính độ dốc gradient
    gd=np.dot(feature.T,(prediction - lable)) # .T: ma trận chuyển vị của nó
    gd=gd/n

    gd=gd*learning_rate

    weights=weights-gd

    return weights

def train(feature,labels,weights,learning_rate,iter):
    cost_hs=[]
    for i in range(iter):
        weights=update_weight(feature,labels,weights,learning_rate)
        cost_hs=cost_function(feature,labels,weights)
    return weights,cost_hs

weights,bias,cost_hs=train(true_x,true_y,false_y,false_y)

