#bài toán: dựa vào dữ liệu của bệnh nhân và chuẩn đoán bệnh tự động
#quy ước dữ liệu:
# nhẹ:          1
#thấp:          2
#trung bình:    3
#cao:           4
#nặng:          5
#ít:            6
#nhiều:         7
#quy ước kết quả:
#có;            1
#ko:            0

#các bước trong nhận dạng dữ liệu (máy học):
#b1: thu thập dữ liệu
#b2: xử lý dữ liệu
#b3: cho vào model để training
#b4: sử dụng model đoán kết quả
#b5: đánh giá xem model hiệu quả ko?

from sklearn import tree

my_tree=tree.DecisionTreeClassifier() #khởi tạo cây quyết định

dactrung=[[1, 3, 3, 7],
          [5, 2, 4, 6],
          [1, 2, 4, 6],
          [5, 4, 4, 3],
          [1, 4, 4, 7],
          [3, 2, 3, 7],
          [3, 3, 3, 6],
          [5, 2, 2, 7]]

nhan=[0, 1, 1, 0, 0, 0, 0, 1]# nhãn

#training

resault= my_tree.fit(dactrung,nhan)

kq= resault.predict([[1, 4, 3, 6]])

if(kq==1):
    kq1='co'
else:
    kq1='ko'
print("benh nhan co bi benh tim: "+kq1)