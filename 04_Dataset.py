from sklearn.datasets import load_iris
from sklearn.model_selection import  train_test_split
from sklearn.tree import DecisionTreeClassifier # cây quyết định
import numpy as np

iris_dataset=load_iris()

print(iris_dataset)
print(len(iris_dataset.target))

#ghi 4 giá trị vì hàm train_test_split trả về 4 tham số
x_train,x_test,y_train,y_test = train_test_split(iris_dataset.data,iris_dataset.target,random_state=0)

print(x_test)

model=DecisionTreeClassifier()

myModel= model.fit(x_train,y_train)

print(myModel.predict(x_test))