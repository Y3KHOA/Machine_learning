#xử lý dữ liệu bị thiếu trong data .csv
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

data = pd.read_csv('dt.csv',header=None) # dùng pandas đọc file .csv (gồm tên file, header)
#hiện tại đang có dạng là data frame
print (data)

x=data.values # convert lại data thành mảng

#C1: lấy giá trị trung bình của cột để bù vào chỗ trống
#hàm xử lý dữ liệu missing
imp=SimpleImputer(missing_values=np.nan,strategy='mean')#nói cho hàm biết missing values là j
imp.fit(x)# hàm này cần chuyền vào mảng
#hàm fit chỉ cho dữ liệu vào thôi
kq=imp.transform(x)# thực sự chuyển đổi cần hàm trainsfrom
print('C1:')
print(kq)

#C2: lấy dữ liệu với tuần suất nhìu nhất để bù bào chỗ trống
imp2=SimpleImputer(missing_values=np.nan,strategy='most_frequent')
imp2.fit(x)
kq2=imp2.transform(x)
print('C2:')
print(kq2)