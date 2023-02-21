from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

data=pd.read_csv("housing.csv")
# data=np.array(data)

s=(data.dtypes == 'object')
object_cols=list(s[s].index)

#独热编码
OH_encoder=OneHotEncoder(handle_unknown='ignore',sparse=False)
OH_cols_data=pd.DataFrame(OH_encoder.fit_transform(data[object_cols]))

OH_cols_data.index=data.index
num_data=data.drop(object_cols,axis=1)
OH_data=pd.concat([num_data,OH_cols_data],axis=1)

#处理total_rooms中的缺失值
OH_data=OH_data.dropna(axis=0,subset=['total_bedrooms'])

#将median_house_value移到最后一列
housing=OH_data.drop('median_house_value',axis=1)
housing_labels=OH_data['median_house_value']
f_data=pd.concat([housing,housing_labels],axis=1)

#划分样本
train_set,test_set=train_test_split(f_data,test_size=0.3,random_state=117)

col=train_set.shape[1]
x_train=train_set.iloc[:,0:col-1]
y_train=train_set.iloc[:,col-1:col]
x_test=test_set.iloc[:,0:col-1]
y_test=test_set.iloc[:,col-1:col]

x_train=np.matrix(x_train.values)
y_train=np.matrix(y_train.values)
x_test=np.matrix(x_test.values)
y_test=np.matrix(y_test.values)

m=y_train.size
n=y_test.size

#特征缩放
def norm(x):
    sigma = np.std(x, axis=0) # axis=0计算每一列的标准差,=1计算行的
    mu = np.mean(x, axis=0)
    x = (x-mu)/sigma
    return x

x_train=norm(x_train)
x_test=norm(x_test)

# print(x_train)
# print(x_test)

#x加一列1
x_train=np.c_[np.ones(m),x_train]
x_test=np.c_[np.ones(n),x_test]

# 初始化theta,theta1的值（其中theta用于梯度下降中，theta1用于正规方程中）
theta=(np.matrix([0,0,0,0,0,0,0,0,0,0,0,0,0,0])).T
theta1=theta

num_iteration = 2000 #初始化迭代次数
alpha = 0.01 #初始化学习速率

#初始化一个一维向量用于存放每次迭代的代价值
J = np.zeros(num_iteration)

#代价函数
def cost(theta, x=x_train, y=y_train, m=m):
    h_x=x*theta
    inner=np.sum(np.power(h_x-y,2))
    return inner/(2*m)

#梯度下降优化
def gradient(theta, alpha,x=x_train,y=y_train):
    for i in range(num_iteration):
        J[i]=cost(theta,x) #将每次迭代的代价函数值计入
        theta=theta-(alpha/m)*(x.T@(x@theta-y))
    return theta

#正规方程
def normal(theta=theta1,x=x_train,y=y_train):
    theta= np.linalg.inv(x.T@x)@x.T@y #求矩阵的逆
    return theta

theta = gradient(theta, alpha)
theta1 = normal(theta1)

print("梯度下降theta:")
print(theta)
print("闭合形式theta:")
print(theta1)

#R^2函数
def R_squared(theta,x=x_test,y=y_test):

    y_pred=x*theta
    mu=np.mean(y,axis=0)
    SSE=np.sum(np.power(y-y_pred,2))
    SSR=np.sum(np.power(y_pred-mu,2))
    SST=SSR+SSE
    r_2=1-SSE/SST
    return r_2


print("测试集上梯度下降的R2:")
print(R_squared(theta))

print("测试集上正规方程的R2:")
print(R_squared(theta1))

print("训练集上梯度下降的R2:")
print(R_squared(theta,x_train,y_train))

print("训练集上正规方程的R2:")
print(R_squared(theta1,x_train,y_train))

print("梯度下降法的代价值:")
print(cost(theta))
print("正规方程的代价值:")
print(cost(theta1))

#各变量与房价的相关系数
temp=data.copy()
corr=temp.corr()
score=corr['median_house_value'].sort_values()
print(score)
