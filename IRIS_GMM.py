import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# 预处理数据
def loadDataSet():
    iris = pd.read_csv('IRIS数据集/iris.data',header = None,names=["Sepal.Length","Sepal.Width","Petal.Length","Petal.Width","species"])
    X = iris['Sepal.Length'].copy()
    X = np.array(X).reshape(-1,1)
    Y = iris['Petal.Length'].copy()
    Y = np.array(Y).reshape(-1,1)
    dataSet = np.hstack((X, Y))
    label = iris['species'].copy()
    enc = LabelEncoder()   #获取一个LabelEncoder
    enc = enc.fit(['Iris-setosa','Iris-versicolor','Iris-virginica'])  #训练LabelEncoder
    label = enc.transform(label)       #使用训练好的LabelEncoder对原数据进行编码
    return dataSet,label


# 计算欧拉距离
def calcDis(dataSet, centroids, k):
    clalist = []
    for data in dataSet:
        diff = np.tile(data, (k, 1)) - centroids    # 分别与三个质心相减
        squaredDiff = diff ** 2
        squaredDist = np.sum(squaredDiff, axis = 1) # (x1-x2)^2 + (y1-y2)^2
        distance = squaredDist ** 0.5
        clalist.append(distance)
    clalist = np.array(clalist)
    return clalist


# 计算质心
def classify(dataSet, centroids, k):
    # 计算样本到质心的距离
    clalist = calcDis(dataSet, centroids, k)
    # 分组并计算新的质心
    minDistIndices = np.argmin(clalist, axis=1)
    newCentroids = pd.DataFrame(dataSet).groupby(minDistIndices).mean()
    newCentroids = newCentroids.values
    # 计算变化量
    change = newCentroids - centroids
    return change, newCentroids

# 使用K-means分类
def kmeans(dataSet, k):
    #随机选取质心
    centroids = random.sample(list(dataSet), k)
    # 更新质心，直到变化量全为0
    change, newCentroids = classify(dataSet, centroids, k)
    while np.any(change != 0):
        change, newCentroids = classify(dataSet, newCentroids, k)
    centroids = sorted(newCentroids.tolist())
    return centroids

# 高斯分布的概率密度函数
def prob(x, mu, sigma):
    n = np.shape(x)[1]
    expOn = float(-0.5 * (x - mu) * (np.linalg.inv(sigma)) * ((x - mu).T))
    divBy = pow(2 * np.pi, n / 2) * pow(np.linalg.det(sigma), 0.5)  # np.linalg.det 计算矩阵的行列式
    return pow(np.e, expOn) / divBy
 
# EM算法
def EM(dataMat,centroids, maxIter,c):
    m, n = np.shape(dataMat)  # m=150, n=2
    # 初始化参数
    alpha = [1 / 3, 1 / 3, 1 / 3]   # 系数
    mu = np.mat(centroids)    # 均值向量
    # mu = random.sample(list(dataMat), c)
    sigma = [np.mat([[0.1, 0], [0, 0.1]]) for x in range(3)]    # 初始化协方差矩阵Σ
    gamma = np.mat(np.zeros((m, c)))   # γ(ik)
    for i in range(maxIter):
        for j in range(m):
            sumAlphaMulP = 0
            for k in range(c):
                gamma[j, k] = alpha[k] * prob(dataMat[j, :], mu[k], sigma[k]) 
                sumAlphaMulP += gamma[j, k]    
            for k in range(c):
                gamma[j, k] /= sumAlphaMulP   # 计算后验分布
        sumGamma = np.sum(gamma, axis=0)
 
        for k in range(c):
            mu[k] = np.mat(np.zeros((1, n)))
            sigma[k] = np.mat(np.zeros((n, n)))
            for j in range(m):
                mu[k] += gamma[j, k] * dataMat[j, :]
            mu[k] /= sumGamma[0, k] #  更新均指向量
            for j in range(m):
                sigma[k] += gamma[j, k] * (dataMat[j, :] - mu[k]).T *(dataMat[j, :] - mu[k])
            sigma[k] /= sumGamma[0, k]  #  更新协方差矩阵
            alpha[k] = sumGamma[0, k] / m   # 更新混合系数
    return gamma


# 高斯混合聚类
def gaussianCluster(dataMat,centroids,k):
    m, n = np.shape(dataMat)
    clusterAssign = np.mat(np.zeros((m, n)))
    gamma = EM(dataMat,centroids,5,3)
    centroids = np.array(centroids)
    for i in range(m):
        clusterAssign[i, :] = np.argmax(gamma[i, :]), np.amax(gamma[i, :])
    for j in range(k):
        pointsInCluster = dataMat[np.nonzero(clusterAssign[:, 0].A == j)[0]]  # 将数据分类
        centroids[j, :] = np.mean(pointsInCluster, axis=0)  # 确定各均值中心，获得分类模型
    return centroids, clusterAssign
 
 
def showCluster(dataMat, k, centroids, clusterAssment):
    numSamples, dim = dataMat.shape

    mark = ['o', '+', '*']
    color = ['green','blue','orange']
    
    for i in range(numSamples):
        Index = int(clusterAssment[i, 0])
        plt.plot(dataMat[i, 0], dataMat[i, 1],mark[Index],c = color[Index],markersize = 7)

    for i in range(k):
        plt.plot(centroids[i, 0], centroids[i, 1], marker='x',c = 'red',label = 'centroid',markersize = 10)
    plt.title("GMM clustering results")
    plt.xlabel("Sepal.Length") 
    plt.ylabel("Petal.Length")
    plt.show()
 
if __name__=="__main__":
    dataSet,label = loadDataSet()       # 导入数据
    dataMat = np.mat(dataSet)
    centroids = kmeans(dataSet, 3)  # 使用K_means算法初始化质心
    centroids, clusterAssign = gaussianCluster(dataMat,centroids,3)  # GMM算法
    curr = 0
    l = len(dataSet)
    for i in range(l):
        if label[i] == clusterAssign[i,0]:
            curr += 1
    currD = curr/l * 100
    print("GMM准确率:",currD,"%")
    showCluster(dataMat, 3, centroids, clusterAssign)  # 可视化结果
