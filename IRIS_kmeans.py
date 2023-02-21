import random
import pandas as pd
import numpy as np
from itertools import chain
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

# 读取数据
def getData():
    iris = pd.read_csv('IRIS数据集/iris.data',header = None,names=["Sepal.Length","Sepal.Width","Petal.Length","Petal.Width","species"])
    X = iris['Sepal.Length'].copy()
    X = np.array(X).reshape(-1,1)
    Y = iris['Petal.Length'].copy()
    Y = np.array(Y).reshape(-1,1)
    label = iris['species'].copy()
    enc = LabelEncoder()   #获取一个LabelEncoder
    enc = enc.fit(['Iris-setosa','Iris-versicolor','Iris-virginica'])  #训练LabelEncoder
    label = enc.transform(label)       #使用训练好的LabelEncoder对原数据进行编码
    label = np.array(label).reshape(-1,1)
    dataSet = np.hstack((X, Y,label))
    return dataSet

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
    # 分组并计算新的质
    minDistIndices = np.argmin(clalist, axis=1)
    newCentroids = (pd.DataFrame(dataSet).groupby(minDistIndices)).mean()
    newCentroids = newCentroids.values

    # 计算变化量
    change = newCentroids - centroids
    return change, newCentroids
 
# 使用K-means分类
def kmeans(dataSet, k):
    #随机选取质心
    centroids = random.sample(list(dataSet[:,:2]), k)
    
    # 更新质心，直到变化量全为0
    change, newCentroids = classify(dataSet[:,:2], centroids, k)
    while np.any(change != 0):
        change, newCentroids = classify(dataSet[:,:2], newCentroids, k)
    
    centroids = sorted(newCentroids.tolist())

    # 根据质心计算每个集群
    cluster = []
    curr = 0
    clalist = calcDis(dataSet[:,:2],centroids,k)   # 计算每个点分别到三个质心的距离
    minDistIndices = np.argmin(clalist,axis = 1)   # 选取距离每个质心距离最小的点返回下标
    for i in range(len(minDistIndices)):
        if minDistIndices[i] == dataSet[i:i+1,2]:
            curr += 1
    for i in range(k):
        cluster.append([])
    for i, j in enumerate(minDistIndices):  # 分类
        cluster[j].append(dataSet[i:i+1,:2])

    return centroids,cluster,minDistIndices,curr

# 可视化结果
def Draw(centroids,cluster):
    cluster0 = list(chain.from_iterable(cluster[0]))# 类别1
    cluster0 = np.array(cluster0)
    cluster1 = list(chain.from_iterable(cluster[1]))# 类别2
    cluster1 = np.array(cluster1) 
    cluster2 = list(chain.from_iterable(cluster[2]))# 类别3
    cluster2 = np.array(cluster2) 
    ax = plt.subplot()
    ax.scatter(cluster0[:,0],cluster0[:,1],marker = '*',c = 'green',label = 'Iris-setosa')
    ax.scatter(cluster1[:,0],cluster1[:,1],marker = 'o',c = 'blue',label = 'Iris-versicolor')
    ax.scatter(cluster2[:,0],cluster2[:,1],marker = '+',c = 'orange',label = 'Iris-virginica')
    ax.scatter(centroids[:,0],centroids[:,1],marker='x',c = 'red',label = 'centroid')
    plt.title("k-means clustering results")
    plt.xlabel("Sepal.Length") 
    plt.ylabel("Petal.Length")
    plt.legend()
    plt.show()

def eva_index(dataSet):
    # 存放不同的SSE值
    sse_list =  []
    # 轮廓系数
    silhouettes = []
    for k in range(2,9):   # k取值范围
        sum = 0
        centroids, cluster, minDistance, curr = kmeans(dataset, k)
        minDistance = np.array(minDistance)
        centroids = np.array(centroids)
        # 计算误方差值
        for i in range(len(cluster)):
            temp = np.sum((cluster[i] - centroids[i])**2)
            sum += temp
        sse_list.append(sum)
        # 计算轮廓系数
        silhouette = metrics.silhouette_score(dataSet[:,:2],minDistance,metric='euclidean')
        silhouettes.append(silhouette)
    # 绘制簇内误方差曲线
    plt.subplot(211)
    plt.title('KMeans Intra-cluster error variance')
    plt.plot(range(2, 9), sse_list, marker='*')
    plt.xlabel('Number of clusters')
    plt.ylabel('Intra-cluster error variance(SSE)')
    plt.show()

    # 绘制轮廓系数曲线
    plt.subplot(212)
    plt.title('KMeans Profile coefficient curve')
    plt.plot(range(2,9), silhouettes, marker = '+')
    plt.xlabel('Number of clusters')
    plt.ylabel('silhouette coefficient')
    plt.show()

if __name__=='__main__':
    dataset = getData()  # 加载数据
    eva_index(dataset)   # kmeans评价指标
    centroids, cluster,minDistance, curr = kmeans(dataset, 3)  # k=3时进行聚类
    centroids = np.array(centroids) 
    currD = curr/len(dataset) * 100  # 计算正确率
    print("k-means正确率:",currD,"%")
    Draw(centroids,cluster)   # 结果可视化
