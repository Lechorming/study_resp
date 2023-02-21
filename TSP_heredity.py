import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties

class YIchuan(object):
    best_distance = -1  # 记录目前最优距离
    best_gene = []  # 记录目前最优旅行方案
    all_best_distance = [] #记录每一代最优距离
    citys = np.array([])  # 城市数组
    citys_name = np.array([])
    population_size = 100  # 种群大小，每个种群含有多少条基因
    cross_rate = 0.9  # 交叉率
    change_rate = 0.1  # 突变率
    population = np.array([])  # 种群数组
    fitness = np.array([])  # 适应度数组
    city_size = -1  # 标记城市数目
    iter_num = 200  # 最大迭代次数
    

    def __init__(self, city_size, cross_rate, change_rate, population_size, iter_num):
        self.fitness = np.zeros(self.population_size)
        self.city_size=city_size
        self.cross_rate = cross_rate
        self.change_rate = change_rate
        self.population_size = population_size
        self.iter_num = iter_num
        self.fig, self.ax = plt.subplots()
        self.plt = plt

    def init(self):
        TSP = self
        TSP.load_city_data()    # 加载城市数据
        TSP.population = TSP.creat_population(TSP.population_size)  # 创建种群
        TSP.fitness = TSP.get_fitness(TSP.population)  # 计算初始种群适应度
        self.ax.axis([0, 100, 0, 100])


    def creat_population(self, size):
        """
        创建种群
        :param size:种群大小
        :return: 种群
        """
        population = [] # 存储种群生成的基因
        for i in range(size):
            gene = np.arange(self.citys.shape[0])
            np.random.shuffle(gene)  # 打乱数组[0，...，city_size]
            population.append(gene)  # 加入种群
        return np.array(population)

    def get_fitness(self, population):
        """
        获得适应度
        :param population:种群
        :return: 种群每条基因对应的适应度
        """
        fitness = np.array([])  # 适应度记录数组
        for i in range(population.shape[0]):
            gene = population[i]  # 取其中一条基因（编码解，个体）
            dis = self.compute_distance(gene)  # 计算此基因优劣（距离长短）
            dis = self.best_distance / dis  # 当前最优距离除以当前population[i]（个体）距离；越近适应度越高，最优适应度为1
            fitness = np.append(fitness, dis)  # 保存适应度population[i]
        return fitness


    def select_population(self, population):
        """
        选择种群，优胜劣汰
        策略：低于平均的要替换
        :param population: 种群
        :return: 更改后的种群
        """
        best_index = np.argmax(self.fitness)
        ave = np.median(self.fitness, axis=0)
        for i in range(self.population_size):
            if i != best_index and self.fitness[i] < ave:
                pi = self.cross(population[best_index], population[i]) #交叉
                pi = self.change(pi) #变异
                population[i, :] = pi[:]

        return population


    def cross(self, parent1, parent2):
        """
        交叉
        :param parent1: 父亲
        :param parent2: 母亲
        :return: 儿子基因
        """
        if np.random.rand() > self.cross_rate:
            return parent1
        index1 = np.random.randint(0, self.city_size - 1)
        index2 = np.random.randint(index1, self.city_size - 1)
        tempgene = parent2[index1:index2]  # 交叉的基因片段
        newgene = []
        xxx = 0
        for g in parent1:
            if xxx == index1:
                newgene.extend(tempgene)  # 插入基因片段
            if g not in tempgene:
                newgene.append(g)
            xxx += 1
        newGene = np.array(newgene)

        return newGene

    def reverse_gene(self, gene, i, j):
        """
        翻转i到j位置的基因
        :param gene: 基因
        :param i: 第i个位置
        :param j: 第j个位置
        :return: 翻转后的基因
        """
        if i >= j:
            return gene
        if j > self.city_size - 1:
            return gene
        parent1 = np.copy(gene)
        tempgene = parent1[i:j]
        newgene = []
        p1len = 0
        for g in parent1:
            if p1len == i:
                newgene.extend(tempgene[::-1])  # 插入基因片段
            if g not in tempgene:
                newgene.append(g)
            p1len += 1
        return np.array(newgene)

    def change(self, gene):
        """
        突变,主要使用翻转
        :param gene: 基因
        :return: 突变后的基因
        """
        if np.random.rand() > self.change_rate:
            return gene
        index1 = np.random.randint(0, self.city_size - 1)
        index2 = np.random.randint(index1, self.city_size - 1)
        new_gene = self.reverse_gene(gene, index1, index2)  #翻转
        return new_gene

    def evolution(self):
        """
        迭代进化种群
        :return: None
        """
        for i in range(self.iter_num):
            #self.fitness = self.get_fitness(self.population)  # 计算种群适应度
            best_index = np.argmax(self.fitness)
            worst_f_index = np.argmin(self.fitness)
            local_best_genee = self.population[best_index]
            local_best_distance = self.compute_distance(local_best_genee)
            if i == 0:  #第一代记录最优基因和最短距离
                self.best_gene = local_best_genee
                self.best_distance = self.compute_distance(local_best_genee)

            if local_best_distance < self.best_distance:
                self.best_distance = local_best_distance  # 记录最优值
                self.best_gene = local_best_genee  # 记录最个体基因
            else:
                self.population[worst_f_index] = self.best_gene  #替换掉最差的基因
            print('代数:%d 最优距离:%s' % (i, self.best_distance))
            self.all_best_distance.append(self.best_distance)
            self.fitness = self.get_fitness(self.population)  # 计算种群适应度
            self.population = self.select_population(self.population)  # 选择淘汰种群
            for j in range(self.population_size):
                k = np.random.randint(0, self.population_size - 1)
                if j != k:
                    self.population[j] = self.cross(self.population[j], self.population[k])  # 交叉种群中第j,k个体的基因
                    self.population[j] = self.change(self.population[j])  # 突变种群中第j个体的基因


    def load_city_data(self, file='city.csv'):
        # 加载实验数据
        data = pd.read_csv(file, header=None).values
        self.citys = data[:self.city_size, 1:]
        self.citys_name = data[:self.city_size, 0]



    def compute_distance(self, gen):
        # 计算该基因的总距离
        distance = 0.0
        for i in range(-1, len(self.citys) - 1):
            index1, index2 = gen[i], gen[i + 1]
            city1, city2 = self.citys[index1], self.citys[index2]
            distance += np.sqrt((city1[0] - city2[0]) ** 2 + (city1[1] - city2[1]) ** 2)
        return distance


    def compute_oushi_distance(self, city1, city2):
        # 计算两个地点之间的欧氏距离
        dis = np.sqrt((city1[0] - city2[0]) ** 2 + (city1[1] - city2[1]) ** 2)
        return dis

    def draw_line(self, from_city, to_city):
        """
        连线
        :param from_city: 城市来源
        :param to_city: 目的城市
        :return: none
        """
        line1 = [(from_city[0], from_city[1]), (to_city[0], to_city[1])]
        (line1_xs, line1_ys) = zip(*line1)
        self.ax.add_line(Line2D(line1_xs, line1_ys, linewidth=1, color='blue'))


    def draw_citys_way(self, gene):
        '''
        根据一条基因绘制一条旅行路线
        :param gene:
        :return:none
        '''
        num = gene.shape[0]
        self.ax.axis([0, 100, 0, 100])
        for i in range(num):
            if i < num - 1:
                best_i = self.best_gene[i]
                next_best_i = self.best_gene[i + 1]
                best_icity = self.citys[best_i]
                next_best_icity = self.citys[next_best_i]
                self.draw_line(best_icity, next_best_icity)
        start = self.citys[self.best_gene[0]]
        end = self.citys[self.best_gene[-1]]
        self.draw_line(end, start)

    def draw_citys_name(self, gen, size=5):
        '''
        根据一条基因gen绘制对应城市名称
        :param gen:
        :param size: text size
        :return:
        '''
        m = gen.shape[0]
        self.ax.axis([0, 100, 0, 100])
        for i in range(m):
            c = gen[i]
            best_icity = TSP.citys[c]
            self.ax.text(best_icity[0], best_icity[1], TSP.citys_name[c], fontsize=10)

    def draw(self):
        """
        绘制最终结果
        :return: none
        """
        self.ax.plot(self.citys[:, 0], self.citys[:, 1], 'ro')
        self.draw_citys_name(self.population[0], 8)
        self.draw_citys_way(self.best_gene)
        plt.xlabel("X");
        plt.ylabel("Y");
        plt.grid();
        plt.show()

    def draw_dis(selfs):
        """
        绘制最优距离曲线
        :return: none
        """
        x = [i for i in range(len(selfs.all_best_distance))]
        y = selfs.all_best_distance

        fig, ax = plt.subplots(figsize=(15, 10))
        ax.plot(x, y, color='darkblue', linestyle='-')

        plt.xlabel("Generation")
        plt.ylabel("Best Answer")
        plt.title("Heredity")
        plt.grid()
        plt.show()


if __name__ == '__main__':
    # 消融实验
    # number = np.ones([10,10])
    # for i in range(10):
    #     for j in range(10):
    #         TSP = YIchuan(30, 0.1*(i+1), 0.1*(j+1), 100, 300)  # 交叉率、突变率、种群大小、进化代数
    #         TSP.init()
    #         TSP.evolution()
    #         # TSP.draw()
    #         # TSP.draw_dis()
    #         print("最优路径为：")
    #         print(np.append(TSP.best_gene, TSP.best_gene[0]) + 1)
    #         number[i,j] = TSP.best_distance

    # number = pd.DataFrame(number)
    # number.to_csv('out.csv')
    
    # 参数：城市数、交叉率、突变率、种群大小、进化代数
    TSP = YIchuan(30, 0.6, 0.1, 500, 300)
    TSP.init()
    TSP.evolution()
    TSP.draw()
    TSP.draw_dis()
    print("最优路径为：")
    print(np.append(TSP.best_gene,TSP.best_gene[0])+1)
