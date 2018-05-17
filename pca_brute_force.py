import matplotlib.pyplot as plt
from sklearn import datasets

iris = datasets.load_iris()

X = iris.data
y = iris.target

labels = ['sepal length', 'sepal width', 'petal length', 'petal width']

for xx in range(4):
    for yy in range(4):
        if yy > xx:
            print(xx, yy)
            plt.xlabel(labels[xx])
            plt.ylabel(labels[yy])
            for index in [0, 1, 2]:
                plt.scatter(iris.data[y == index][:, xx],
                            iris.data[y == index][:, yy])
            plt.show()
