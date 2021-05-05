from sklearn import datasets
iris = datasets.load_iris()
feature = iris['data']
label = iris['target']
print(feature.shape)
print(iris.keys())
print(iris['feature_names'])
print(label)