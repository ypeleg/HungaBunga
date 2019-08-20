

from sklearn import datasets
iris = datasets.load_iris()
x, y = iris.data, iris.target




from hunga_bunga import HungaBungaClassifier, HungaBungaRegressor

clf = HungaBungaClassifier()
clf.fit(x, y)
clf.predict(x)





