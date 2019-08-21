

from hunga_bunga import HungaBungaClassifier, HungaBungaRegressor
from hunga_bunga.regression import gen_reg_data
from sklearn import datasets


# ---------- Getting The Data ----------

iris = datasets.load_iris()
X_c, y_c = iris.data, iris.target
X_r, y_r = gen_reg_data(10, 3, 100, 3, sum, 0.3)



# ---------- Classification ----------

clf = HungaBungaClassifier()
clf.fit(X_c, y_r)
print(clf.predict(X_c))



# ---------- Regression ----------

mdl = HungaBungaRegressor()
mdl.fit(X_c, y_r)
print(mdl.predict(X_c))


