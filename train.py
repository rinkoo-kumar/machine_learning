from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np
import pickle

def training():
	l = []
	mnist = fetch_openml('mnist_784', version=1, as_frame=False)
	X = mnist.data.astype(np.float64)
	y = mnist.target.astype(np.float64)
	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=10000, random_state=10)
	std = StandardScaler()
	l.append(std)
	X_train = std.fit_transform(X_train)
	X_test = std.transform(X_test)
	pca = PCA(.95)
	l.append(pca)
	X_train_pca = pca.fit_transform(X_train)
	X_test_pca = pca.transform(X_test)
	log = LogisticRegression()
	l.append(log)
	log.fit(X_train_pca, y_train)
	rf = RandomForestClassifier()
	l.append(rf)
	rf.fit(X_train_pca, y_train)
	gb = GradientBoostingClassifier()
	l.append(gb)
	gb.fit(X_train_pca, y_train)
	print('LogisticRegression accuracy:', log.score(X_test_pca, y_test))
	print('RandomForest accuracy:', rf.score(X_test_pca, y_test))
	print('GradientBoosting accuracy:', gb.score(X_test_pca, y_test))
	with open('training.pkl', 'wb') as f:
		pickle.dump(l, f)

if __name__ == '__main__':
	training()