from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
import json

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
	models = {
		'LogisticRegression': LogisticRegression(),
		'RandomForest': RandomForestClassifier(),
		'GradientBoosting': GradientBoostingClassifier(),
	}
	metrics = {}
	for name, model in models.items():
		model.fit(X_train_pca, y_train)
		l.append(model)
		acc = model.score(X_test_pca, y_test)
		metrics[name] = acc
		print(f'{name} accuracy: {acc:.4f}')
	with open('training.pkl', 'wb') as f:
		pickle.dump(l, f)
	with open('metrics.json', 'w') as f:
		json.dump(metrics, f, indent=2)
	print('Metrics saved to metrics.json')

if __name__ == '__main__':
	training()