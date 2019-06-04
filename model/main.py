import math
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from keras.utils.vis_utils import plot_model
from sklearn.metrics import mean_squared_error
from regressor import Regressor
from feature_extractor import FeatureExtractor

output_path = "output/"
_forecast_h = 24

def _read_data(path, dataset):
	try:
		Data = pd.read_csv(path + '/data/' + dataset + '.csv')
	except IOError:
		raise IOError("Data not found")

	y = np.empty((len(Data)))
	y[:] = np.nan
	# only one instant every 6 h,
	# so the forecast window is 'windowt' timesteps ahead
	windowt = _forecast_h / 6
	for i in range(len(Data)):
		if i + windowt >= len(Data):
			continue
		if Data['instant_t'][i + windowt] - Data['instant_t'][i] == windowt:
			y[i] = Data['windspeed'][i + windowt]
	X = Data
	i_toerase = []
	for i, yi in enumerate(y):
		if math.isnan(yi):
			i_toerase.append(i)
	X = X.drop(X.index[i_toerase])
	X.index = range(len(X))
	y = np.delete(y, i_toerase, axis=0)
	# y[y == 0] = 10
	return X, y

def plot_history(path, history, do_cv=False):
	keys = list(history.history.keys())
	plt.figure()
	plt.plot(history.history[keys[0]])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	if do_cv:
		plt.plot(history.history[keys[1]])
		plt.title('validation loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
	plt.savefig(path+"model_loss.png")
	plt.close()

def save_score(path, function, train_true, train_pred, test_true, test_pred):
	f = open(path + 'scores.txt', 'w+')
	f.write("Scores train: " + str(function(train_true, train_pred)) + "\n")
	f.write("Scores test: " + str(function(test_true, test_pred)) +  "\n")
	f.close()

if __name__=="__main__":
	do_cv = True
	do_feature_ext = False
	X_train, y_train = _read_data("..", "train")
	X_test, y_test = _read_data("..", "test")

	feature_ext = FeatureExtractor()
	feature_ext.fit(X_train, y_train)
	if do_feature_ext:
		X_array = feature_ext.transform(X_train)
		np.save("../data/train_norm", X_array[0])
		np.save("../data/train_scalar", X_array[1])
		X_array_test = feature_ext.transform(X_test)
		np.save("../data/test_norm", X_array_test[0])
		np.save("../data/test_scalar", X_array_test[1])
	else:
		X_array = [np.load("../data/train_norm.npy"), np.load("../data/train_scalar.npy")]
		X_array_test = [np.load("../data/test_norm.npy"), np.load("../data/test_scalar.npy")]
	model = Regressor(epochs = 350)
	history = model.fit(X_array, y_train, do_cv)
	pred_train = model.predict(X_array)
	pred_test = model.predict(X_array_test)

	plot_history(output_path, history, do_cv)
	save_score(output_path, lambda x,y: np.sqrt(mean_squared_error(x,y)), y_train, pred_train, y_test, pred_test)
