#svm classifier 
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR

dates = []
prices = []

def get_data(filename):
	with open(filename,'r') as csvfile:
		csvFileReader = csv.reader(csvfile)
		next(csvFileReader)
		for row in csvFileReader:
			dates.append(int(row[0].split('-')[0]))
			prices.append(float(row[1]))
		return

def predict_price(dates, prices, x):
	dates = np.reshape(dates,(len(dates) , 1))
	lin_clf = SVR(kernel = 'linear' , C = 1e3)
	poly_clf = SVR(kernel = 'poly' , C = 1e3 , degree = 2)
	rbf_clf = SVR(kernel = 'rbf' , C = 1e3 , gamma = 0.1)
	lin_clf.fit(dates, prices)
    #poly_clf.fit(dates, prices)
	rbf_clf.fit(dates, prices)

	plt.scatter(dates, prices, color = 'black' , label = 'data')
	plt.plot(dates, lin_clf.predict(dates), color = 'red' , label = 'LINEAR')
	plt.plot(dates, rbf_clf.predict(dates), color = 'green' , label = 'RBF')
	#plt.plot(dates, poly_clf.predict(dates), color = 'blue' , label = 'POLYNOMIAL')
	plt.xlabel('Date')
	plt.ylabel('Price')
	plt.title('Stock Price Destribution')
	plt.legend()
	plt.show()

	return lin_clf.predict(x)[0] , rbf_clf.predict(x)[0] 

get_data('ttm.csv')#tata motors data
print predict_price(dates, prices, 20) 






