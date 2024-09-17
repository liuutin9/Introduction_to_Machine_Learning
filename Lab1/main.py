import numpy as np
import pandas as pd
import csv
import math
import random
import os

training_dataroot = 'lab1_advanced_training.csv' # Training data file file named as 'lab1_advanced_training.csv'
testing_dataroot = 'lab1_advanced_testing.csv'   # Testing data file named as 'lab1_advanced_testing.csv'
output_dataroot = 'lab1_advanced.csv' # Output file will be named as 'lab1_advanced.csv'

training_datalist =  [] # Training datalist, saved as numpy array
testing_datalist =  [] # Testing datalist, saved as numpy array

output_datalist =  [] # Your prediction, should be a list with 3000 elements\

dataType = {'age': 0, 'gender': 1, 'height': 2, 'weight': 3, 'bodyFat': 4, 'diastolic': 5, 'systolic': 6, 'gripForce': 7}

# Read input csv to datalist
with open(training_dataroot, newline='') as csvfile:
	training_datalist = pd.read_csv(training_dataroot).to_numpy()

with open(testing_dataroot, newline='') as csvfile:
	testing_datalist = pd.read_csv(testing_dataroot).to_numpy()

def SplitData(data, split_ratio):
	"""
	Splits the given dataset into training and validation sets based on the specified split ratio.

	Parameters:
	- data (numpy.ndarray): The dataset to be split. It is expected to be a 2D array where each row represents a data point and each column represents a feature.
	- split_ratio (float): The ratio of the data to be used for training. For example, a value of 0.8 means 80% of the data will be used for training and the remaining 20% for validation.

	Returns:
	- training_data (numpy.ndarray): The portion of the dataset used for training.
	- validation_data (numpy.ndarray): The portion of the dataset used for validation.

	"""
	training_data = []
	validation_data = []

	# TODO
	data_length = len(data)
	training_data_length = math.floor(data_length * split_ratio)
	random_index = random.sample(range(0, data_length), data_length)
	training_data_index = random_index[:training_data_length]
	validation_data_index = random_index[training_data_length + 1:]
	for idx in training_data_index:
			training_data.append(data[idx])
	for idx in validation_data_index:
			validation_data.append(data[idx])

	training_data = np.array(training_data)
	validation_data = np.array(validation_data)

	return training_data, validation_data

def PreprocessDataAdvanced(data, x_type):
	"""Preprocess the given dataset and return the result.

	Args:
			data (numpy.ndarray): The dataset to preprocess. It is expected to be a 2D array where each row represents a data point and each column represents a feature.
			x_type (str): The type of attribute

	Returns:
			- preprocessedData (numpy.ndarray): Preprocessed data.
	"""
	preprocessedData = []
	x_dataType = dataType[x_type]
	y_dataType = dataType['gripForce']

	# TODO
	noNanData = []
	for row in data:
			isNan = np.isnan(row[x_dataType]) or np.isnan(row[y_dataType]) 
			if not isNan:
					noNanData.append(row)

	noNanData = np.array(noNanData)

	x_data = noNanData[:, x_dataType]
	y_data = noNanData[:, y_dataType]

	x_q1 = np.percentile(x_data, 25)
	x_q3 = np.percentile(x_data, 75)
	x_qd = x_q3 - x_q1
	y_q1 = np.percentile(y_data, 25)
	y_q3 = np.percentile(y_data, 75)
	y_qd = y_q3 - y_q1

	x_upper_bound = x_q3 + 1.5 * x_qd
	x_lower_bound = x_q1 - 1.5 * x_qd
	y_upper_bound = y_q3 + 1.5 * y_qd
	y_lower_bound = y_q1 - 1.5 * y_qd

	for row in range(len(noNanData)):
			x_out_of_bound = x_data[row] > x_upper_bound or x_data[row] < x_lower_bound
			y_out_of_bound = y_data[row] > y_upper_bound or y_data[row] < y_lower_bound
			is_error_data = x_out_of_bound or y_out_of_bound

			if (noNanData[row][1] != 'nan' and (not is_error_data)):
					preprocessedData.append(noNanData[row])

	preprocessedData = np.array(preprocessedData)

	return preprocessedData

def SplitMaleAndFemale(data):
	"""Split the data by gender

	Args:
			data (2Darray): row data

	Returns:
			maleData,femaleData (2Dnumpy.array, 2Dnumpy.array): maleData, femaleData 
	"""
	maleData = []
	femaleData = []

	for row in data:
			if row[1] == 'M':
					row[1] = 0
					maleData.append(row)
			elif row[1] == 'F':
					row[1] = 0
					femaleData.append(row)

	maleData = np.array(maleData)
	femaleData = np.array(femaleData)

	return maleData, femaleData

def RegressionAdvanced(dataset, degree, num_attributes):
	"""
	Performs regression on the given dataset and return the coefficients.

	Parameters:
	- dataset (numpy.ndarray): A 2D array where each row represents a data point.

	Returns:
	- w (numpy.ndarray): The coefficients of the regression model. For example, y = w[0] + w[1] * x + w[2] * x^2 + ...
	"""

	X = []
	for i in range(num_attributes):
			X.append(dataset[:, i:i+1])
	y = dataset[:, num_attributes]

	# Add polynomial features to X
	X_poly = []
	for i in range(num_attributes):
			X_poly_item = np.ones((X[i].shape[0], 1))
			for d in range(1, degree + 1):
					X_poly_item = np.hstack((X_poly_item, X[i] ** d))
			X_poly.append(X_poly_item)

	# Initialize coefficients (weights) to zero
	num_dimensions = X_poly[0].shape[1]  # Number of features (including intercept and polynomial terms)
	w = [] # w = [w_age, w_gender, w_height, w_weight, w_bodyFat, w_diastolic, w_systolic]
	for i in range(num_attributes):
			w.append(np.zeros(num_dimensions))

	# TODO: Set hyperparameters
	num_iteration = 10000000
	learning_rate = 0.000000001

	# Gradient Descent
	m = len(y)  # Number of data points
	for iteration in range(num_iteration):
			# TODO: Prediction using current weights and compute error
			predicted_y_vec = []
			for i in range(num_attributes):
					predicted_y_vec.append(X_poly[i].dot(w[i]))

			predicted_y = predicted_y_vec[0]
			for i in range(1, num_attributes):
					predicted_y += predicted_y_vec[i]

			err = (y - predicted_y)

			# TODO: Compute gradient
			g = [] # g = [g_age, g_gender, g_height, g_weight, g_bodyFat, g_diastolic, g_systolic]
			for i in range(num_attributes):
					g.append(-2 * X_poly[i].T.dot(err))

			# TODO: Update the weights
			for row in range(num_attributes):
					w[row] = w[row] - (learning_rate) * g[row]

			# TODO: Optionally, print the cost every 100 iterations
			if iteration % 100 == 0:
					cost = np.sum((1 / m) * abs(err / y))
					os.system('cls')
					print(f"Iteration {iteration}, Cost: {cost}")


	w = np.array(w)

	return w

def MakePredictionAdvanced(w_male, w_female, test_dataset, num_attributes):
	"""
	Predicts the output for a given test dataset using a regression model.

	Parameters:
	- w_male (numpy.ndarray): The coefficients of the model (male), where each element corresponds to
														 a coefficient for the respective power of the independent variable.
	- w_female (numpy.ndarray): The coefficients of the model (female), where each element corresponds to
														 a coefficient for the respective power of the independent variable.
	- test_dataset (numpy.ndarray): A 1D array containing the input values (independent variable)
																				for which predictions are to be made.
	- num_attributes (int): The number of the types of x

	Returns:
	- list/numpy.ndarray: A list or 1d array of predicted values corresponding to each input value in the test dataset.
	"""
	test_dataset_x = test_dataset[:, :num_attributes]
	degree = w_male.shape[1]
	prediction = []
	w_male_coef = []
	w_female_coef = []

	for i in range(degree):
			w_male_coef.append(w_male[:, i:i+1])
			w_female_coef.append(w_female[:, i:i+1])

	y = []

	for row in range(len(test_dataset_x)):
			if test_dataset_x[row][1] == 'M':
					test_dataset_x[row][1] = 0
					tmp_y = np.sum(w_male_coef[0])
					for i in range(1, degree):
							tmp_y += test_dataset_x[row].dot(w_male_coef[i])[0]
					y.append(tmp_y)
			else:
					test_dataset_x[row][1] = 0
					tmp_y = np.sum(w_female_coef[0])
					for i in range(1, degree):
							tmp_y += test_dataset_x[row].dot(w_female_coef[i])[0]
					y.append(tmp_y)

	prediction = np.array(y)
	return prediction

# (1) Split data
training_dataset, validation_dataset = SplitData(training_datalist, 0.8)

# (2) Preprocess data
training_dataset_age = PreprocessDataAdvanced(training_dataset, 'age')
validation_dataset_age = PreprocessDataAdvanced(validation_dataset, 'age')
training_dataset_height = PreprocessDataAdvanced(training_dataset_age, 'height')
validation_dataset_height = PreprocessDataAdvanced(validation_dataset_age, 'height')
training_dataset_weight = PreprocessDataAdvanced(training_dataset_height, 'weight')
validation_dataset_weight = PreprocessDataAdvanced(validation_dataset_height, 'weight')
training_dataset_bodyFat = PreprocessDataAdvanced(training_dataset_weight, 'bodyFat')
validation_dataset_bodyFat = PreprocessDataAdvanced(validation_dataset_weight, 'bodyFat')
training_dataset_diastolic = PreprocessDataAdvanced(training_dataset_bodyFat, 'diastolic')
validation_dataset_diastolic = PreprocessDataAdvanced(validation_dataset_bodyFat, 'diastolic')
training_dataset = PreprocessDataAdvanced(training_dataset_diastolic, 'systolic')
validation_dataset = PreprocessDataAdvanced(validation_dataset_diastolic, 'systolic')

training_dataset_male, training_dataset_female = SplitMaleAndFemale(training_dataset)

# (3) Train regression model
w_male = RegressionAdvanced(training_dataset_male, 1, 7)
w_female = RegressionAdvanced(training_dataset_female, 1, 7)

# (4) Predict validation dataset's answer, calculate MAPE comparing to the ground truth
validation_predict_datalist = MakePredictionAdvanced(w_male, w_female, validation_dataset, 7)
mape = 0
for i in range(len(validation_dataset)):
		mape += (1 / len(validation_dataset)) * abs((validation_dataset[i][-1] - validation_predict_datalist[i]) / validation_dataset[i][-1])
print(f"mape = {mape}")

# (5) Make prediction of testing dataset and store the values in output_datalist
output_datalist = MakePredictionAdvanced(w_male, w_female, testing_datalist, 7)

# Assume that output_datalist is a list (or 1d array) with length = 3000
with open(output_dataroot, 'w', newline='', encoding="utf-8") as csvfile:
	writer = csv.writer(csvfile)
	writer.writerow(['Id', 'gripForce'])
	for i in range(len(output_datalist)):
		writer.writerow([i,output_datalist[i]])