testing
quantcap >>>


helohi! -grace



class GatherData():
	""" Pulls data from ThetaData and returns a dataframe """
	def get_historical_data():
		pass
	""" Pulls data from StockGeist and returns a dataframe """
	def get_sentiment_data():
		pass
	""" Pulls data from Tingo and returns a dataframe"""
	def get_fundamental_data():
		pass
	""" Pulls data from NASDAQ and returns a dataframe """
	def get_liquidity_data():
		pass
	""" Pulls data from Yahoo Calendar and returns a dataframe"""
	def get_earnings_data():
		pass
	""" Pulls data from 13F-Form Dataset and returns a dataframe"""
	def get_institutional_data():
		pass
	""" Parses through response and constructs and returns a dataframe"""
	def json_to_dtf():
		pass

class ProcessData():
	"""joins all of the datasets in an array of dataframes using the specified methods in the string array of methods. Methods include full, left, right, inner"""
	def join_all_datasets(arr, methods):
		pass

	""" handles missing NaN values using an approach passed in as a string variable named approach for dataset d. Approaches allowed include one-hot encoding, dropping all rows with NaN, etc."""
	def handle_missing_values(approach, dataset):
		pass
	""" Normalize features of dataframe x using sklearn methods using an approach passed in as a string variable approach"""
	def normalize_features(x, approach):
		pass
	
	""" Select target features of x, where the features are given in an array feature list"""
	def select_features(dataset, feature_list):
		pass
	""" Splits data into training, validation, and testing """ 
	def split_data(dataset, column, test_size, val_size):
		pass
	""" Deletes rows of data containing outliers above a certain threshold from dataset """
	def handle_outliers(dataset, threshold):
		pass

