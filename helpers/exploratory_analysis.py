# Perform exploratory data analysis

def df_stats(df):
		"""
		Find the statistical attributes of the dataframe
		"""
		return df.describe()


def df_missing(df):
		"""
		Find missing value for all the columns
		"""
		return df.isnull().sum()
