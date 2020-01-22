import pandas as pd
import models as md
import string

model_name = input('What model would you like to use?\n[tfidf, ...]: ')
data = input('Please enter the data file: ')
queries = input('Enter the query file: ')

if model_name == 'tfidf':

	# getting the data
	import pandas as pd
	df = pd.read_csv('data/' + data)
	data = df['Translation'].to_list()

	# getting the queries
	df = pd.read_csv('data/queries.csv')
	queries = df['Verse'].to_list()
	# pre-processing the queries
	queries = [query.rstrip().lower().translate(str.maketrans('','',string.punctuation)) for query in queries]

	md.tf_idf_model(data, queries)
