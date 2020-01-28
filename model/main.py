import pandas as pd
import models as md
import string

model_name = input('What model would you like to use?\n[tfidf, word2vec]: ')
queries = input('Enter the query file: ')

if model_name == 'tfidf':

	# getting the data
	import pandas as pd

	data = input('Please enter the data file: ')
	df = pd.read_csv('data/' + data)
	data = df['Translation'].to_list()

	# getting the queries
	df = pd.read_csv('data/queries.csv')
	queries = df['Verse'].to_list()
	# pre-processing the queries
	queries = [query.rstrip().lower().translate(str.maketrans('','',string.punctuation)) for query in queries]

	md.tf_idf_model(data, queries)

elif model_name == 'word2vec':

	# getting the data for training our model
	f_name = input('Please enter the data file to train the model: ')
	sample = open('data/' + f_name,'r')
	s = sample.read()
	data = s.replace('\n', ' ')

	# getting the verses data
	f_name = input('Please enter the file which contains the verses: ')
	verses = pd.read_csv('data/' + f_name, usecols=['#verse', 'Translation'])

	# getting the queries
	df = pd.read_csv('data/' + queries)

	md.word2vec_model(data, df, verses)




