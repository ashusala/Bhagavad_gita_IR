# need a list of documents (need not be pre-processed)
# need a list of pre-processed queries
def tf_idf_model (data, queries):
	# importing important libraries
	import pandas as pd
	from sklearn.feature_extraction.text import TfidfVectorizer

	# instantiating the model
	tfidf_vectorizer = TfidfVectorizer(use_idf=True)
	# applying in on data
	tfidf_vectors = tfidf_vectorizer.fit_transform(data)

	# feature_names
	feature_names = tfidf_vectorizer.get_feature_names()

	# putting it all together in a dataframe
	document_term_matrix = pd.DataFrame(tfidf_vectors.T.todense(), index = feature_names)

	# now let's move towars getting the answers for the queries
	f = open('best3_tfidf.txt','w+')

	for query in queries:
		# getting the common words between query and our features
		common = set(query.split()).intersection(set(feature_names))

		# getting the indices for top 3 documents
		ind = document_term_matrix.loc[common].apply(sum).sort_values(ascending = False).head(3).index.to_list()

		# printing the results in a file
		f.write('For query:\n' + '""' + query + '""' + '\n\n')
		f.write('The Best 3 matching verses are:\n')
		for ind, i in enumerate(ind):
			f.write(str(ind + 1) + ') ' + data[i] + '\n\n')
	
	# closing the file after the work is done 
	f.close()








