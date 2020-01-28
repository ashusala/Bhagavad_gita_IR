from tools import *
import pandas as pd
import numpy as np

# --- need a list of documents (need not be pre-processed)
# --- need a list of pre-processed queries
def tf_idf_model (data, queries):
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

#######################################################################################################
def word2vec_model(data_, queries_, verse_data_df):
	### import important libraries ----------------------
	from nltk import sent_tokenize, word_tokenize
	from gensim.models import Word2Vec 
	from numpy.linalg import norm
	from numpy import dot
	import string
	import gensim

	### preparing data for Word2Vec model --------------------
	data = []

	# iterate through each sentence in the file
	for i in sent_tokenize(data_):
	    temp = []

	    # tokenize the setence into words
	    for j in word_tokenize(i):
	        temp.append(j)
	        
	    data.append(temp)

    ### training the model --------------------
	model = gensim.models.Word2Vec(data, min_count = 1, size = 100, window = 3)

	### Calculating sentence embedding for verses ----------------

	verse_data = verse_data_df['Translation'].to_list()

	#list to store all the vectors
	verse_vectors = []

	# loop over all the verses
	for verse in verse_data:
	    # tokenize and preprocess the verse
	    verse_tok = word_tokenize(verse.lower().translate(str.maketrans('','',string.punctuation)))
	    verse_vectors.append(senemb(1, verse_tok, model, 100))   

    ### Here we calculate the sentence embedding vectors for queries ---------------

	# getting the queries
	queries_ = queries_['Verse'].to_list()

	# pre-processing the queries
	queries = [query.lower().translate(str.maketrans('','',string.punctuation)).split() for query in queries_]

	# a list to store all the vectors
	query_vectors = []

	for query in queries:
	    query_vectors.append(senemb(1, query, model, 100)) 

    ### Calculating the scores ---------------------------------

    # queries_scores: to store cos. sim. of each query with all the verses. [basically a 2d list]
	queries_scores = []

	# looping over all the queries
	for q_vec in query_vectors:
	    query_scores = []
	    # storing the scores for a particular query
	    for v_vec in verse_vectors:
	        # getting the score
	        score = (dot(q_vec , v_vec))/(norm(q_vec)*norm(v_vec))
	        query_scores.append(score)
	        
	    queries_scores.append(query_scores)


    ### now let's move towars getting the answers for the queries -------------------

    # opening a file to output all our answers
	f = open('best3_word2vec.txt','w+')

	# top3_verseNo: to store verse nos. of top three verses for each query
	# to be put as results in mAP fun.
	top3_verseNo = []

	# looping over 
	for ind, i in enumerate(queries_scores):
	    temp = [] # to store the top 3 verses 
	    indices = [] # to store the top 3 indices
	    
	    #take three elements with highest scores   
	    for ele in sorted(i)[-1:-4:-1]:
	        #top_result = verses.iloc[i.index(ele)]
	        indices.append(i.index(ele))
	        temp.append(verse_data_df.iloc[i.index(ele)]['#verse'].replace(":",""))
	        
	    # printing the results in a file
	    f.write('For query:\n' + '""' + queries_[ind] + '""' + '\n\n')
	    f.write('The Best 3 matching verses are:\n')
	    for ind, i in enumerate(indices):
	        f.write(str(ind + 1) + ') ' + verse_data_df.iloc[i]['#verse'] + verse_data_df.iloc[i]['Translation'] + '\n\n')

	    top3_verseNo.append(temp)


