import numpy as np

# function to get the word frequency of each word in a verse/sentence
# --- list_of_words: a pre-processed list of words in a verse/sentence.
def word_count(list_of_words):
    
    #dictionary keeping frequency for all the unique words
    counts = dict()
    
    for word in list_of_words:
        #if word already listed, increase its count
        if word in counts:
            counts[word] += 1
        # else putting the new word in dictionary
        else:
            counts[word] = 1

    return counts


#########################################################################
# function to calculate composite sentence/verse embeddings
# -- verse/query to be given in tokenzied and preprocessed format to func.
def senemb(alpha, verse_tok, model, dim):
    # create an empty vector
    sentEmb = np.zeros(dim)
    v_count = word_count(verse_tok)
    
    for word in verse_tok:
        prob = alpha / (alpha + v_count[word]/len(verse_tok))
        sentEmb = sentEmb +  model.wv[word]  * prob
        
    return sentEmb/len(verse_tok)