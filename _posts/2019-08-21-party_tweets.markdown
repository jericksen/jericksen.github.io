---
layout: post
title:      "Party Tweets"
date:       2019-08-21 15:16:41 +0000
permalink:  party_tweets
---


This blog was published on Medium. Visit the original post using the link below:

https://medium.com/@jericksen20/party-tweets-91011b85feb6

---

It wasn't easy. Attempting to predict the political affiliation of a tweet from a sitting U.S. representative in Congress was harder than it originally seemed. I thought it was simply a matter of converting the text contained within a tweet to integer representations and running the matrices through an embedding layer and recurrent neural network and checking the results. Easy, right? 
Not so. This project was my first attempt at Natural Language Processing. The required steps involved sourcing the text data, cleaning the text data (in itself a difficult task as tweets contain a wide variety of characters, links, punctuation, et cetera), tokenizing the text data, splitting the predictor from the target data, padding the word matrices, vectorizing the word data, and finally, training a neural network to make accurate party affiliation predictions. 
There's a lot going on in a project like this. Not surprisingly, as a first attempt, it was quite challenging. Not only does working with neural networks involve a lot of moving pieces, but the conceptual grasping of embedding text data in multidimensional vector space can be taxing. For the remainder of this post, I'm going to focus on the steps used to do just that, vectorizing text data. 


---

The interesting thing about word vectorization is the results. Once our words are embedded in a vector space, we can do some interesting things. Below I'll briefly walk through the steps used for vectorizing word data followed by some cool analysis. 
Once we have our cleaned text data, we start with tokenizing the text data:
dem_tokenized_text = [text.split(" ") for text in democrat_list]
Followed by creating our word vectors using 30 dimensions using Word2Vec: 
dem_embedding = Word2Vec(sentences = dem_tokenized_text, size=30)
With our words in vector form, we'll take a look at the first word's vector-matrix:
X_dem = dem_embedding[dem_embedding.wv.vocab]
X_dem[:1]
Which displays an output array as: 
With these vectors, it's difficult to produce any meaningful analysis of the words. With Principal Component Analysis, we can reduce the number of dimensions to two, which would allow of some graph plotting:
pca = PCA(n_components=2)
X_dem_fitted = pca.fit_transform(X_dem)
With our two-dimensional word vectors, we'll build a function that plots distinct words in a vector space in relation to other words used in the tweet text that house a semantic relationship to the target word. Before building a function, we first need to generate a list of all words in the democrat's tweet vocab: 
words_dem = list(dem_embedding.wv.vocab)
Now we'll build the function: 
def dem_similar_words(word):
    
    words_to_plot = [word] + [w[0] for w
               in dem_embedding.similar_by_word(word,10)]     
    indices = [words_dem.index(word_to_plot)
               for word_to_plot in words_to_plot]
   
    plt.figure(figsize=(8,6))
    
    for index, word in zip(indices, words_to_plot):
        
        vec = X_dem_fitted[index]
        
        plt.scatter(vec[0], vec[1], 
                    c = 'mediumpurple', s = 200, alpha=0.5)
        plt.annotate(word, 
                     vec + [0.15, 0], fontsize = 14)
    print('Similar words with positive semantic relationship:')
    display(words_to_plot)
    plt.show()
With our word vectors, word list and function established, we can plot the 10 most similar words to any target word. We'll call the function using the word 'tax':
dem_similar_words('tax')
Which produces the following output: 
The graphic above shows the 10 words used within the democrat's total tweet corpus with the closest semantic relationship to our target word 'tax'. The data collected for this project occurred in May 2018. At that time, congress was grappling with the GOP tax reform bill. Perhaps unsurprisingly, many of the words used by democrats in relation to the word tax produce an unsavory characterization of the tax reform bill, i.e., 'goptaxscam', 'deficit', 'explode', 'wealthy', and 'break'. Clearly, democrats were trying to paint the reform act in a negative tone. 


---

I hope this post was informative to some and interesting to others. Among the many facets of NLP, I find analyzing word vectors to be among the most entertaining and insightful of steps.
