The core innovation vLLM introduces to solve AI memory bottlenecks is called PagedAttention.

Vllms  - Virtual LLM
	- implements continuous batching
	- PagedAttention breaks the KV cache into fixed-size "pages" or "blocks"

A Key-Value (KV) Cache is a fundamental optimization technique in Large Language Models (LLMs) that dramatically accelerates text generation by storing intermediate attention calculations.
Instead of re-evaluating past tokens from scratch for every new word, the model reuses pre-computed Key and Value states, turning slow, recursive computations into fast, constant-time operations

- Exploding gradients : At times, the steps grow excessively large, resulting in increased updates to weights and bias terms — to the point where the weights overflow (or become NaN, that is, Not a Number). An exploding gradient is the result of this, and it is an unstable method.

Vanishing gradient : On the other hand, if the steps are excessively small, it results in minor – even negligible – changes in the weights and bias terms. As a result, we may end up training a deep learning model with nearly identical weights and biases every time, never reaching the least error function. The vanishing gradient is what it's called.

* questions for experienced : 8th question



## Assembly AI 

Natural Language Processing (NLP) is a branch of Artificial Intelligence that gives computers the ability to read, understand, and generate human language


problem 		 high bias 				high variance 
		training performance is low 		validation performance is low 
		
cause 			underfitting 				overfitting 

solution		train more 				introduce more data 
		increase more complexity			use regularization 
		try diff model/architecture 			try diff model/architecture 

## classifications metrics:

accuracy  = correct instances/no of instance

precision  = TP / TP + FP

recall  = TP / TP + FN

F1 score = harmonic mean of precision and recall 

		     2 
	---------------------------
	(1/precision) + 1(1/recall)

pr curve - precision and recall curve 

AUC = Area under the curve  : should be as high as possible 

cross entropy - calculates distance between 2 probability distributions 
	binary cross entropy 
	categorical cross entropy
	sparse categorical cross entropy


## regression evalution metrics ; 

MAE - sum of all the errors but they are absolute values 

MSE - square of all the errors and then taking there mean - by doing that you're letting the greater error values to impact this mean square value 

RMSE - 

R2 : R square (coefficient of determination)
	- when values are best fitted with line - then the score will be 1 
	- does not fit - 0 
	- it is little bit fitting as well as not fitting - (the score will be between(0 and 1))

cosine similarity: very similar to classification metrics, but it is for regression problems it can deal with real values (tells us how similar two diff vectors are to each other) 

#### TRANSFORMERS - https://www.youtube.com/watch?v=_UVfwBqcnbM	
	
RNNs (can't do true parallelism because of recurrency + vanishing gradients → loss of long context)
→ LSTMs (improved RNNs with gates to better preserve long-term context) - but still fundamentally sequential/recurrent
→ Transformers (full parallelism via self-attention, no recurrence, excellent long-range context)

```
Embeddings - n length vectors (in transformers they are using 512 length vectors)

Normalization layers - normalize the output 

normal multi head attention - all the words are compared to every other word that are inputted that are in a sentence 

mask multi head attention - only the words coming are compared to that word in the sentence 

scaled dot product - 

input embeddings are multiple with q k v 
input 		i 								love 							you
embeddings
queries 	q1								q2								q3	
keys 		k1								k2								k3	
value 		(q1*k1 + q1*k2 + q1*k3)   (q2*k1 + q2*k2 + q2*k3)			(q3*k1 + q3*k2 + q3*k3)
score		
	/ 8 (devide by 8)
SoftMax 
multipy all value vectors with this weights (softmax)
			wv1		wv2
sum 		z1

output 		z (z1+z2+z3)

		z1.....z8

		concatenate z1...z8 * another weight matrix 

only one output of attention layer 
```

<img width="680" height="820" alt="multi_head_attention_flow" src="https://github.com/user-attachments/assets/d9bf52fd-0003-4afa-b3fb-515b44545638" />


Here's the full flow at a glance. A few key ideas to nail it down:

**Why ÷√8?** The dot products (q·k) grow large as dimension increases, which pushes softmax into regions with tiny gradients. Dividing by √d_model (e.g. √64 = 8) keeps the scores in a stable range.

**Why 8 heads?** Each head learns to attend to *different* relationships — one might track syntax, another coreference, another proximity. Running them in parallel is cheap because each head uses a smaller slice of the full dimension.

**The W₀ matrix** at the end projects the concatenated heads back to the model's original dimension, letting the network learn how to *combine* what each head found.

positional encodings
	1. learn positional encodings 
	2. fixed positional encodings	- recommended (they have advantage to being able to handle length of the sentences, we haven't seen in training set)

sine and cosine functions in diff frequencies 

## Transfer learning 

## Word embeddings 
- represent text into numbers 
- embeddings aim to represent the word in a dense vector, while making sure that similar words are close to each other in the embedding space 
	- one-hot encoding : 
	- count based representation : generally tries to sqeeze whole sentence into a one vector 
		- bag of words : how many times each word occurs and you create a vector to represent this 
		- n-gram : grups of n words and counts there occurance in sentence 
		- tf-idf 
				tf = no of times word occured / no of words in document 
				idf = log(no of documents/no of documents where this word occured)
	- embeddings
	- word to vec 
		- continuors bag of words 
		- skipgam 
	- glove - global vectors (it is an exention to word to vec)
	- fasttext : extention of word to vec algorithm (words very good at rare words/ not seen in training data)
	- elmo : our representation differs from the traditional word embeddings, in that each token is assigned a representation that is a function of the entire input sentence. 

# BERT : bidirection encoder representations from transformer

train on 2 diff task 
	1. masked language modeling 
	2. next sentence prediction 

Applications: 
	- sentiment analysis 
	- Question Answering
	- Named entity recognition 
	- text classification/summarization 


## classification - when we know the classes we want to predict and have a training data with true labels avaiable 

## clustering - when don't have any labels and want to find unknown labels 

## Normalization: Collapse input to be between 0 and 1 

## Stadardization: change their values so that their mean is 0 and variance equals to 1 

## Regularization - to overcome the overfitting (high variance i.e validation performance is low)


## Batch Noralization: Insted of only normalizing our inputs then feeding the data into our network we normalize all the outputs/ all the layers in our network - normalize the data, and feed output of prev layer to the next layer.




