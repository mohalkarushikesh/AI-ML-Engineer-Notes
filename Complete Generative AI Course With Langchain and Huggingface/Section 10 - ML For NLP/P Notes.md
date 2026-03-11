Morphology: analysis of words into meaningful components 

lexemen : is about lexical analysis, namely about normalizing and disambiguating words

syntax: is related to linguistic theories about what makes some sentences well-formed and other's not.

In NLP, a lexeme is the abstract, fundamental unit of a word, representing its core meaning and all its inflected forms (like run, runs, ran, running all being forms of the lexeme RUN). It's the dictionary entry or base form (lemma) from which variations stem, helping computers group related words, understand grammatical functions, and process language beyond just individual characters, forming the basis for tasks like stemming and lemmatization.  

semantics : is about mapping of natural language sentences into domain representations

pragmatics: is about any nong-logical phenomena (ex. can you pass the salf or are you 18?)

discourse: is about texts, dialogues or multi-party conversations. 

Normalization : task of putting words in a stadard form (ex. how to match U.S.A to USA)

case folding : mapping everything to lowercase is an example of normalization 

Lemmatization : redues varients forms to their base forms. (ex. running ran to run)

Stemming: reduces terms to their stems. (ex. automatic, automation becomes automat)

Setence sengmentation: is a process of segmenting sentences in the running text. Question marks and exclamation points are relatively unambiguous, while periods are more ambiguous.

Levenshtein distance is a string metric used to measure the difference between two sequences. It quantifies the minimum number of single-character edits required to transform one string into another. 

Edit distance: How similar are the two strings, the minimum edit distance represents the minium no of edit operations needed to transform one string into another 
	these edit operations can be insertions, deletions, subtitutions
		for ex:  Intention
			 Execution

Spelling correction 
	non-word spelling correction: graffe => giraffe 
	real word spelling correction: dessert => desert
Noisy channel 
	
N-gram language models
