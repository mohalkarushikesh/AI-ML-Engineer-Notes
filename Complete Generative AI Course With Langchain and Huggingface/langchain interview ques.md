* not returning expected outputs: 

1. check the prompt 
2. verbose mode = True
3. Log the API calls
4. Test LLM separately 	

* irrelevant results : 

1. check the embeddings - openAIEmbedding or sentenceTransformer
2. inspect the vector search - retrive.similarity_search("query", k=3)
3. re-tune the chunking strategy - RecursiveCharaterTextSplitter, adjust chunk_size and chunk_overlap
4. re-rank the results : rerank=True, if using the hybrid retrieval method

* debug memory-related issues: 

1. inspect the stored messages : memory.buffer 
2. use diff memory type: 
	ConversationBufferMemory 
	conversationSummaryMemory 
	conversationTokenBufferMemory
3. token lims : 4096 for gpt4 turbo 
4. clear the memory : memory.clear()


