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

- API failure in langchain 
1. api exponential backoff : retry the api call with delays with  time.sleep() or tenacity library
2. async execution : reduce api load by making the cocurrent calls suing asynt def function with awaits 
3. monitor api status codes : we can capture the exception or ratelimiterror by logging and retrying 
4. cache results : store the cached results using redis or local to minimize the redundant api calls

from langchain.debug import set_debug 
set_debug(True)

- Extract text
PyPDFLoader or UnStructuredLoader

- compare a resume with a job description
convert both to the embeddings and use vector store like FAISS or pinecone to similarity search

- keyword extaction from resume
use LLMChain with prompt: Extract key skills, technologies, experience
alternatively use NLTK or spaCy

- handle diff resume format efficently
implement multiple document loaders - PyPDFLoader, UnStructuredLoader, DocxLoader and process each format accordinly 
