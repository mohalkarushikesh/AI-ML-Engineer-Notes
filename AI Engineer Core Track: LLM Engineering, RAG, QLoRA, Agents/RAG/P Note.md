Ways to enhance the retrieval process
	- Query expansion: Query expansion can help capture a broader range of relevant documents by using multiple variations of the retrieval query. 
	- Re-ranking: After retrieving an initial set of chunks, apply additional ranking criteria (for example, sort by time) or a reranker model (such as mxbai-rerank and ColBERTv2) to re-order the results
	- Metadata filtering: Use metadata filters extracted from the query understanding step to narrow down the search space based on specific criteria. Metadata filters can include attributes like document type, creation date, author, or domain-specific tags. 


Precision formula: (how many releavant doc's) Precision measures “Of the chunks I retrieved, what % of these items are actually relevant to my user's query?” Computing precision does not require knowing all relevant items
	
<img width="665" height="114" alt="precision-formula-2667ca497a3e0573dddc19879d398617" src="https://github.com/user-attachments/assets/5302c5be-b2d5-45c6-8fcd-ddddac8eb21e" />

Recall forumla: (how many relevant doc's actually retrived) Recall measures “Of ALL the documents that I know are relevant to my user's query, what % did I retrieve a chunk from?”

<img width="665" height="114" alt="recall-formula-ba4cf87fb5bf78c139fbbcd45c66d2fe" src="https://github.com/user-attachments/assets/b5ef4e78-6d72-40aa-b114-71ad950d9fd3" />

<img width="947" height="494" alt="precision-recall-daigram-5aa1ff93361303c911134f8f80c5bf66" src="https://github.com/user-attachments/assets/497a24d9-a442-4ca7-9452-ffc8d74443d0" />
