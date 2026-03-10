LangGraph - python framework that helps you build ai agent as a graph where each note does a specific task 

Memory - ability of an AI agent to retain the information between the steps and conversations

Human in the loop - guide, approve and correct 

DAG (Direct Acylic Graph) - After the process of one process/step it goes to the next, but no repetation or looping back 
	- Sequential flow 
	- An AI agent that remembers the steps during it's work, so it can use that memory
	to make better decisions at each step
	
Components 
	Nodes : python funtions 
	Edges : connect the nodes 
	State : State schema serves as the input for all the nodes and endges
	State Graph : Structure of entire Gprah 

Diff ways we can create state schema 
	1. Using TypedDict - string notatations
	2. Using dataclass - dot notations 

12 the page 
