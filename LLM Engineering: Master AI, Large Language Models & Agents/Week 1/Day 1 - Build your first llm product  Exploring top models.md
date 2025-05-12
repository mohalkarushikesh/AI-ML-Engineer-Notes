day 1 : LLM Engineering
	
	install ollama
	ollama --version
	ollama run gpt3.2

1. clone the repo : https://github.com/ed-donner/llm_engineering
	git config --global http.sslVerify false

3. follow the readme recommended conda or python virtual env
	python -m venv venv
	venv/bin/activate	
	pip install -r requirements.txt
	launch jypyter lab
  or
  Download/Install Anaconda => setup
     conda env create -f environment.yml

4. create a openai key (free api key : https://github.com/dan1471/FREE-openai-api-keys) format: OPENAI_API_KEY=sk-proj-efghijkl5678mnopabcd1234efghijkl5678mnop
	open-ai >> settings (make payment) >> billings 
		dashborads >> API keys

5. create the .env file
	vim .env
  OPENAI_API_KEY=sk-proj-(paste-key-here)

6. activate your environment run jypyter lab
	jupyter lab





