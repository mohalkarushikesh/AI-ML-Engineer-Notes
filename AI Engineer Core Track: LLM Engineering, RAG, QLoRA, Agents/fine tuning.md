fine tuning : 

	sft : 
		1. full fine tune 
		2. prompt-response pairs
		3. partial fine tune 
	peft : 
		1. LoRA : (Low-Rank Adaptation): Instead of altering all model weights, LoRA decomposes the weight updates into two smaller, low-rank matrices. This makes training incredibly fast and much cheaper to run.
		2. QLoRA (Quantized LoRA): A memory-efficient variant of LoRA that compresses the base model into 4-bit precision. This allows massive models (like 70B+ parameters) to be fine-tuned on a single consumer-grade GPU.
		3. 
		
