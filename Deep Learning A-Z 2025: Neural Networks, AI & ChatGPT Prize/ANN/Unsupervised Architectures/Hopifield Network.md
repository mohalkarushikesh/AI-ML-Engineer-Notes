## Hopifield Network 

-   A Hopfield network is a form of recurrent artificial neural network that serves as a content-addressable, auto-associative memory system. Invented by John Hopfield in 1982, it models human memory by storing patterns as "energy minima" and can recall or reconstruct full memories from partial or noisy inputs. [1, 2, 3, 4, 5, 6]  
Core Mechanics 

• Structure: It consists of a single layer of fully connected neurons where every neuron connects to every other neuron (except itself). 
• Symmetry: The connection weights are bidirectional and symmetric (i.e., the weight from Neuron A to Neuron B equals the weight from Neuron B to Neuron A). 
• Energy Function: The network possesses a defined "energy landscape". The stored memories represent local valleys (or minima) on this surface. 
• Retrieval: When given a corrupted or partial pattern, the network iteratively updates the states of its neurons to minimize its energy until it stabilizes in the nearest valley, successfully retrieving the original, intact memory. [2, 3, 7]  

How It Works 

1. Training (Memory Storage): Patterns are encoded using the Hebbian learning rule. The connection weights are calculated as the sum of the outer products of the patterns to be stored. 
2. State Updates: Neurons (typically having binary values like +1 and -1) are updated either synchronously or asynchronously. The new state of a neuron is determined by calculating the sum of its inputs from all other neurons multiplied by their respective weights. 
3. Convergence: The network continues to update iteratively, continually decreasing the overall "energy" of the system, until it settles into a stable state—representing the closest matching memory. [2, 7]  

Limitations and Advancements 

• Capacity: Classic discrete Hopfield networks have a limited memory storage capacity, roughly calculated as 0.15 × N (where N is the number of neurons). Overloading the network can result in "spurious states" or false memories. 
• Modern Hopfield Networks: Advanced continuous variations have been developed that overcome traditional capacity limitations. These modern versions are mathematically equivalent to the self-attention mechanisms used in contemporary transformer models. [2, 6, 8, 9]  

AI responses may include mistakes.

[1] https://en.wikipedia.org/wiki/Hopfield_network
[2] https://www.geeksforgeeks.org/machine-learning/hopfield-neural-network/
[3] https://www.doc.ic.ac.uk/~ae/papers/Hopfield-networks-15.pdf
[4] https://www.sciencedirect.com/topics/computer-science/hopfield-network
[5] https://www.sciencedirect.com/topics/mathematics/hopfield-network
[6] https://towardsdatascience.com/hopfield-networks-neural-memory-machines-4c94be821073/
[7] https://www.youtube.com/watch?v=1WPJdAW-sFo
[8] https://ml-jku.github.io/hopfield-layers/
[9] https://arxiv.org/abs/2008.02217

