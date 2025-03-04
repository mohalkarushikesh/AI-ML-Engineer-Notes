Tensor Transposition
- transpose of the scalar is itself ex: xT=x
- transpose of vector converts column to row
- matrix transposition: flip of axes over main diagonal
![image](https://github.com/user-attachments/assets/a897dcd1-a14b-472a-a854-3d7a73399041)
- Hadamard product/ element-wise product: if the two tensors are same size, operations are often by default applied element-wise this is not matrix multiplication
  ![image](https://github.com/user-attachments/assets/ee903636-78cf-4b3b-9b4d-876dadab1a0a)

Basic Tensor Arithmatics
Reduction: calculating sum across all the elements of a tensor is common operation 
for ex: 
  for vector x of length n we calculate 
  for matrix X with m by n dimentions we calculate 
  0 - rows 
  1 - columns 
  all the operations can be applied with reduction along all or a selection of axes (max, min, mean, product)
The Dot Product: 
  two vectors with same length let's say x.y
 the dot product is uniquitous in deep learning it is performed at the every artificial neuron in a deep neural network which may be made up of millions (or orders of magnitude more)of these neurons 
Solving Linear Systems
