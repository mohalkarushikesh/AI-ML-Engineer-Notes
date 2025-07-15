what are CNN ?
  
  Yann Lecun : founding fathers of deep learning

steps: 
  step 1: Convolution : Convolution is a general purpose filter effect for images. □ Is a matrix applied to an image and a mathematical operation.
    <img width="781" height="134" alt="image" src="https://github.com/user-attachments/assets/8b8991b0-c256-454a-88fb-4cb83afc2a55" />

    Convolution Setup in the Image
    Input Image: 7×7 matrix of binary values (0s and 1s)
    Kernel (Feature Detector): 3×3 matrix
    Output Feature Map: 5×5 matrix
    
    <img width="1006" height="441" alt="image" src="https://github.com/user-attachments/assets/c87b1a02-1f53-4617-91ba-a6274a50e009" />
  
    This process helps the neural network learn spatial hierarchies in images.
    Early layers might detect simple patterns (edges), while deeper layers detect complex features (faces, objects).

    sharpen 
      5 * 5 matrix 
        000000
        00-100
        0-15-10
        00-100
        00000
    blur 
    edge enhance 

      000
    -1-10
    000
    **edge detect** : important 
      010
      1-41
      010
    emboss 

    Additional Learning: Introduction to CNN by jianxin wu 2017 
    
  step 2: Max Pooling 
  step 3: Flattening 
  step 4: Full Connection 

filters 

feature detectors: kernel/ filter 
  useually 3 * 3 vector can be 5 * 5 or 7 * 7 

  stride: The kernel slides over the input image, one step (stride) at a time.
  
feature maps 

RLEU layer
  <img width="1093" height="458" alt="image" src="https://github.com/user-attachments/assets/30fade0c-a15d-4f35-ba15-d0127b0c0a26" />

  why breaking linearirty : 


Additional Learning : 
  1. Understanding CNN with mathematical model by jay kuo 2016 
  2. Delving deep into rectifiers : Surpassing Human Level performance on imagenet classification by kaiming he et al 2015 
Pooling layer 

Flattening 

Full Connection 

extrac topic: Softmax & Cross-Entropy

Additional Learning : Gradient based learning applied to Document Recognition by yann lecun 1998

