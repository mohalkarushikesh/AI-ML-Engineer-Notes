Linear algebra 2 : matrix application

seesion : use tensors  in python and solve the system of equations and indentity meaninful patterns in data 

1. review introductory linear algebra
## 1. What is Linear Algebra?
Linear algebra is the branch of mathematics that focuses on solving for unknowns within systems of **linear equations**. It deals with concepts such as **vectors, matrices, determinants, eigenvalues**, and **eigenvectors**, which are essential in many computational fields.

## 2. Modern Applications of Linear Algebra

### a. Solving for Unknowns in Machine Learning Algorithms
Linear algebra plays a crucial role in **machine learning** and **deep learning**, where it helps in solving optimization problems, performing **matrix operations**, and manipulating high-dimensional data.

### b. Dimensionality Reduction
Dimensionality reduction techniques like **Principal Component Analysis (PCA)** leverage linear algebra to transform high-dimensional data into a lower-dimensional space while preserving important information.

### c. Ranking Results
Ranking algorithms rely on **eigenvector calculations**, such as those used in **Google's PageRank algorithm** (Saaty & Hu, 1998). This helps in determining the importance of webpages in search engine results.

### d. Recommender Systems
Recommendation systems use **Singular Value Decomposition (SVD)** to break down matrices and identify patterns in user preferences, enabling personalized recommendations.

### e. Natural Language Processing (NLP)
Linear algebra is fundamental in NLP, with applications in **matrix factorization** techniques for various tasks, including:

- **Topic Modeling**: Techniques like Latent Semantic Analysis (LSA) use SVD to extract hidden topics in textual data.
- **Semantic Analysis**: Matrix factorization helps in capturing relationships between words, improving language understanding in AI models.

  matrix inversion: aovids overdetermination, underdetermination, no solutions, infinite solutions
4. Eigen decomposition
  1. applying matrices
  2. affine transformations
### What are Affine Transformations?
Affine transformations are a set of functions that preserve points, straight lines, and planes. These transformations also maintain the relative proportions of figures, though angles and lengths might not always stay the same. In simpler terms, affine transformations ensure that "straightness" and "parallelism" are preserved.

### Types of Affine Transformations
Affine transformations include several familiar operations:
1. **Translation:** Shifting an object from one location to another (like moving an image across a screen).
2. **Scaling:** Enlarging or shrinking an object while keeping its shape proportional.
3. **Rotation:** Rotating an object around a fixed point or axis.
4. **Reflection:** Flipping an object over a line or plane (like a mirror image).
5. **Shear:** Skewing an object in one direction, like tilting a rectangle into a parallelogram.

### Mathematical Representation
Affine transformations can be represented using matrices. In 2D, it often looks like this:

![image](https://github.com/user-attachments/assets/7397e3eb-00b5-4206-bb8e-ab94d9876219)


Here:
- (a, b, c, d): Define how the shape is scaled, rotated, or sheared.
- (e, f): Represent translation (shifting the object).

In 3D, affine transformations use 4x4 matrices and include operations in the \(z\)-axis.

### Applications
Affine transformations are widely used in:
- **Computer Graphics:** For scaling, rotating, and translating images or 3D models.
- **Image Processing:** Aligning or transforming images.
- **Robotics:** Mapping movements in space.
- **Machine Learning:** Working with data transformations in certain algorithms.

    check blog post: affine transformations in python, including how apply them on images as well as vectors

  4. Eigenvector: A vector that doesn't change its direction during a transformation, only its magnitude (scaled by the eigenvalue).
  - eigen in german typical in english translation: charateristics vector
  5. Eigenvalue: A scalar value that shows how much a vector is stretched or squished during a transformation.
  - is scalar that simply scales the eigen vector v 
![image](https://github.com/user-attachments/assets/8b35dd4c-0fda-46eb-a9a2-24d6d58d264c)

![{C8E1DBA3-154D-4BA9-8983-ECA8D7337540}](https://github.com/user-attachments/assets/4dc954d5-99a4-4978-a347-55aa11fb4da7)


  6. matrix determinants
  7. matrix decomposition
  8. applications of eigen decomposition
9. matrix operations for machine learning
10. 


