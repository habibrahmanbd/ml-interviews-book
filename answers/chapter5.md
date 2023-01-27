## 5.1.1 Vectors
If some characters seem to be missing, it's because MathJax is not loaded correctly. Refreshing the page should fix it.

1. Inner Product / Dot product

    i. **[E] What’s the geometric interpretation of the dot product of two vectors?**
    
    **Answer:**
    The dot product (also known as the scalar product or inner product) of two vectors is a scalar value that represents the cosine of the angle between the two vectors. The dot product is defined as:
    
    v1.v2 = ||v1|| * ||v2|| * cos(θ)
    
    where ||v1|| and ||v2|| are the magnitudes (lengths) of the two vectors, and θ is the angle between them.
    
    Geometrically, the dot product can be interpreted as the projection of one vector onto another. The dot product of two vectors can be thought of as the length of the projection of one vector onto the other, multiplied by the cosine of the angle between them.
    
    For example, if two vectors are perpendicular to each other, the angle between them is 90 degrees, and the cosine of this angle is 0. This means that the dot product will be zero, which corresponds to the fact that there is no projection of one vector onto the other. On the other hand, if two vectors are parallel to each other, the angle between them is 0 degrees, and the cosine of this angle is 1. This means that the dot product will be the product of the magnitudes of the two vectors, which corresponds to the fact that one vector is a scaled version of the other.
    
    ii. **[E] Given a vector , find vector of unit length such that the dot product of and is maximum.**
    
    **Answer:**
    Given a non-zero vector u, the vector v of unit length such that the dot product of u and v is maximum is given by:
	   $\frac{u}{||u||}$
	   
	where ||u|| is the length of vector u. This is because the dot product of two vectors is maximized when the two vectors are pointing in the same direction. Dividing u by its length will give a vector in the same direction with unit length, and therefore the dot product with this vector will be the maximum possible value which is equal to the length of u.


2. #### Outer product
    1. [E] Given two vectors $$a = [3, 2, 1]$$ and  $$b = [-1, 0, 1]$$. Calculate the outer product $$a^Tb$$?    
    **Answer:**
       The outer product of two vectors a and b is calculated as the matrix product of a transpose of a with b. In this case, the outer product of vectors a and b is:
       
       
    
    ii. [M] Give an example of how the outer product can be useful in ML.
    
    **Answer:**
    One example of how the outer product can be useful in machine learning is in Principal Component Analysis (PCA). PCA is a technique used to reduce the dimensionality of a dataset by identifying the directions of maximum variance in the data and projecting the data onto those directions. The outer product of two vectors can be used to calculate the projection of one vector onto another.

    In PCA, the data is represented as a matrix X where each row is a datapoint and each column is a feature. The goal of PCA is to find a new set of coordinates for the data, where the new coordinates are linear combinations of the original features. These new coordinates, or principal components, are chosen such that they explain the most variance in the data.

    The first step in PCA is to calculate the covariance matrix of the data, which is the outer product of the data matrix with its transpose. This outer product can be represented by X.X^T.

    ```python
    import numpy as np

    # Define data matrix X
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])

    # Calculate the covariance matrix
    cov_matrix = np.cov(X.T)

    print("Covariance matrix: \n", cov_matrix)
    
    # Find the eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    print("Eigenvalues: ", eigenvalues)
    print("Eigenvectors: \n", eigenvectors)
    ```
    The eigenvectors of the covariance matrix are the principal components and the eigenvalues are the variances along those directions. The eigenvectors are the directions of maximum variance in the data. By projecting the data onto these directions, we can reduce the dimensionality of the data while still preserving most of the variance.

    In this way, the outer product is used to calculate the covariance matrix, which plays an important role in PCA and helps to reduce the dimensionality of the data.


### Inner and Outer product:
```python
  1 import numpy as np
  2
  3 # Define two vectors
  4 a = np.array([3,2,1])
  5 b = np.array([-1,0,1])
  6
  7 # Calculate the inner product (dot product) of the two vectors
  8 inner_product = np.dot(a, b)
  9
 10 # Calculate the outer product of the two vectors
 11 outer_product = np.outer(a, b)
 12
 13 print("Inner Product of a and b: ", inner_product)
 14 print("Outer Product of a and b: \n", outer_product)
```
3. [E] What does it mean for two vectors to be linearly independent?

    **Answer:**
    Two vectors are considered to be linearly independent if neither of them can be represented as a linear combination of the other. In other words, if the only way to express one vector as a linear combination of the other is with a coefficient of 0, then the two vectors are linearly independent.

    For example, let's consider two vectors a and b:
    ```
    a = [1, 2, 3]
    b = [4, 5, 6]
    ```

    These two vectors are linearly independent because neither of them can be expressed as a linear combination of the other. We can't find any scalar coefficients (x,y) such that $a = xb + yc$ for any vector c.

    On the other hand, if two vectors can be expressed as a linear combination of each other, then they are considered to be linearly dependent. For example,
    ```
    a = [1, 2, 3]
    b = [2, 4, 6]
    ```
    These two vectors are linearly dependent, because the vector b can be expressed as a linear combination of the vector a, $b = 2a$.

    Linearly independent vectors have the property that they span a subspace of a vector space that has a dimension equal to the number of vectors that span it, while linearly dependent vectors can be linearly represented by a combination of other vectors and so they do not add new information to the space.

4. [M] Given two sets of vectors $$A = {a_1, a_2, a_3, ..., a_n}$$ and $$B = {b_1, b_2, b_3, ... , b_m}$$. How do you check that they share the same basis?

    **Answer:**
        To check mathematically that two sets of vectors, A and B, share the same basis, we can use the following methods:
            - Check if the vectors in A can be written as a linear combination of the vectors in B, and vice versa. If this is possible, then A and B share the same basis.

6. [M] Given $$n$$ vectors, each of $$d$$ dimensions. What is the dimension of their span?

    **Answer:**
        Given n vectors, each of d dimensions, the dimension of their span is the rank of the matrix formed by the concatenation of these vectors. The rank of a matrix is defined as the dimension of the vector space spanned by its columns. So, the dimension of the span of these n vectors is equal to the rank of the matrix formed by these vectors.

    The rank of a matrix can be found using linear algebra techniques like Gaussian elimination, it's also the number of non-zero rows after row-reducing the matrix.

    The dimension of the span of these n vectors can range from 0 to d. If none of the vectors in the set is linearly independent, the dimension of the span will be 0. If all the vectors in the set are linearly independent, the dimension of the span will be equal to d. If some of the vectors in the set are linearly dependent, the dimension of the span will be less than d.
    
8. Norms and metrics
	1. [E] What's a norm? What is $$L_0, L_1, L_2, L_{norm}$$?
	**Answer:**
    A norm is a function that assigns a scalar value to a vector, measuring the "size" or "magnitude" of the vector. It is a mathematical way to generalize the concept of length or distance in a vector space. The most common norm is the Euclidean norm, also known as the L2 norm.
    
    $L_0$, $L_1$, $L_2$, and $L_{norm}$ are different types of norms, also known as the p-norms, and are defined as follows:
        - $L_0$ norm: It is also called the "zero norm", it counts the number of non-zero elements in a vector, mathematically defined as:
        
        - $L_1$ norm: It's defined as the sum of the absolute values of the elements in a vector. It's also referred to as the "Taxicab / Manhattan" norm. It can be written as

        - $L_2$ norm: It's defined as the square root of the sum of the squares of the elements in a vector. It's also referred to as the "Euclidean" norm or the "2-norm". It can be written as
        
        - $L_{norm}$: It's defined as the p-th root of the sum of the p-th power of the absolute values of the elements in a vector. It's a general form of the above norms. It can be written as
        - 


	3. [M] How do norm and metric differ? Given a norm, make a metric. Given a metric, can we make a norm?

    **Answer:**
    
    A norm is a function that assigns a scalar value to a vector, usually denoted as ||x||, and it must satisfy certain properties such as non-negativity, identity, homogeneity, and the triangle inequality. A metric, on the other hand, is a function that assigns a scalar value to a pair of vectors, usually denoted as d(x, y), and it must also satisfy certain properties such as non-negativity, identity, symmetry and the triangle inequality.

    A norm can be used to define a metric by taking the square of the norm, e.g. d(x, y) = ||x - y||^2. This will satisfy all the properties of a metric and it's called Euclidean metric.

    Given a metric, it may not be possible to define a norm, as a metric does not have the homogeneity property, which is required for a norm.




## 5.1.2 Matrices

1. [E] Why do we say that matrices are linear transformations?

    **Answer:**
    A matrix can be thought of as a linear transformation because it represents a linear map from one vector space to another. Specifically, if A is an m x n matrix and x is a vector in R^n, then the product Ax is a vector in R^m, and the map that sends x to Ax is a linear transformation. This is because for any scalars a and b and any vectors x and y, the matrix-vector product satisfies the properties of a linear transformation:

    (A(ax + by) = a(Ax) + b(Ay)) and (A(x+y) = Ax + Ay)

    Additionally, matrix can also be represented with linear combination of the standard basis vectors, that is if we have a matrix A and standard basis vectors e1,e2,e3....en then it can be written as A = a11e1e1^T+a12e1e2^T+a13e1e3^T+...+a1ne1en^T, this representation also confirms that matrix is a linear transformation.

2. [E] What’s the inverse of a matrix? Do all matrices have an inverse? Is the inverse of a matrix always unique?

    **Answer:**
    The inverse of a matrix A, denoted as A^-1, is a matrix such that AA^-1 = A^-1A = I, where I is the identity matrix. The inverse of a matrix A exists if and only if the determinant of A is non-zero, and if it exists, it is unique.

    Not all matrices have an inverse. For example, a singular matrix (i.e. a matrix with a determinant of zero) does not have an inverse. A square matrix that has an inverse is called invertible or non-singular matrix.

    When it exists, the inverse of a matrix A can be found using various methods such as Gaussian elimination, Cramer's rule, or using the adjoint of the matrix.

    It's worth mentioning that not all matrices have inverse, for example, a matrix that is not square or a matrix that is a singular matrix (i.e. a matrix with a determinant of zero) does not have an inverse.

3. [E] What does the determinant of a matrix represent?

    **Answer:**
    The determinant of a matrix, denoted by |A| or det(A), is a scalar value that can be calculated from the elements of a square matrix. It is a useful mathematical tool that has many applications in linear algebra and other areas of mathematics and physics.

    The determinant of a matrix can be thought of as a measure of the "volume" of the linear transformation associated with the matrix. For example, if A is a 2x2 matrix, then |A| represents the scaling factor of the linear transformation that stretches or shrinks a unit square by a factor of |A| in the direction of the transformation.
    
    Determinant can be calculated using a variety of methods, depending on the size of the matrix, such as using Laplace expansion for larger matrices or using the rule of Sarrus for smaller matrices. For example, for 2x2 matrix the determinant can be found by (ad - bc) where a,b,c,d are the elements of the matrix.


4. [E] What happens to the determinant of a matrix if we multiply one of its rows by a scalar $$t \times R$$?

    **Answer:**
    If we multiply one of the rows of a matrix A by a scalar t, the determinant of the new matrix will be multiplied by t.

    Formally, let A be an n x n matrix and let R be one of its rows. If we multiply R by a scalar t, we get a new matrix A' = A with the row R replaced by t*R. The determinant of A' is then given by det(A') = t * det(A).

    In other words, the determinant of a matrix is multiplied by the scalar factor when one of its rows is multiplied by a scalar. This is because the determinant of a matrix is a scalar value, and multiplying a scalar value by a scalar factor results in a new scalar value that is equal to the original value multiplied by the scalar factor.

    This is also true when we multiply a column of the matrix by a scalar, in that case the determinant will be scaled by the same scalar too.

    This property is used in linear algebra to simplify matrix operations, such as in finding the inverse of a matrix or solving systems of linear equations.

5. [M] A $$4 \times 4$$ matrix has four eigenvalues $$3, 3, 2, -1$$. What can we say about the trace and the determinant of this matrix?

    **Answer:**
    The trace of a square matrix is the sum of its diagonal elements, so the trace of a 4x4 matrix is the sum of its four diagonal elements.

    Given that the matrix has four eigenvalues 3, 3, 2, and -1, we can say that the trace of this matrix is:

    trace = 3 + 3 + 2 + (-1) = 7

    The determinant of a matrix is a scalar value that can be calculated from the elements of a square matrix. For a 4x4 matrix, the determinant is a polynomial function of the eigenvalues of the matrix.

    Given that the matrix has four eigenvalues 3, 3, 2, and -1, we can say that the determinant of this matrix is:

    det = (3)(3)(2)*(-1) = -18

    Therefore, the determinant of the matrix is -18.

    It's worth noting that if the matrix has multiple equal eigenvalues, then the determinant will be the product of all the eigenvalues raised to the power of the number of times they repeat.

6. [M] Given the following matrix:<br>
	$$
	\begin{bmatrix}
		1 & 4 & -2 \\
		-1 & 3 & 2 \\
		3 & 5 & -6
	\end{bmatrix}
	$$

	Without explicitly using the equation for calculating determinants, what can we say about this matrix’s determinant?
  **Hint**: rely on a property of this matrix to determine its determinant.
  
    **Answer:**
    Last column of this matrix is linearly dependeant of the first column. If we multiply the first column by -2, we will get the last column.
    This means that the determinant of this matrix will be zero, as it is not invertible.
    It is worth noting that this is a general property of matrices with linearly dependent rows/columns, and this property can be useful to quickly check if a matrix is invertible without calculating its determinant explicitly.


7. [M] What’s the difference between the covariance matrix $$A^TA$$ and the Gram matrix $$AA^T$$?

    **Answer:**
    The covariance matrix of a matrix A is defined as $A^TA$, where $A^T$ is the transpose of A. The covariance matrix is a square matrix that describes the variances and covariances of the columns of A. It is used in statistics and machine learning to understand the relationships between different features in a dataset.

    On the other hand, the Gram matrix of a matrix A is defined as $AA^T$, where $A^T$ is the transpose of A. It is a square matrix that describes the inner product of the columns of A. It is used in linear algebra and machine learning to understand the relationships between the columns of A.

    In other words, the covariance matrix measures the linear relationship between columns of A, whereas the Gram matrix measures the inner product between the columns of A.

      - In practice, the covariance matrix is used in linear discriminant analysis, principal component analysis and also in Gaussian Mixture Model, 
      - whereas the Gram matrix is used in kernel methods, such as support vector machines and kernel principal component analysis.

    It is worth noting that the size of the Gram matrix is the number of columns of **A squared**, whereas the size of the covariance matrix is the number of **columns of A**.

8. Given $$A \in R^{n \times m}$$ and $$b \in R^n$$
	1. [M] Find $$x$$ such that: $$Ax = b$$.
	
		**Answer:**
		To find x such that Ax = b, we need to solve the equation for x.

		One way to do this is by using matrix inverse, if the matrix A is invertible, we can multiply both sides of the equation by A^-1, to get:

		x = A^-1 b

		Another way is to use Gaussian elimination method or Gauss-Jordan method, which are used to solve a system of linear equations in the form of A*x = b, where A is a matrix, x is a column vector and b is a column vector or a constant.

		It's also worth noting that if A is not invertible, it means that the system of equations is either inconsistent or has infinitely many solutions. In this case, the matrix A is called a singular matrix, the equation Ax = b doesn't have a unique solution, and it's not possible to find x using matrix inverse.

		It's also to use other methods such as LU decomposition, QR decomposition, and other iterative methods like Jacobi, Gauss-Seidel, and conjugate gradient method.
	
	3. [E] When does this have a unique solution?
	
		**Answer:**
		The equation Ax = b has a unique solution when the matrix A is invertible. In other words, when the matrix A is non-singular.

		A matrix is considered invertible or non-singular if its determinant is non-zero.

		A matrix whose determinant is not equal to zero is said to be regular or non-singular matrix. It means that there exists a unique matrix A^-1, such that AA^-1 = A^-1A = I, where I is the identity matrix.

		If the matrix A is invertible, then we can always find a unique solution x by multiplying both sides of the equation Ax = b by the inverse matrix A^-1, as we've discussed in the first question.

		However, if the matrix A is singular, it means that it is not invertible, and therefore it doesn't have a unique solution. In this case, the equation Ax = b will have either infinitely many solutions or no solutions at all.
	
	5. [M] Why is it when A has more columns than rows, $$Ax = b$$ has multiple solutions?
	
		**Answer:**

		When a matrix A has more columns than rows, it is called a "tall" matrix. In this case, the equation Ax = b is called an over-determined system of linear equations.

		An over-determined system of linear equations is one in which the number of equations is greater than the number of unknowns. In this case, there are more columns in matrix A than rows, meaning that there are more equations than variables.

		In this case, the system of equations is inconsistent, meaning that there is no solution that satisfies all the equations simultaneously. However, it is still possible to find a solution that approximately satisfies the equations.

		This is because there are infinitely many solutions that are not exact solutions but still approximate solutions. In other words, if a matrix A has more columns than rows, it means that there are more equations than variables and thus multiple solutions for x.
	
	7. [M] Given a matrix A with no inverse. How would you solve the equation $$Ax = b$$? What is the pseudoinverse and how to calculate it?
	
		**Answer:**
		When a matrix A has no inverse, it is called a singular matrix or a rank-deficient matrix, and it is not possible to solve the equation Ax = b using the traditional method of multiplying both sides by A^-1. However, there are still ways to find a solution to the equation.

		One way to solve the equation Ax = b when A is singular is to use the concept of the "pseudoinverse" of a matrix. The pseudoinverse of a matrix A is denoted by A^+, and it is a generalization of the inverse matrix.

		The most common ways to calculate the pseudoinverse of a matrix is by using the Singular Value Decomposition (SVD) of A, which is defined as A = UΣV^T. Where U and V are orthogonal matrices and Σ is a diagonal matrix.

		The pseudoinverse of A is then defined as A^+ = VΣ^+U^T, where Σ^+ is the matrix of reciprocals of the non-zero singular values of A on its diagonal, and zeroes elsewhere.

		Once we have the pseudoinverse of A, we can use it to solve the equation Ax = b by multiplying both sides of the equation by A^+:

		A^+Ax = A^+b

		x = A^+b

		This method is known as the "Moore-Penrose inverse" or the "generalized inverse" and it can be used to find a least-squares solution to the equation Ax = b, even when A is singular or not invertible.

		It's also worth noting that, the solution found using the pseudoinverse is not unique, as it is an approximate solution.

9. Derivative is the backbone of gradient descent.
	1. [E] What does derivative represent?
		**Answer:**
		The derivative of a function represents the rate of change of the function with respect to one of its variables. More specifically, it represents the slope of the tangent line to the graph of the function at a given point. The derivative can be thought of as the instantaneous rate of change of the function at a particular point.
		The derivative is typically represented by the symbol f'(x) or dy/dx and can be found by applying the limit definition of the derivative or by using calculus rules and formulas such as the power rule, product rule, quotient rule and chain rule.

	3. [M] What’s the difference between derivative, gradient, and Jacobian?
	
		**Answer:**
		The derivative, gradient, and Jacobian are all related concepts in calculus, but they have slightly different meanings:

		Derivative: A derivative is a measure of how much a function changes as the input changes. It measures the rate of change of a function with respect to one of its variables. The derivative of a function f(x) is typically represented by the symbol f'(x) or dy/dx. For example, the derivative of the position with respect to time is velocity.

		Gradient: The gradient of a scalar-valued function of multiple variables is a vector-valued function that points in the direction of the greatest rate of increase of the function. It is a vector whose components are the partial derivatives of the function with respect to each variable. The gradient of a function f(x,y,z) is typically represented by the symbol ∇f or ∇(f).

		Jacobian: The Jacobian is a matrix that describes the change of a vector-valued function with respect to its inputs. It is a matrix whose elements are the partial derivatives of the function with respect to each input variable. For example, the Jacobian of a function f(x,y) = (u(x,y), v(x,y)) would be the matrix
		$$
		\begin{bmatrix}
		\frac{\partial u}{\partial x} & \frac{\partial u}{\partial y} \\
		\frac{\partial v}{\partial x} & \frac{\partial v}{\partial y}
		\end{bmatrix}
		$$

		In summary, the derivative is a scalar that measures the rate of change of a function with respect to one of its variables, the gradient is a vector that points in the direction of the greatest rate of increase of a scalar-valued function, and the Jacobian is a matrix that describes the change of a vector-valued function with respect to its inputs.
10. [H] Say we have the weights $$w \in R^{d \times m}$$ and a mini-batch $$x$$ of $$n$$ elements, each element is of the shape $$1 \times d$$ so that $$x \in R^{n \times d}$$. We have the output $$y = f(x; w) = xw$$. What’s the dimension of the Jacobian $$\frac{\delta y}{\delta x}$$?

	**Answer:**
	The dimension of the Jacobian of the function y = f(x; w) = xw with respect to x is determined by the number of rows in the output matrix y, and the number of columns in the input matrix x.

	y = f(x; w) = xw is a matrix multiplication between the input matrix x and the weight matrix w, so the output matrix y will have the same number of rows as x and the same number of columns as w. Therefore, the dimension of the output matrix y is n x m.

	The Jacobian of the function y = f(x; w) = xw with respect to x will have the same number of rows as y, and the same number of columns as x. The dimension of the Jacobian is n x d.

	Therefore, the dimension of the Jacobian 
	
	$\frac{δy}{δx}$  is $n×d$.

12. [H] Given a very large symmetric matrix A that doesn’t fit in memory, say $$A \in R^{1M \times 1M}$$ and a function $$f$$ that can quickly compute $$f(x) = Ax$$ for $$x \in R^{1M}$$. Find the unit vector $$x$$ so that $$x^TAx$$ is minimal.
	
	**Hint**: Can you frame it as an optimization problem and use gradient descent to find an approximate solution?
	
	**Answer:**
	
	The problem of finding the unit vector x so that $$x^TAx$$ is minimal can be framed as an optimization problem. The objective function to minimize is given by:
	
	$$x^TAx=x^TAx=x^T(Ax)=||Ax||^2$$
	We can find the unit vector x that minimizes this function by using gradient descent. The gradient of the objective function with respect to x is given by:
	
	$$∇_x{x}^TAx=2Ax$$

	We can use this gradient to update the values of x in the direction that decreases the value of the objective function. To ensure that the vector x remains a unit vector, we can normalize it after each update.

	In practice, we can use stochastic gradient descent (SGD) to approximate the solution for very large matrices A, instead of the vanilla gradient descent. SGD uses random samples of the data to compute the gradients, and it is more computationally efficient than vanilla gradient descent when the data is too large to be stored in memory.

	We would also need to take care of the learning rate and the number of iteration which is going to be used in the algorithm to find the approximate solution.
	
	```python
	import numpy as np

	def find_min_x(A, max_iterations=1000, learning_rate=0.01):
	    # Initialize x with random values
	    x = np.random.rand(A.shape[0])
	    x = x / np.linalg.norm(x)

	    for i in range(max_iterations):
		# Compute the gradient
		grad = 2 * np.dot(A, x)
		# Update x
		x = x - learning_rate * grad
		# Normalize x
		x = x / np.linalg.norm(x)
	    return x
	
	def find_min_x_SGD(A, max_iterations=1000, learning_rate=0.01, batch_size=64):
	    # Initialize x with random values
	    x = np.random.rand(A.shape[0])
	    x = x / np.linalg.norm(x)

	    for i in range(max_iterations):
		# Sample a random mini-batch of rows from A
		indices = np.random.randint(A.shape[0], size=batch_size)
		mini_batch = A[indices]

		# Compute the gradient
		grad = 2 * np.dot(mini_batch.T, np.dot(mini_batch, x))
		# Update x
		x = x - learning_rate * grad
		# Normalize x
		x = x / np.linalg.norm(x)
	    return x
	A = np.random.rand(1000, 1000)
	x = find_min_x(A)
	x = find_min_x_SGD(A)
	```





