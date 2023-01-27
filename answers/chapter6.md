#### 5.1.2 Matrices

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
	1. [E] When does this have a unique solution?
	1. [M] Why is it when A has more columns than rows, $$Ax = b$$ has multiple solutions?
	1. [M] Given a matrix A with no inverse. How would you solve the equation $$Ax = b$$? What is the pseudoinverse and how to calculate it?

9. Derivative is the backbone of gradient descent.
	1. [E] What does derivative represent?
	1. [M] What’s the difference between derivative, gradient, and Jacobian?
10. [H] Say we have the weights $$w \in R^{d \times m}$$ and a mini-batch $$x$$ of $$n$$ elements, each element is of the shape $$1 \times d$$ so that $$x \in R^{n \times d}$$. We have the output $$y = f(x; w) = xw$$. What’s the dimension of the Jacobian $$\frac{\delta y}{\delta x}$$?
11. [H] Given a very large symmetric matrix A that doesn’t fit in memory, say $$A \in R^{1M \times 1M}$$ and a function $$f$$ that can quickly compute $$f(x) = Ax$$ for $$x \in R^{1M}$$. Find the unit vector $$x$$ so that $$x^TAx$$ is minimal.
	
	**Hint**: Can you frame it as an optimization problem and use gradient descent to find an approximate solution?
