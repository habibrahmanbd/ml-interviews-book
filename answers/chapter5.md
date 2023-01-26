### 5.1.1 Vectors
If some characters seem to be missing, it's because MathJax is not loaded correctly. Refreshing the page should fix it.

1. Dot product

    i. **[E] What’s the geometric interpretation of the dot product of two vectors?**
    
    **Answer:**
    The dot product (also known as the scalar product or inner product) of two vectors is a scalar value that represents the cosine of the angle between the two vectors. The dot product is defined as:
    
    v1.v2 = ||v1|| * ||v2|| * cos(θ)
    
    where ||v1|| and ||v2|| are the magnitudes (lengths) of the two vectors, and θ is the angle between them.
    
    Geometrically, the dot product can be interpreted as the projection of one vector onto another. The dot product of two vectors can be thought of as the length of the projection of one vector onto the other, multiplied by the cosine of the angle between them.
    
    For example, if two vectors are perpendicular to each other, the angle between them is 90 degrees, and the cosine of this angle is 0. This means that the dot product will be zero, which corresponds to the fact that there is no projection of one vector onto the other. On the other hand, if two vectors are parallel to each other, the angle between them is 0 degrees, and the cosine of this angle is 1. This means that the dot product will be the product of the magnitudes of the two vectors, which corresponds to the fact that one vector is a scaled version of the other.
    
    ii. **[E] Given a vector , find vector of unit length such that the dot product of and is maximum.**
    
    **Answer:**
    
    Given a vector "a" , in order to find a vector "b" of unit length such that the dot product of a and b is maximum, we can take the following steps:

    Normalize the vector "a" to create a unit vector "a_hat" by dividing it by its magnitude (length) ||a||.
    
  $a_{hat} = \frac{a} {||a||}$

    Use the unit vector "a_hat" as the vector "b" of unit length such that the dot product of a and b is maximum.
The dot product of a and a_hat will be maximum because a_hat is a unit vector, and its magnitude is 1, which means that it is already of unit length. And since the dot product is defined as the product of the magnitudes of the two vectors multiplied by the cosine of the angle between them, when one of the vectors is a unit vector, the dot product will be equal to the product of the magnitudes of the two vectors. Since the magnitude of a_hat is 1 and the dot product of a and a_hat is a.a_hat, it will be equal to the magnitude of a and thus maximum.


2. #### Outer product

[E] Given two vectors and . Calculate the outer product ?
[M] Give an example of how the outer product can be useful in ML.
[E] What does it mean for two vectors to be linearly independent?

[M] Given two sets of vectors and . How do you check that they share the same basis?
[M] Given vectors, each of dimensions. What is the dimension of their span?
Norms and metrics
[E] What's a norm? What is ?
[M] How do norm and metric differ? Given a norm, make a metric. Given a metric, can we make a norm?
