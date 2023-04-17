# EE-399-hw2
``Author: Ben Li``
``Date: 4/16/2023``
``Course: SP 2023 EE399``

## Abstract
In this homework we are given a matrix that contains 2414 downsampled grayscale images of faces. Each image has 32x32 pixels. The objective of this is to perform different linear algebra operations to gain insight intot he dataset. 

## Introduction and Overview
This homework provides a comprehensive introcuation to the important concept in linear algebra and its application in machine learning. This also shows how to implement SVD. This provides a measure of how muh information is retained by projection.

There are five task in total. 
1. The first is to calculate the correlation matrix between the first 100 imagse in X. This is done by calculating the dot product for each pair of images. The resulting matrix is plotted using a color map. 

2. Repeat the same computation for subset of 10x10 images to observe patterns in the coorelations between a smaller groups of images. This correlation matrix is also plotted. 

3. Use eigen-decomposition to find the first six eigenvectors of the matrix 
$$Y = XXT$$
> This represent the covariance matrix of the data. 
Also, singular value decomposition (SVD) is used to find the first six principal components of X.

4. Use the first eigenvector from the eigen-decomposstion to compare with the first SVD mode. Compute the norm of the difference between their absolute value.

5. Find the percentage of variance captured by each of the first six SVD mode. Each of these modes are plotted as 32x32 images to see the patterns in the most significant components of the data.


## Theoretical Background
There are several components from linear algebra and statistics that were applied in this assignment. We uses correlation matrix to study the relationship between variables. In this context, correlation matrix can be use to study the similarity between pairs of images. The matrix contains pairwise correlations between the columns of data matrix. The result of this matrix is symmetric with ones along the diagonal, and values ranging from -1 to 1 off the diagonal. 

<p align="center" width="100%">
    <img width="25%" src="https://user-images.githubusercontent.com/121909443/232352223-5fffd5e5-bf73-4335-9adc-65fb12318280.png"> 
    <em>Figure 1: Example of a correlation matrix</em>
</p>

We also use SVD to get the most significant features in a dataset. In this case we use SVD to extract the principal components of the face dataset. The concept of SVD is to factorizes a matrix into three different matrices: ``U, S, and V``. U, S, and V. U and V are orthogonal matrices, and S is a diagonal matrix containing the singular values of the original matrix. In this case, we are using the SVD to extract the first six principal component directions from the data matrix X.

Variance is a measure of the spread or variability of a set of data. In this case, we are computing the percentage of variance captured by each of the first six SVD modes. This provides a measure of how much information is retained by the projection onto a lower-dimensional space, and it can help us determine how many principal components are needed to adequately represent the data.

## Algorithm Implementation and Development 

We begin by loading the data from the provided "yalefaces.mat" file and store it in matrix X
```python
# Load data
results = loadmat('yalefaces.mat')
X = results['X']

# Part (a)
C = np.dot(X[:,:100].T, X[:,:100])
plt.pcolor(C)
plt.colorbar()
plt.xlabel("image index (horizontal)")
plt.ylabel("image index (vertical)")
plt.title('Correlation Matrix')
plt.show()

# Part (b)
# Find the indices of the most and least correlated images
max_corr_idx = np.unravel_index(np.argmax(C), C.shape)
min_corr_idx = np.unravel_index(np.argmin(C), C.shape)
```
In part a, we computes a 100x100 correlation matrix C using the dot product (correlation) between the first 100 images in the matrix X. It then visualizes the matrix using pcolor and displays it using the matplotlib library.

Next, Part (b) finds the indices of the most and least correlated images in the correlation matrix C using the numpy function argmax and argmin, respectively. It then plots the two most correlated images and the two least correlated images using the matplotlib library.

```python
# Part (c)
C = np.dot(X[:,[0,312,511,4,2399,112,1023,86,313,2004]].T, X[:,[0,312,511,4,2399,112,1023,86,313,2004]])
plt.pcolor(C)
plt.colorbar()
plt.title('Correlation Matrix')
plt.xlabel("image index (horizontal)")
plt.ylabel("image index (vertical)")
plt.show()

# Part (d)
Y = np.dot(X, X.T)
eig_vals, eig_vecs = np.linalg.eig(Y)
idx = eig_vals.argsort()[::-1]
eig_vals = eig_vals[idx]
eig_vecs = eig_vecs[:,idx]
eig_vecs = np.dot(X.T, eig_vecs)
eig_vecs = eig_vecs / np.linalg.norm(eig_vecs, axis=0)

# Part (e)
U, s, V = np.linalg.svd(X)
V = V.T
```

Part (c) creates a new correlation matrix using only 10 specific images from the Yale Faces dataset. The indices of these images are [0,312,511,4,2399,112,1023,86,313,2004].

Part (d) computes the eigendecomposition of the covariance matrix Y = X*X^T. Specifically, it computes the eigenvalues and eigenvectors of Y and sorts them in descending order by eigenvalue. The eigenvectors are then transformed to the eigenvectors of the original covariance matrix by multiplying them with X^T and normalizing them.

Part (e) computes the SVD of the original data matrix X.

```python
# Part (f)
diff_norm = np.linalg.norm(np.abs(eig_vecs[:,0]) - np.abs(V[:,0]))

print("Norm of difference between first  v1 and first SVD mode u1:", diff_norm)

# Part (g)
variance = (s ** 2) / (X.shape[1] - 1)
variance_ratio = variance / np.sum(variance)
print("Percentage of variance captured by each SVD mode:")
for i in range(6):
    print(f"Mode {i+1}: {variance_ratio[i]*100:.2f}%")

  
# Reshape the first 6 columns of U to 32x32 images
svd_modes = U[:, :6].reshape(32, 32, 6)

# Plot the first 6 SVD modes
fig, axes = plt.subplots(1, 6, figsize=(12, 3))


for i in range(6):
    axes[i].imshow(svd_modes[:, :, i], cmap='gray')
    axes[i].set_title(f'SVD Mode {i+1}')
    axes[i].axis('off')

plt.show()
```

Part (f) computes the norm of the difference between the first principal component (eigenvector) obtained from the eigendecomposition in part (d) and the first left singular vector (column of U) obtained from the SVD in part (e).

Part (g) computes the percentage of variance captured by each singular value (element of s) in the SVD and prints them out for the first 6 modes. The variance is computed using the formula (s^2)/(n-1), where s is the vector of singular values and n is the number of data points. The variance ratios are obtained by dividing the variance of each mode by the sum of all variances.

## Computational Results
In part a, the result is a 100x100 correlation matrix where each element c_jk is the dot product between the jth and kth columns of the first 100 images in the matrix X. Figure 1 shows teh heat map of the coorelation matrix where darker colors represent higher correlations and lighter colors represent lower correlations. From this, we can tell which images has a higher correlation with which image.
<p align="center" width="100%">
    <img width="25%" src="https://user-images.githubusercontent.com/121909443/232354087-1ffaaa42-5142-40ef-8d2a-788d78b64418.png"> 
    <em>Figure 2: Correlation Matrix for the first 100 images</em>
</p>

Using the result from part A we can identify the image that are highly correlated by finding the indices of the maximum value in the correlation matrix. In this case, we can use the np.argmax() function to find the index of the maximum value in the matrix. We can then use the np.unravel_index() function to convert the index of the maximum value into a pair of indices that correspond to the row and column of the maximum value in the matrix. These indices correspond to the most highly correlated images.

<p align="center" width="100%">
    <img width="25%" src="https://user-images.githubusercontent.com/121909443/232354628-43bda260-c86c-41d3-b3f6-790f69394ad1.png"> 
    <em>Figure 3: Most correlated image</em>
</p>

<p align="center" width="100%">
    <img width="25%" src="https://user-images.githubusercontent.com/121909443/232354667-62e91678-5b4a-4b47-9a3a-689b4fb55cca.png"> 
    <em>Figure 4: least correlated image</em>
</p>
This shows the correlation matrix for the first 10 images, we can clearly see the image horizontal index and vertical index for the area with high correlation and low correlation.

<p align="center" width="100%">
    <img width="40%" src="https://user-images.githubusercontent.com/121909443/232354977-d6767974-54a8-4c73-8184-1e4860f51502.png"> 
    <em>Figure 5: Correlation Matrix for 10 images</em>
</p>

The norm of the difference between the first eigenvector obtained by computing the eigendecomposition of the covariance matrix and the first singular vector obtained by computing the SVD of the data matrix is very small, indicating that the two methods are producing very similar results.

The percentage of variance captured by each SVD mode is printed, and we can see that the first SVD mode captures the majority of the variance in the data (around 72.93%). This means that if we only kept the first mode and discarded the rest, we would still be able to represent the data with relatively high fidelity. The remaining modes capture decreasing amounts of variance.

```
Norm of difference between first  v1 and first SVD mode u1: 9.424563828286949e-16
Percentage of variance captured by each SVD mode:
Mode 1: 72.93%
Mode 2: 15.28%
Mode 3: 2.57%
Mode 4: 1.88%
Mode 5: 0.64%
Mode 6: 0.59%
```
This shows teh principle components of the face in the dataset. The first SVD mode represents the important components of the dataset. This capstures the largest amount of of variation. Each subsequent mode captures smaller and smaller amounts of variation. We can see that the first mode captures 72.93% of teh variance. Therefore, using just the first mode can be a good approximation of the original image.

<p align="center" width="100%">
    <img width="50%" src="https://user-images.githubusercontent.com/121909443/232354841-1e7063f3-79f5-4e7d-beaa-2cbc5cf1cb48.png"> 
    <em>Figure 6: SVD mode for six images</em>
</p>

## Summary and Conclusions

In this assignment, we explored the concept of SVD and correlation matrices using a dataset of 2414 images of faces. We started by computing a 100x100 correlation matrix by taking the dot product of the first 100 images. We then visualized the matrix using a heatmap and observed that some images are highly correlated while others are not. We were able to identify the most and least correlated images and plotted them side by side.

Next, we compared the eigenvectors obtained from computing the eigendecomposition of the covariance matrix with the right singular vectors obtained from SVD. We found that the first singular vector of SVD is proportional to the eigenvector associated with the largest eigenvalue of the covariance matrix.

Finally, we computed the variance captured by each SVD mode and found that the first few modes capture most of the variance in the data. This implies that we can reconstruct the original data using only a subset of the modes, which can be useful for dimensionality reduction and compression.

In conclusion, SVD and correlation matrices are powerful tools for analyzing high-dimensional data such as images. They allow us to identify patterns and relationships between the data points and can help us reduce the dimensionality of the data while preserving important information.
