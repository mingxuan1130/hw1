"""
use this file to script the creation of plots, run experiments, print information etc.

Please put in comments and docstrings in to make your code readable
"""
from matlib import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigvals
from scipy.stats import norm
import numpy.linalg as la
from scipy.linalg import lu
from scipy.linalg import cholesky, solve_triangular,lu_factor, lu_solve
import time

# Problem 0
# part A
sizeN = 1000
allEigenvalues = []

# Collect eigenvalues from all matrices
for _ in range(100):
    A = np.random.normal(loc=0, scale=1, size=(sizeN, sizeN))
    symm = (A + A.T) / 2
    eigenvalues = eigvals(symm)
    plt.hist(eigenvalues,alpha=0.5)

# Convert to a numpy array
plt.title('Histogram of Eigenvalues')
plt.xlabel('Eigenvalue')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

mu, std = norm.fit(allEigenvalues)
print(mu, std)

# Compute expected values
expectedValues = norm.pdf(allEigenvalues, mu, std)

# Compute errors
errors = allEigenvalues - expectedValues

# Plotting the errors
plt.hist(errors, bins=50, alpha=0.7, label='Errors')
plt.axvline(0, color='r', linestyle='dashed', linewidth=1)
plt.title('Errors in Fit of Normal Distribution')
plt.xlabel('Error')
plt.ylabel('Frequency')
plt.legend()
plt.show()

sizeN = 200
for _ in range(100):
    A = np.random.normal(loc=0, scale=1, size=(sizeN, sizeN))
    symm = (A + A.T) / 2
    eigenValues = eigvals(symm)
    plt.hist(eigenvalues,alpha=0.5)

plt.title('Histogram of Eigenvalues')
plt.xlabel('Eigenvalue')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

sizeN = 400
for _ in range(100):
    A = np.random.normal(loc=0, scale=1, size=(sizeN, sizeN))
    symm = (A + A.T) / 2
    eigenValues = eigvals(symm)
    plt.hist(eigenvalues,alpha=0.5)

plt.title('Histogram of Eigenvalues')
plt.xlabel('Eigenvalue')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

sizeN = 800
for _ in range(100):
    A = np.random.normal(loc=0, scale=1, size=(sizeN, sizeN))
    symm = (A + A.T) / 2
    eigenValues = eigvals(symm)
    plt.hist(eigenvalues,alpha=0.5)

plt.title('Histogram of Eigenvalues')
plt.xlabel('Eigenvalue')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

sizeN = 1600
for _ in range(100):
    A = np.random.normal(loc=0, scale=1, size=(sizeN, sizeN))
    symm = (A + A.T) / 2
    eigenValues = eigvals(symm)
    plt.hist(eigenValues,alpha=0.5)

plt.title('Histogram of Eigenvalues')
plt.xlabel('Eigenvalue')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

#PartB
sizeN = 200
maxEigenvalues = []

# Collect eigenvalues from all matrices
for _ in range(1000):
    A = np.random.normal(loc=0, scale=1, size=(sizeN, sizeN))
    symm = (A + A.T) / 2
    eigenValues = eigvals(symm)
    maxEigen = max(eigenValues)
    maxEigenvalues.append(maxEigen)

# Convert to a numpy array
plt.hist(maxEigenvalues,alpha=0.5)
plt.title('Histogram of maxEigenValues')
plt.xlabel('maxEigenValues')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

#PartC
sizeN = 200
maxGaps = []

# Collect eigenvalues from all matrices
for _ in range(1000):
    A = np.random.normal(loc=0, scale=1, size=(sizeN, sizeN))
    symm = (A + A.T) / 2
    eigenValues = eigvals(symm)
    sortEigenvalues = sorted(eigenValues)

    maxGap = 0
    for i in range(1, len(sortEigenvalues)):
        gap = sortEigenvalues[i] - sortEigenvalues[i-1]
        maxGap = max(maxGap, gap)
    maxGaps.append(maxGap)
    

# Convert to a numpy array
plt.hist(maxGaps,alpha=0.5)
plt.title('Histogram of maxGaps')
plt.xlabel('maxGaps')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

#PartD 
def generateSymm(sizeN):
    A = np.random.normal(loc=0, scale=1, size=(sizeN, sizeN))
    symm = (A + A.T) / 2
    return symm 

def computeSingularValues(matrixSize, trials=100):
    singularValues = []
    for _ in range(trials):
        matrix = generateSymm(matrixSize)
        singularValue = np.linalg.svd(matrix, compute_uv=False)
        singularValues.append(singularValue)
    return np.array(singularValues)

matrixSizes = [200, 400, 800, 1600]
fig, axs = plt.subplots(2, 2, figsize=(15, 12))

for i, size in enumerate(matrixSizes):
    singularValues = computeSingularValues(size)
    
    ax = axs[i//2, i%2]
    ax.hist(singularValues, bins=50, edgecolor='k', alpha=0.7)
    ax.set_title(f"Matrix size {size}")
    ax.set_xlabel("Singular value magnitude")
    ax.set_ylabel("Frequency")

plt.tight_layout()
plt.show()

#PartE
def generateSymm(sizeN):
    A = np.random.normal(loc=0, scale=1, size=(sizeN, sizeN))
    symm = (A + A.T) / 2
    return symm 

def computeConditionNumber(matrixSize, trials=100):
    singularValues = []
    for _ in range(trials):
        matrix = generateSymm(matrixSize)
        singularValue = np.linalg.svd(matrix, compute_uv=False)
        singularValues.append(singularValue)
    return max(singularValues) - min(singularV

# Problem 1
#PartA:
def solve_chol(A, b):
    L = cholesky(A, lower=True)
    y = solve_triangular(L, b, lower=True) 
    x = solve_triangular(L.T, y)
    return x

# # Test the function
# A = np.array([[4, 12, -16], [12, 37, -43], [-16, -43, 98]])
# A = A @ A.T
# b = np.array([1, 2, 3])
# x = solve_chol(A, b)
# print(x)
# #check it is indeed a solution
# la.norm(A@x-b)

#PartB
def solve_lu(A,b):
    lu, piv = lu_factor(A)
    x = lu_solve((lu, piv), b)
    return x

def generateSpd(size):
    A = np.random.rand(n,n)
    b = np.random.rand(n)
    return np.dot(A.T, A),b

listSizes = np.round(np.logspace(np.log10(10), np.log10(2000), 10)).astype(int)
timeCholesky = []
timeLu = []

for n in listSizes:
    # Generate random SPD matrix
    A,_ = generateSpd(n)
    
    # Time Cholesky decomposition
    start = time.time()
    L = cholesky(A, lower=True)
    end = time.time()
    timeCholesky.append(end - start)
    
    # Time LU decomposition
    start = time.time()
    P, L, U = lu(A)
    end = time.time()
    timeLu.append(end - start)


plt.loglog(listSizes, timeCholesky, 'o-', label='Cholesky Decomposition')
plt.loglog(listSizes, timeLu, 's-', label='LU Decomposition')
plt.xlabel('Size')
plt.ylabel('Time')
plt.title('Time Comparison: Cholesky vs LU Decomposition')
plt.legend()
plt.grid(True, which="both", ls="--", c='0.65')
plt.show()

#PartC
#A = QΛQT, A^n = (QΛQT)(QΛQT)(QΛQT)... = Q(Λ^n)QT
def calculatePower(A,n):
    eigenVal, eigenVec = np.linalg.eigh(A)
    Lambda_n = np.diag(eigenVal**n)
    return  eigenVec @ Lambda_n @ eigenVec.T

# #check if it is correct:
# A = np.array([[4, 2], [2, 3]])
# print(la.norm(calculatePower(A, 2)-A@A))
"""
Faster, because we are mostly using vectror power by numpy rather than doing 
matrix multiplication time by time 
"""

#PartD
def abs_det(A):
    P, L, U = lu(A)
    detA = np.prod(np.diag(U)) * np.linalg.det(P)
    return abs(detA)

# Problem 2
#PartA 
class myComplex:
    def __init__(self, real, imaginary):
        self.real = real
        self.imaginary = imaginary

    def __add__(self, other):
        newReal = self.real + other.real 
        newImag = self.imaginary + other.imaginary
        return myComplex(newReal, newImag)

    def __mul__(self, other):
        newReal = self.real*other.real - self.imaginary*other.imaginary
        newImag = self.real*self.imaginary + self.imaginary*other.real
        return myComplex(newReal, newImag)

    def conj(self):
        return myComplex(self.real, -self.imaginary)

    def __str__(self):
        return f"{self.real} + {self.imaginary}i"

#check if works 
a = myComplex(1,1)
b = a.conj()
res = a + b
print(res)

#PartB
def generateComplex(listComplex):
    #suppose my favorite number is generated by addtion of the list
    sum = myComplex(0,0)
    for num in listComplex:
        sum = sum + num 
    return sum 
    
# #test if works
# listComplex = [myComplex(4,5), myComplex(3,7), myComplex(2,8)]
# a = generateComplex(listComplex)
# print(a)

# QUESTION what do you mean by "Don't forget the complex conjugate."
def dot(a,b): #a and b are list
    if len(a) != len(b):
        return f"Vector a and vector b should have the same size."
    else:
        return np.sum([a[i]*b[i] for i in range(len(a))])

# #test if works:
# a = [1,2]
# b = [1,2]
# dot(a,b)

#Manually generate a vector and compute its norm
def manualVecNorm(size):
    vec = np.array([myComplex(np.random.rand(), np.random.rand()) for _ in range(size)])
    return np.sum([(num.real**2 + num.imaginary**2) for num in vec]) ** 0.5

#Do the same with numpy  
def numpyVecNorm(size):
    vec = np.array([np.random.rand(size), np.random.rand(size)])
    return np.linalg.norm(vec)


###THE SIZE NEED TO BE CHANGED BEFORE SUMMITION !!!
#Keep track of time using for both methods 
listSizes = list(range(1,1001))
timeManual = []
timeNumpy = []

for size in listSizes:
    startTime = time.time()
    manualVecNorm(size)
    endTime = time.time()
    timeManual.append(endTime - startTime)

    startTime = time.time()
    numpyVecNorm(size)
    endTime = time.time()
    timeNumpy.append(endTime - startTime)

listSizes = np.array(listSizes)
timeManual = np.array(timeManual)
timeNumpy = np.array(timeNumpy)

print(listSizes.shape)
print(timeManual.shape)
print(timeNumpy.shape)

#Plot the result from the previous step
plt.plot(listSizes, timeManual, label = "Manually")
plt.plot(listSizes, timeNumpy, label = "numpy")
plt.xlabel("sizes")
plt.ylabel("run time")
plt.title("run time comparison Manual vs Numpy")
plt.legend()
plt.show()
