# lalg - blas/cuda integration with nodejs

[![Join the chat at https://gitter.im/node-linalg/Lobby](https://badges.gitter.im/node-linalg/Lobby.svg)](https://gitter.im/node-linalg/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

This module integrates the efficient libraries created by really smart folks
into the rich javascript environment provided by nodejs. 

At present this is in alpha mode. It's tested in real ML apps and holds up very well. 

I need to make the install easier - sorry. Check the Docker file for linux
prerequistes. 

Of note:

* underlying data type is float (can we template this?)
* matrices are stored in column major order (for cuda compatibility) 
* limited validation of inputs is present in this version (to be be improved) 
* more non-blocking options ( e.g. svdp and pinvp )

# Help
* need anyone who can build on windows
* (polite) suggestions for matrix operations

# Coming Soon
* NVIDIA CUDA support - beta testing in progress (20kx20k mul() in 500ms!!!)
* caffe integration - let's get the caffe chaps' good work into nodejs :)

# Contribute

This library builds on the following. Great work chaps - so good it's worth stealing!

* [CppNumericalSolvers](https://github.com/PatWie/CppNumericalSolvers)
* [eigen3](http://eigen.tuxfamily.org/)


# Installation

npm install lalg

BUT  there are several prerequesites (see below)
Promise to make this better, but I am still learning about software.

## Prerequisites

* nodejs of course ( tested on 7.2.0 )
* gyp - npm install -g node-gyp
* make
* python ( oh boy! - can't avoid python - even integrating C++ to javascript !!! )
* C++ compiler 
* openblas  
* lapacke 

The [Dockerfile](https://github.com/rcorbish/node-linalg/blob/master/Dockerfile) shows the requirements in detail

```
	docker run -i -t rcorbish/lalg 
```

```
	var lalg = require ( "lalg" ) ;
        lalg.rand( 5 ) ;
```

Verify correct installation by: ``` node node_modules/lalg/test/test.js ```

# API 

This is the C++ docs, which shows all the nodejs functions and a few
examples. I'll keep this up to date as the unerlying code changes. NB.
the javascript calls start with lowercase letters.

[documentation](https://rcorbish.ydns.eu/lalg/classWrappedArray.html)

# Examples

## Create a matrix

Create an uninitialized array (not too useful really)
```
	var lalg = require('lalg');
	var A = new lalg.Array( 10, 5 ) ;
```

Create an randomly initialized array
```
	var lalg = require('lalg');
	var A = lalg.rand( 10, 5 ) ;
```

Create two arrays one initialized to all 0.0, the other to all 1.0
```
	var lalg = require('lalg');
	var Zeros = lalg.zeros( 10, 5 ) ;
	var Ones = lalg.ones( 10, 5 ) ;
```

Create the identity matrix ( identity is square, only need 1 arg )
```
	var lalg = require('lalg');
	var Identity5 = lalg.eye( 5 ) ;
```

## Scalar functions

Add a value to each element in the matrix
Multiply each element in the matrix by a single value. NB add() also
takes vectors and matrices as args too (see later).

```
	var lalg = require('lalg');
	var A = lalg.zeros( 5 ) ;
	var T = A.add( 10 ) ;   // now all elements in T are equal to 10
```

Multiply each element in the matrix by a single value. NB mul() also
takes vectors and numbers (be careful with vectors, they must be
compatible shapes)
```
	var lalg = require('lalg');
	var A = lalg.ones( 5 ) ;
	var T = A.mul( 10 ) ;   // now all elements in T are equal to 10
```

## Linear functions 

Multiply two matrices. This is the same call as the scalar version, but passing
in a matrix or vector fires off the linear matrix multiplies
```
	var lalg = require('lalg');
	var A = lalg.rand( 5 ) ;
	var B = lalg.rand( 5 ) ;
	var T = A.mul( B ) ;   // T = A x B  - another 5x5 matrix
	console..log( T ) ;
```

## Other functions 

Other functions that may be useful:

* mean - calculate the mean of rows or columns
* sum - calculate the sum of elements in rows or columns
* norm - calculates the Euclidean norm of rows or columns
* inv - the matrix inverse
* pinv - the pseudo inverse, can calculate an inverse for non-square and singular matrices
* log - calculate the log of each element
* abs - absolute value of each element
* sqrt - the sqrt of each element
* svd - singular value decomp of a matrix - return U,S, Vt in once object
* pca - principal components analysis, reduces the dimension of a vector
* transpose - transpose a matyix
* dup - copy a matrix 

## Element manipulation

Some functions of a matrix are provided to extract/add/move rows and columns

* rotate - rotate the columns in a matrix
* addColumn - adds a row vector to a matrix
* addRow - adds a column vector to a matrix
* removeRow - rfemoves a row vector from a matrix
* removeCoOlumn - removes a column from a matrix
* getRows - copies rows from a matrix to a new matrix
* getColumns - copies columns from a matrix to a new matrix

## Linear regression 
OK we'll try a more complex example. It showcases the non-blocking 
features of the library. 

This implements the linear regression simple cals: ``` theta = inv(X' X) X' y ```
where X' is a transpose operation ( thanks MATLAB )

This needs fast-csv ``` npm install fast-csv```

Click [here](https://github.com/rcorbish/node-linalg/blob/master/test/wine.js) for a sample. 
Linear regression as a first step. 

Plot the results (in a spreadsheet for example) to see if we're accurate. Don't worry we'll do better 
with logistic regression.

## Equation solving 

This implements a solver to find the minimum value of a function. Given a matrix of features
and a means to calculate a partial differential (gradients) finds the global minimum.

Click [here](https://github.com/rcorbish/node-linalg/blob/master/test/solve.js) for a sample. 

See the [docs](https://rcorbish.ydns.eu/lalg/classWrappedArray.html#a528d9aae6c7cc261d8aa4b457cb2250b) for the requirements to define the gradient calculations. 
