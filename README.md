# node-linalg

This module is to use the power of the efficient blas (and soon cuda)
linear algebra in the rich javascript environment provided by nodejs.

Of note:

* underlying data type is float
* matrices are stored in column major order (for cuda compatibility)
* limited validation of inputs is present in this version (to be be improved)


# Installation

## Prerequisites

* nodejs of course ( tested on 7.1.0 )
* gyp - npm install -g node-gyp
* C++ compiler apt install -y g++5  ( cuda nvcc is not ready for 6 yet )
* openblas  apt install -y libopenblas-dev


# API 

[documentation](https://rcorbish.ydns.eu/lalg/)

# Examples

## create a matrix

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

## scalar functions

Multiply each element in the matrix by a single value
```
	var lalg = require('lalg');
	var A = lalg.ones( 5 ) ;
	var T = A.mul( 10 ) ;   // now all elements in T are equal to 10
```

Add a value to each element in the matrix
```
	var lalg = require('lalg');
	var A = lalg.zeros( 5 ) ;
	var T = A.add( 10 ) ;   // now all elements in T are equal to 10
```


