# lalg - blas/cuda integration with nodejs

[![Join the chat at https://gitter.im/node-linalg/Lobby](https://badges.gitter.im/node-linalg/Lobby.svg)](https://gitter.im/node-linalg/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

This module integrates the efficient libraries created by really smart folks
into the rich javascript environment provided by nodejs. 

At present this is in pre-alpha ( == void? ). It's tested in real world ML
apps and holds up very well. 

I need to make the install easier - sorry. Check the Docker file for linux
prerequistes. 

Of note:

* underlying data type is float (can we template this?)
* matrices are stored in column major order (for cuda compatibility) 
* limited validation of inputs is present in this version (to be be improved) 

# Help
* need anyone who can build on windows
* (polite) suggestions for matrix operations

# Coming Soon
* NVIDIA CUDA support - beta testing in progress (20kx20k mul() in 500ms!!!)
* caffe integration - let's get the caffe chaps' good work into a real environment (running for cover)
* human brain integration - (stretch goal)

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

# API 

This is the C++ docs, which shows all the nodejs functions and a few
examples. I'll keep this up to date as the unerlying code changes.

[documentation](https://rcorbish.ydns.eu/lalg/classWrappedArray.html)

# Examples

## linear regression 
OK we'll start with a complex example, why not? It showcases the non-blocking 
features of the library. 

Who wants to wait? A skilled nodejs developer ;)

This implements the linear regression simple cals: ``` theta = inv(X' X) X' y ```
where X' is a transpose operation ( thanks MATLAB )

To run this you need fast-csv ``` npm install fast-csv```

```
var linalg = require('lalg/build/Release/lalg');
var fs = require('fs');
var csv = require("fast-csv");

const rr = fs.createReadStream('wine.csv');
var csvStream = csv() ;
rr.pipe(csvStream);

linalg.read( csvStream ) 
.then( function(X) { 
  	var y = X.removeColumn() ;
	var COV = X.transpose().mulp(X);
	return Promise.all( [ COV, X, y ] )  ;
})
.then( function(X) { 
	var Z = X[0].inv().mulp( X[1].transpose() ) ;
	return Promise.all( [ Z, X[1], X[2] ] ) ;
})
.then( function(X) { 
	return Promise.all( [ X[0].mulp( X[2] ), X[1], X[2] ] ) ;
})
.then( function(X) { 
	console.log( "Theta", X[0] ) ;
        var predicted = X[1].mul( X[0] ) ;
	console.log( "Predicted", predicted ) ;
	
})
.catch( function(err) {
	console.log( "Fail", err ) ;
});

```


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



