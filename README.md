# node-linalg

This module integrates the efficient libraries created by really smart folks
into the rich javascript environment provided by nodejs. 

I need to make the install easier - sorry. Check the Docker file for linux
prerequistes. [[ Any help with windows would be appreciated ]]

Of note:

* underlying data type is float (can we template this?)
* matrices are stored in column major order (for cuda compatibility)
* limited validation of inputs is present in this version (to be be improved) 


# Installation

I need to figure out how to do this way more automatically

## Prerequisites

* nodejs of course ( tested on 7.2.0 )
* gyp - npm install -g node-gyp
* make
* python ( oh boy! - can't avoid python - even integrating C++ to javascript !!!!! )
* C++ compiler 
* openblas  
* lapacke 

Example Dockerfile  ( again 16.04 is for the cuda compatibility )
```
FROM ubuntu:16.04

RUN apt update && \
        apt -y install curl && \
        apt -y install xz-utils && \
        curl -L -s https://nodejs.org/dist/v7.2.0/node-v7.2.0-linux-x64.tar.xz -o node-v7.2.0-linux-x64.tar.xz  && \
	tar -xJf node-v7.2.0-linux-x64.tar.xz  && \
	rm node-v7.2.0-linux-x64.tar.xz  && \
	chown -R root:root /node-v7.2.0-linux-x64/bin/node && \
	ln -s /node-v7.2.0-linux-x64/bin/node /usr/bin && \
	ln -s /node-v7.2.0-linux-x64/bin/npm /usr/bin && \
        apt -y install python && \
        apt -y install libopenblas-dev && \
        apt -y install liblapacke-dev && \
        apt -y install make && \
        apt -y install g++ && \
	npm install lalg

```

# API 

This is the C++ docs, which shows all the nodejs functions and a few
examples. I'll keep this up to date as the unerlying code changes.

[documentation](https://rcorbish.ydns.eu/lalg/classWrappedArray.html)

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



