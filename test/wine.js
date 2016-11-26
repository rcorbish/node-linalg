var linalg = require('lalg');
var fs = require('fs');
var csv = require("fast-csv");

// Implement linear regression:
// theta = inv(X' X) X' y ;

var train = fs.createReadStream('node_modules/lalg/data/wine-train.csv');
var csvStreamTrain = csv() ;
train.pipe(csvStreamTrain);

var test = fs.createReadStream('node_modules/lalg/data/wine-test.csv');
var csvStreamTest = csv() ;
test.pipe(csvStreamTest);

Promise.all( [ linalg.read( csvStreamTrain) , linalg.read( csvStreamTest ) ] ) 
.then( function( data ) { 
	var X = data[0] ;
  	var y = X.removeColumn() ;
        var tmp = X.dup() ;
        for( var i=7 ; i<tmp.n ; i++ ) {		// now create new features by combining two features
  	  H = tmp.hadamard( tmp.rotateColumns(i) )  ;   // feature x * feature y -> H
	  X = X.appendColumns( H ) ;  			// add the features to X
        }
	return Promise.all( [ X.transpose().mulp(X), X, y, data[1] ] )  ;
})
.then( function(X) { 
	return Promise.all( [ X[0].inv().mulp( X[1].transpose() ), X[1], X[2], X[3] ] ) ;
})
.then( function(X) { 
	return Promise.all( [ X[0].mulp( X[2] ), X[1], X[2], X[3] ] ) ;
})
.then( function(X) { 
	var theta = Array.from( X[0] ) ;
	//console.log( theta ) ;

	var D = X[3] ;
	var y = D.removeColumn() ;

console.log( y ) ;
        var tmp = D.dup() ;
        for( var i=7 ; i<tmp.n ; i++ ) {		// now create new features by combining two features
  	  H = tmp.hadamard( tmp.rotateColumns(i) )  ;   // feature x * feature y -> H
	  D = D.appendColumns( H ) ;  			// add the features to X
        }
        var predicted = D.mul( X[0] ) ;

	predicted = predicted.appendColumns( y  ) ;
        predicted.maxPrint = 200 ;
        predicted.name = 'Predicted' ;
	console.log( predicted ) ;
})
.catch( function(err) {
	console.log( "Fail", err ) ;
});

