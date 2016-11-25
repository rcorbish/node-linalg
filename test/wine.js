var linalg = require('lalg');
var fs = require('fs');
var csv = require("fast-csv");

const rr = fs.createReadStream('node_modules/lalg/data/wine.csv');
var csvStream = csv() ;
rr.pipe(csvStream);

// Implement linear regression:
// theta = inv(X' X) X' y ;


linalg.read( csvStream ) 
.then( function(X) { 
  	var y = X.removeColumn() ;
	H = X.hadamard( X )  ; // square of each element in X -> H
	X = X.appendColumns( H ) ;  // add the features to X
	return Promise.all( [ X.transpose().mulp(X), X, y ] )  ;
})
.then( function(X) { 
	return Promise.all( [ X[0].inv().mulp( X[1].transpose() ), X[1], X[2] ] ) ;
})
.then( function(X) { 
	return Promise.all( [ X[0].mulp( X[2] ), X[1], X[2] ] ) ;
})
.then( function(X) { 
	var theta = X[0].transpose() ;
	theta.name = 'Theta' ;
	theta.maxPrint = 200 ;
	console.log( "Theta:", theta ) ;
        var predicted = X[1].mul( X[0] ) ;
        predicted.maxPrint = 200 ;
        predicted.name = 'Predicted' ;
	console.log( "Predicted", predicted ) ;
	
})
.catch( function(err) {
	console.log( "Fail", err ) ;
});

