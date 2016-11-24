var linalg = require('./build/Release/linalg');
var fs = require('fs');
var csv = require("fast-csv");

const rr = fs.createReadStream('wine.csv');
var csvStream = csv() ;
rr.pipe(csvStream);

// Implement linear regression:
// theta = inv(X' X) X' y ;
// where X' denotes transpose of X


linalg.read( csvStream ) 
.then( function(X) { 
  	var y = X.removeColumn() ;
/*
	var factor = X.pca(.999) ;	
	console.log( "factor", factor ) ;
	var X = X.mul( factor ) ;
	console.log( "X", X ) ;
*/

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

