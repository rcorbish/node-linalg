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
	return Promise.all( [ X.transpose().mulp(X), X, y ] )  ;
})
.then( function(X) { 
	return Promise.all( [ X[0].inv().mulp( X[1].transpose() ), X[1], X[2] ] ) ;
})
.then( function(X) { 
	return Promise.all( [ X[0].mulp( X[2] ), X[1], X[2] ] ) ;
})
.then( function(X) { 
	console.log( "Theta:", X[0].transpose() ) ;
        var predicted = X[1].mul( X[0] ) ;
	console.log( "Predicted", predicted ) ;
	
})
.catch( function(err) {
	console.log( "Fail", err ) ;
});

