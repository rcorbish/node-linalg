
var linalg = require('./build/Release/linalg');
var fs = require('fs');
var csv = require("fast-csv");

A = linalg.rand(10) ;
var CO = A.getColumns() ;
console.log( "getColumns     ", (CO.m==A.m && CO.n==1)?"PASS":" *** FAIL ***" ) ;
CO = A.getColumns([0,1,2,3,4,5,6,7,8,9]) ;
tot = Math.abs( CO.subi(A).sum().sum() ) ;
console.log( "getColumns all ", (tot<0.001)?"PASS":" *** FAIL ***" ) ;

var RO = A.getRows() ;
console.log( "getRows        ", (RO.n==A.n && RO.m==1)?"PASS":" *** FAIL ***" ) ;
RO = A.getRows([0,1,2,3,4,5,6,7,8,9]) ;
tot = Math.abs( RO.subi(A).sum().sum() ) ;
console.log( "getRows all    ", (tot<0.001)?"PASS":" *** FAIL ***" ) ;

A.removeRow() ;
console.log( "removeRow      ", (A.m==9)?"PASS":" *** FAIL ***" ) ;

A.removeColumn() ;
console.log( "removeColumn   ", (A.n==9 && A.m==9)?"PASS":" *** FAIL ***" ) ;


const rr = fs.createReadStream('foo.csv');
var csvStream = csv() ;
rr.pipe(csvStream);

var P = linalg.read( csvStream ) ;
if( P ) {
P
.then( function(val) { 
	if( val.m == 4 ) {
		console.log( "Read Promise   ", "PASS" ) ; 
	} else {
		console.log( "Read Promise   ", " *** FAIL ***" ) ; 
	}
})
.catch( function(err) {
	console.log( "Read Promise   ", " *** FAIL ***" ) ; 
});
}

var A = new linalg.Array( 3, 3, [] ) ;
console.log( "new Array empty", (A.m==A.n && A.n==3)?"PASS":" *** FAIL ***" ) ; 

var A = new linalg.Array( 3, 3, [1,2,3,4,5,6,7,8,9] ) ;


A.mmulp(A, function(err,x) { 
        if( err ) {
		console.log( "MULP Callback  ", " *** FAIL ***" ) ; 
	} else {
		if( x.m == 3 ) {
			console.log( "MULP Callback  ", "PASS" ) ; 
		} else {
			console.log( "MULP Callback  ", " *** FAIL ***" ) ; 
		}
	}
}) ;

var P = A.mmulp(A) ; 

if( P ) {
P
.then( function(val) { 
	if( val.m == 3 ) {
		console.log( "MULP Promise   ", "PASS" ) ; 
	} else {
		console.log( "MULP Promise   ", " *** FAIL ***" ) ; 
	}
})
.catch( function(err) {
	console.log( "MULP Promise   ", " *** FAIL ***" ) ; 
});
}

B = linalg.rand(7) ;
A.mmulp(B, function(err,x) { 
        if( err ) {
		console.log( "MULP err       ", "PASS" ) ; 
	} else {
		console.log( "MULP err       ", " *** FAIL ***" ) ; 
	}
}) ;

var A = new linalg.Array( 3, 3, [1,2,3,4,5,6,7,8,9] ) ;
var B = new linalg.Array( 3, 3, [10,11,12,13,14,15,16,17,18] ) ;
A = linalg.rand( 30 ) ;
B = linalg.rand( 30 ) ;
var tot = Math.abs( A.dup().subi(A).sum().sum() ) ;
console.log( "A-A            ", (tot<0.001)?"PASS":" *** FAIL ***" ) ;

console.time("mmul");
for( var i=0 ; i<100 ; i++ ) {
	A.mmul( B ) ;
}
//console.timeEnd("mmul");

A = linalg.rand(50) ;

console.time("inv");
for( var i=0 ; i<100 ; i++ ) {
	A.invi() ;
}
//console.timeEnd("inv");

A = linalg.rand(50) ;
var tot = Math.abs( 50 - A.mmul( A.inv() ).sum().sum() ) ;
console.log( "A * inv(A)     ", (tot<0.001)?"PASS":" *** FAIL ***" ) ;
tot = A.length ;
console.log( "A.m            ", (50==A.m)?"PASS":" *** FAIL ***" ) ;
console.log( "A.n            ", (50==A.n)?"PASS":" *** FAIL ***" ) ;
console.log( "A.length       ", (tot==(A.m * A.n))?"PASS":" *** FAIL ***" ) ;

A = linalg.rand(50) ;
var S = A.sum() ;
S.subi( A.transpose().sum(1).transpose() ) ;
tot = Math.abs( S.sum() ) ;
console.log( "A transpose    ", (tot<0.001)?"PASS":" *** FAIL ***" ) ;

A = linalg.rand( 10, 7 ) ;
var svd = A.dup().svd() ;
var SS = linalg.diag( svd.S, 10 ) ;
var R = svd.U.mmul(SS).mmul(svd.VT) ;

var tot = Math.abs( R.subi(A).sum().sum() ) ;
console.log( "A.svd()        ", (tot<0.001)?"PASS":" *** FAIL ***" ) ;

A = new linalg.Array( 3,4, [1,3,5,2,6,10,3,9,15,1,3,5] ) ;
P = A.dup().pca(.99) ;
console.log( "pca(A) - linear", (P.length==4)?"PASS":" *** FAIL ***" ) ;
A = linalg.rand( 3,4 ) ;
P = A.dup().pca(1.0) ;
console.log( "pca(A) - rand  ", (P.length==12)?"PASS":" *** FAIL ***" ) ;

A = new linalg.rand( 25 ) ;
B = A.add(A) ;
var tot = Math.abs( B.sub( A.mul(2) ).sum().sum() ) ;
console.log( "mul            ", (tot<0.001)?"PASS":" *** FAIL ***" ) ;


var BIG1 = linalg.rand(3000) ;
var BIG2 = linalg.rand(3000) ;
console.log( "rand big       ", (BIG1.m==BIG1.n&&BIG2.m==BIG2.n)?"PASS":" *** FAIL ***" ) ;

console.time("mmulp 3k");
BIG1.mmulp( BIG2 )
.then( function( res ) {
  console.log( "mulp big       ", "PASS" ) ;
  console.timeEnd("mmulp 3k");
})
.catch( function( err ) {
  console.log( "mulp big       ", " *** FAIL ***" ) ;
});


