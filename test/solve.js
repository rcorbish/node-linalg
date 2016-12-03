
var lalg = require('lalg');

A = new lalg.Array( 2, 1, [-1, 2.5] ) ;

function userFunction1() {

	this.value = function( x ) {
		   var t1 = (1 - x.get(0));
		   var t2 = (x.get(1) - x.get(0) * x.get(0));
		   return   t1 * t1 + 100 * t2 * t2;
	} ;

	this.gradient = function( x ) {
		var rc = new lalg.Array(x.length, 1 ) ;

		var x0 = x.get(0) ;
		var x1 = x.get(1) ;
		var g0  = -2 * (1 - x0) + 200 * (x1 - x0 * x0) * (-2 * x0);
	        var g1  = 200 * (x1 - x0 * x0);

		rc.set( g0, 0 ) ;
		rc.set( g1, 1 ) ;
		return rc ;
	}
} ;

function userFunction2() {

	// f(x,y)
	this.value = function( x ) {
		return   x.get(0)*x.get(0) + x.get(1)*x.get(1) - x.get(0) - x.get(1)  ;
	} ;

	// df(x,y)/dx , df(x,y)/dy
	this.gradient = function( x ) {
		var rc = new lalg.Array(x.length, 1 ) ;
		
		rc.set( 2*x.get(0) - 1	,0 ) ;
		rc.set( 2*x.get(1) - 1	,1 ) ;
		return rc ;
	}
} ;


function userFunction3() {

	// f(x,y)
	this.value = function( x ) {
		var x2 = x.get(0)*x.get(0) ;
		var y2 = x.get(1)*x.get(1) ;

		return   x2 - y2 + x2*x2 + y2*y2 ;
	} ;

	// df(x,y)/dx , df(x,y)/dy
	this.gradient = function( x ) {
		var rc = new lalg.Array(x.length, 1 ) ;
		
		rc.set( 4*x.get(0)*x.get(0)*x.get(0) + 2*x.get(0)	,0 ) ;
		rc.set( 4*x.get(1)*x.get(1)*x.get(1) + 2*x.get(1)	,1 ) ; 
		return rc ;
	}
} ;


var f = new userFunction3() ;

var solvers = [ "BFGS", "CGD", "NEWTON","NELDERMEAD", "LBFGS","CMAES" ] ;
console.log( "PROPER ANSWER")
console.log( " 0    @ 1  , 1" ) 
console.log( "-0.5  @ 0.5, 0.5" ) 
console.log( "-0.25 @ 0  , ±0.707" ) ;

A = new lalg.Array( 2, 1, [ 0.5, -0.5] ) ;

for( var i=0 ; i<solvers.length ; i++ ) {
	console.log( solvers[i] ) ;
	var f = new userFunction1() ;
	A = new lalg.rand( 2, 1 ) ;
	A.solve( f, solvers[i] ) ;
	console.log( "Actual Min:", f.value(A), "@", Array.from(A) ) ;
	var f = new userFunction2() ;
	A = new lalg.rand( 2, 1 ) ;
	A.solve( f, solvers[i] ) ;
	console.log( "Actual Min:", f.value(A), "@", Array.from(A) ) ;
	var f = new userFunction3() ;
	A = new lalg.rand( 2, 1 ) ;
	A.solve( f, solvers[i] ) ;
	console.log( "Actual Min:", f.value(A), "@", Array.from(A) ) ;
	console.log() ;
}

/*
A = new lalg.Array( 2, 1, [ 0.1, 0.1] ) ;
var P = A.solve( f, "BFGS" ) ;
console.log( P ) ;
*/


