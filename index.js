
module.exports = require( "lalg/build/Release/lalg" ) ;


function pinv( A ) {
if( A.m > A.n ) {
	return A.transpose().mul( A ).inv().mul( A.transpose() ) ;
} else {
	return A.transpose().mul( A.mul( A.transpose() ).inv() ) ;
}
}

module.exports.pinv = pinv ;
