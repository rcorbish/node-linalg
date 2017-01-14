#include <node.h>
#include <uv.h>
#include <node_object_wrap.h>
#include <iostream>
#include <cmath>
#include <stdio.h>
#include <string.h>
#include <lapacke.h>
#include <cblas.h>
#include <thread>

#include "cppoptlib/meta.h"
#include "cppoptlib/problem.h"
#include "cppoptlib/solver/bfgssolver.h"
#include "cppoptlib/solver/conjugatedgradientdescentsolver.h"
#include "cppoptlib/solver/newtondescentsolver.h"
#include "cppoptlib/solver/neldermeadsolver.h"
#include "cppoptlib/solver/lbfgssolver.h"
#include "cppoptlib/solver/cmaessolver.h"

using namespace std;
using namespace v8;

// forward reference only
void CreateObject(const FunctionCallbackInfo<Value>& info) ;

/**
 This is the main class to represent a matrix. It is a nodejs compatible
 object.

 Matrices are stored in column order, this is to simplify access to (some)
 libraries (esp. cuda). The internal data type is always a float, which is
 good enough for most situations. The comments for the C++ code contain
 some javascript examples

*/
class WrappedArray : public node::ObjectWrap
{


  public:
    /*
       Initialize the prototype of class Array. Called when the module is loaded.
       ALl the methods and attributes are defined here.
    */
    static void Init(v8::Local<v8::Object> exports, Local<Object> module) {
      Isolate* isolate = exports->GetIsolate();

      // Prepare constructor template and name of the class
      Local<FunctionTemplate> tpl = FunctionTemplate::New(isolate, New);
      tpl->SetClassName(String::NewFromUtf8(isolate, "Array"));
      tpl->InstanceTemplate()->SetInternalFieldCount(1);


      // Prototype - methods. These can be called from javascript
      NODE_SET_PROTOTYPE_METHOD(tpl, "toString", ToString);
      NODE_SET_PROTOTYPE_METHOD(tpl, "inspect", Inspect);
      NODE_SET_PROTOTYPE_METHOD(tpl, "dup", Dup);
      NODE_SET_PROTOTYPE_METHOD(tpl, "transpose", Transpose);
      NODE_SET_PROTOTYPE_METHOD(tpl, "mul", Mul);
      NODE_SET_PROTOTYPE_METHOD(tpl, "mulp", Mulp);
      NODE_SET_PROTOTYPE_METHOD(tpl, "hadamard", Hadamard);
      NODE_SET_PROTOTYPE_METHOD(tpl, "asum", Asum);
      NODE_SET_PROTOTYPE_METHOD(tpl, "sum", Sum);
      NODE_SET_PROTOTYPE_METHOD(tpl, "mean", Mean);
      NODE_SET_PROTOTYPE_METHOD(tpl, "norm", Norm);
      NODE_SET_PROTOTYPE_METHOD(tpl, "add", Add);
      NODE_SET_PROTOTYPE_METHOD(tpl, "sub", Sub);
      NODE_SET_PROTOTYPE_METHOD(tpl, "find", Find);
      NODE_SET_PROTOTYPE_METHOD(tpl, "findGreater", FindGreater);
      NODE_SET_PROTOTYPE_METHOD(tpl, "findLessEqual", FindLessEqual);
      NODE_SET_PROTOTYPE_METHOD(tpl, "neg", Neg);
      NODE_SET_PROTOTYPE_METHOD(tpl, "log", Log);
      NODE_SET_PROTOTYPE_METHOD(tpl, "sqrt", Sqrt);
      NODE_SET_PROTOTYPE_METHOD(tpl, "abs", Abs);
      NODE_SET_PROTOTYPE_METHOD(tpl, "inv", Inv);
      NODE_SET_PROTOTYPE_METHOD(tpl, "invp", Invp);
      NODE_SET_PROTOTYPE_METHOD(tpl, "pinv", Pinv);
      NODE_SET_PROTOTYPE_METHOD(tpl, "svd", Svd);
      NODE_SET_PROTOTYPE_METHOD(tpl, "pca", Pca);
      NODE_SET_PROTOTYPE_METHOD(tpl, "getRows", GetRows);
      NODE_SET_PROTOTYPE_METHOD(tpl, "removeRow", RemoveRow);
      NODE_SET_PROTOTYPE_METHOD(tpl, "getColumns", GetColumns);
      NODE_SET_PROTOTYPE_METHOD(tpl, "appendColumns", AppendColumns );
      NODE_SET_PROTOTYPE_METHOD(tpl, "removeColumn", RemoveColumn);
      NODE_SET_PROTOTYPE_METHOD(tpl, "rotateColumns", RotateColumns);
      NODE_SET_PROTOTYPE_METHOD(tpl, "reshape", Reshape);
      NODE_SET_PROTOTYPE_METHOD(tpl, "solve", Solve);
      //NODE_SET_PROTOTYPE_METHOD(tpl, "solvep", Solvep);
      NODE_SET_PROTOTYPE_METHOD(tpl, "set", Set);
      NODE_SET_PROTOTYPE_METHOD(tpl, "get", Get);


     // the macro (or inline) doesn't work for a symbol
     // NODE_SET_PROTOTYPE_METHOD(tpl, Symbol::GetIterator(isolate), MakeIterator);
     // do this by hand to use Symbol::GetIterator
      Local<FunctionTemplate> t = v8::FunctionTemplate::New(isolate, MakeIterator);
      t->SetClassName( String::NewFromUtf8(isolate, "values") );
      tpl->PrototypeTemplate()->Set(Symbol::GetIterator(isolate), t);

      // Factories - call these on the module - to create a new array
      NODE_SET_METHOD(exports, "eye", Eye);
      NODE_SET_METHOD(exports, "ones", Ones);
      NODE_SET_METHOD(exports, "zeros", Zeros);
      NODE_SET_METHOD(exports, "rand", Rand);
      NODE_SET_METHOD(exports, "diag", Diag);
      NODE_SET_METHOD(exports, "read", Read);

      // define how we access the attributes
   //   tpl->InstanceTemplate()->SetAccessor(Local<String>::Cast( Symbol::GetIterator(isolate) ) , GetCoeff);
      tpl->InstanceTemplate()->SetAccessor(String::NewFromUtf8(isolate, "m"), GetCoeff);
      tpl->InstanceTemplate()->SetAccessor(String::NewFromUtf8(isolate, "n"), GetCoeff);
      tpl->InstanceTemplate()->SetAccessor(String::NewFromUtf8(isolate, "length"), GetCoeff);
      tpl->InstanceTemplate()->SetAccessor(String::NewFromUtf8(isolate, "maxPrint"), GetCoeff, SetCoeff);
      tpl->InstanceTemplate()->SetAccessor(String::NewFromUtf8(isolate, "name"), GetCoeff, SetCoeff);

      constructor.Reset(isolate, tpl->GetFunction());
      exports->Set(String::NewFromUtf8(isolate, "Array"), tpl->GetFunction());
    }
    /*
	The nodejs constructor. 
	@see New
    */
    static Local<Object> NewInstance(const FunctionCallbackInfo<Value>& args);
  private: 
   /*
	The C++ constructor, creates an mxn array.
   */
    explicit WrappedArray(int m=0, int n=0) : m_(m), n_(n) {
      dataSize_ = m*n ;
      data_ = new float[dataSize_] ;
      isVector = m==1 || n== 1 ;
      name_ = NULL ;
      maxPrint_ = 10 ;
    }
    /*
	The destructor needs to free the data buffer
    */
    ~WrappedArray() { 
	delete data_ ;
        delete name_ ;
    }

    /*
	The javascript constructor.
	It takes up to 3 args: 
	@param [in] number of rows (m) defaults to 0
	@param [in] number of columns (n) defaults to m
	@param [in] optional an array of data to use for the array (column major order) or a number to put in all elements
    */
    static void New(const v8::FunctionCallbackInfo<v8::Value>& args) {
      Isolate* isolate = args.GetIsolate();

      // in a constructor? Which one?

      if (args.IsConstructCall()) {

        WrappedArray* self = NULL ;	// will fill this up later

        int m = args[0]->IsUndefined() ? 0 : args[0]->NumberValue();
        int n = args[1]->IsUndefined() ? m : args[1]->NumberValue();

	// Do we have an array to use for data?
        if( !args[2]->IsUndefined() && args[2]->IsArray() ) {

          Local<Array> buffer = Local<Array>::Cast(args[2]->ToObject() );

          Local<Context> context = isolate->GetCurrentContext() ;
          self = Create(m, n, context, buffer )	;
        } else {  // No array? just create it
          self = Create(m, n) ;
          if( !args[2]->IsUndefined() && args[2]->IsNumber() ) {
	    float v = args[2]->NumberValue() ;
	    for( int i=0 ; i<m*n ; ++i) {
	      self->data_[i] = v ;
	    }
	  }
        }

        if( self != NULL ) {  // double check we managed to create something
          self->Wrap(args.This());
//          args.This()->SetInternalField(0, External::New(isolate, self));
          args.GetReturnValue().Set(args.This());
        }
        else {
          args.GetReturnValue().Set( Undefined(isolate) ) ;
        }
      }
    }

/**
	Create an array with some data provided in a javascript Array
*/
    static WrappedArray *Create( 
	int m, /**< [in] number of rows in the matrix */
	int n, /**< [in] number of cols in the matrix */
	Local<Context> context, Local<Array> array /**< [in] data to put into the array - column major order */ 
	) {

      WrappedArray* self = new WrappedArray(m, n) ;

	// Copy the minimum of ( input array size & internal buffer size ) numbers
        // into the internal buffer
        int l = array->Length() ;
        l = std::min( self->m_*self->n_, l ) ;
        for( int i=0 ; i<l ; i++ ) {
          self->data_[i] = array->Get( context, i ).ToLocalChecked()->NumberValue() ;
        }
      return self ;
    }

/**
	Create an array with some (optional) data. This will be the base call
	for creation for internal methods.
	
*/
    static WrappedArray *Create( 
	int m, /**< [in] number of rows in the matrix */
	int n, /**< [in] number of cols in the matrix */
	float *data = NULL /**< [in] data to put into the array - column major order */ 
	) {
      WrappedArray* self = new WrappedArray(m, n);

      if( data != NULL ) {
        for( int i=0 ; i<(m*n) ; i++ ) {
          self->data_[i] = data[i] ;
        }
      }

      return self ;
    }

    static void ToString(const FunctionCallbackInfo<Value>& args);
    static void Inspect(const FunctionCallbackInfo<Value>& args);
    static void Ones(const FunctionCallbackInfo<Value>& args );
    static void Zeros(const FunctionCallbackInfo<Value>& args );
    static void Eye(const FunctionCallbackInfo<Value>& args );
    static void Diag(const FunctionCallbackInfo<Value>& args );
    static void Read(const FunctionCallbackInfo<Value>& args );
    static void Rand(const FunctionCallbackInfo<Value>& args );
    static void Dup(const FunctionCallbackInfo<Value>& args );
    static void Find(const FunctionCallbackInfo<Value>& args );
    static void FindGreater(const FunctionCallbackInfo<Value>& args );
    static void FindLessEqual(const FunctionCallbackInfo<Value>& args );
    static void Neg(const FunctionCallbackInfo<Value>& args );
    static void Log(const FunctionCallbackInfo<Value>& args );
    static void Sqrt(const FunctionCallbackInfo<Value>& args );
    static void Abs(const FunctionCallbackInfo<Value>& args );
    static void Transpose(const FunctionCallbackInfo<Value>& args );
    static void Hadamard(const FunctionCallbackInfo<Value>& args );
    static void Mul(const FunctionCallbackInfo<Value>& args );
    static void Mulp(const FunctionCallbackInfo<Value>& args );
    static void Asum( const FunctionCallbackInfo<v8::Value>& args  );
    static void Sum( const FunctionCallbackInfo<v8::Value>& args  );
    static void Mean( const FunctionCallbackInfo<v8::Value>& args  );
    static void Norm( const FunctionCallbackInfo<v8::Value>& args  );
    static void Add( const FunctionCallbackInfo<v8::Value>& args  );
    static void Sub( const FunctionCallbackInfo<v8::Value>& args  );
    static void Inv( const FunctionCallbackInfo<v8::Value>& args  );    
    static void Invp( const FunctionCallbackInfo<v8::Value>& args  );    
    static void Pinv( const FunctionCallbackInfo<v8::Value>& args  );
    static void Svd( const FunctionCallbackInfo<v8::Value>& args  );
    static void Pca( const FunctionCallbackInfo<v8::Value>& args  );
    static void GetRows( const FunctionCallbackInfo<v8::Value>& args  );
    static void RemoveRow( const FunctionCallbackInfo<v8::Value>& args  );
    static void GetColumns( const FunctionCallbackInfo<v8::Value>& args  );
    static void RemoveColumn( const FunctionCallbackInfo<v8::Value>& args  );
    static void AppendColumns( const FunctionCallbackInfo<v8::Value>& args  );
    static void RotateColumns( const FunctionCallbackInfo<v8::Value>& args  );
    static void Reshape( const FunctionCallbackInfo<v8::Value>& args  );
    static void Solve( const v8::FunctionCallbackInfo<v8::Value>& args ) ;
    static void Solvep( const v8::FunctionCallbackInfo<v8::Value>& args ) ;
    static void Get( const FunctionCallbackInfo<v8::Value>& args  );
    static void Set( const FunctionCallbackInfo<v8::Value>& args  );

    static void MakeIterator( const FunctionCallbackInfo<v8::Value>& args  );

    static void GetCoeff(Local<String> property, const PropertyCallbackInfo<Value>& info);
    static void SetCoeff(Local<String> property, Local<Value> value, const PropertyCallbackInfo<void>& info);

    static v8::Persistent<v8::Function> constructor; /**< a nodejs constructor for this object */
    int m_;  /**< the number of rows in the matrix */
    int n_;  /**< the number of columns in the matrix */
    float *data_ ;   /**< the data buffer holding the values */
    bool isVector ; /**< helper flag to see whether the target is a vector Mx1 or 1xN */
    int dataSize_ ;  /**< private - used to remember the last data allocation size */
    int maxPrint_ ;  /**< the number of rows & columns to print out in toString() */
    char *name_ ; /**< The name of this matrix - useful for keeping track of things */

    static void NextCallback( const v8::FunctionCallbackInfo<v8::Value>& args ) ;
    static void DataCallback(const FunctionCallbackInfo<Value>& args) ;
    static void DataEndCallback(const FunctionCallbackInfo<Value>& args) ;
    static void PrepareWork( const v8::FunctionCallbackInfo<v8::Value>& args, bool block, int callbackIndex, uv_work_cb work_cb, Local<Object> instance, Local<Object> xtraObj, int xtraInt  ) ;
    static void PrepareWork( const v8::FunctionCallbackInfo<v8::Value>& args, bool block, int callbackIndex, uv_work_cb work_cb, Local<Object> instance ) ;

    static void WorkAsyncComplete(uv_work_t *req,int status) ;
    static void InvpWorkAsync(uv_work_t *req) ;
    static void MulpWorkAsync(uv_work_t *req) ;
    static void ReadWorkAsync(uv_work_t *req) ;
    static void MulHelper( const v8::FunctionCallbackInfo<v8::Value>& args, bool block, int callbackIndex ) ;
    static void InvHelper( const v8::FunctionCallbackInfo<v8::Value>& args, bool block, int callbackIndex ) ;

    static void SolveWorkAsync(uv_work_t *req) ;
    static void SolveHelper( const v8::FunctionCallbackInfo<v8::Value>& args, bool block, int callbackIndex ) ;


    struct Work {
      uv_work_t  request;
      Persistent<Function> callback;
      Persistent<Promise::Resolver> resolver ;
      WrappedArray* self ;
      WrappedArray* other;
      float otherNumber ;
      WrappedArray* result ;
      char *err ;
      Persistent<Object> resultLocal;
      Persistent<Object> selfObj;
      Persistent<Object> xtraObj;
      Isolate *isolate ;
      int xtraInt;
    } ;

class UserGradientFunction : public cppoptlib::Problem<float, 2> {
  public:
    using typename cppoptlib::Problem<float, 2>::TVector;
    using typename cppoptlib::Problem<float, 2>::THessian;


    UserGradientFunction( Isolate* isolate, WrappedArray* self, Local<Object> instance, Local<Object> suppliedFunctionHolder ) {

	Local<Context> context = isolate->GetCurrentContext() ;
/*
  {
    Local<String> className = suppliedFunctionHolder->GetConstructorName() ;
    char *c = new char[className->Utf8Length() + 10 ] ;
    className->WriteUtf8( c ) ;
    printf( "Self sfh ... %s\n", c ) ;
    delete c ;
  }
*/
	Local<String> valueKey = String::NewFromUtf8(isolate, "value") ;
suppliedFunctionHolder->GetPropertyNames( context ) ;
        MaybeLocal<Value> mlv = suppliedFunctionHolder->Get( context, String::NewFromUtf8(isolate, "value") ) ;

	Local<Value> valuetmp = suppliedFunctionHolder->Get( context, String::NewFromUtf8(isolate, "value") ).ToLocalChecked() ;
	value_ = Local<Function>::Cast( valuetmp ) ;
	Local<Value> gradtmp = suppliedFunctionHolder->Get( context, String::NewFromUtf8(isolate, "gradient") ).ToLocalChecked() ;
	gradient_ = Local<Function>::Cast( gradtmp ) ;
	self_ = self ;
	isolate_ = isolate ;
	instance_ = instance ;
    }
    
    float value(const TVector &x) {
	for( int i=0 ; i<self_->m_ * self_->n_ ; i++ ) {
		self_->data_[i] = x[i] ;
	}
        Handle<Value> argv[] = { instance_ } ;
        MaybeLocal<Value> value = value_ -> Call(isolate_->GetCurrentContext()->Global(), 1, argv) ;
	for( int i=0 ; i<self_->m_ * self_->n_ ; i++ ) {
		self_->data_[i] = x[i] ;
	}
  	
	float rc = value.ToLocalChecked()->NumberValue() ;
	return rc ;
    }


    void gradient(const TVector &x, TVector &grad) {
	for( int i=0 ; i<self_->m_ * self_->n_ ; i++ ) {
		self_->data_[i] = x[i] ;
	}
        Handle<Value> argv[] = { instance_ } ;
        Local<Value> obj = gradient_ -> Call(isolate_->GetCurrentContext()->Global(), 1, argv) ;

	WrappedArray *gradient = WrappedArray::ObjectWrap::Unwrap<WrappedArray>( obj->ToObject() );

	for( int i=0 ; i<gradient->m_*gradient->n_ ; i++ ) {
	  grad[i] = gradient->data_[i] ;
	}
    }

   private:
	Local<Function> value_ ;
	Local<Function> gradient_ ;
	WrappedArray* self_ ;
	Isolate* isolate_ ;
	Local<Object> instance_ ;

};


} ;
Persistent<Function> WrappedArray::constructor;

Local<Object> WrappedArray::NewInstance(const FunctionCallbackInfo<Value>& args)
{
  Isolate* isolate = args.GetIsolate();
  Local<Context> context = isolate->GetCurrentContext() ;
  EscapableHandleScope scope(isolate) ; ;

  const unsigned argc = 2;
  Local<Value> argv[argc] = { args[0], args[1] };
  Local<Function> cons = Local<Function>::New(isolate, constructor);
  MaybeLocal<Object> instance = cons->NewInstance(context, argc, argv);

  return scope.Escape(instance.ToLocalChecked() );
}

/** 
	returns a string representation of the matrix 

	This will print up to maxPrint rows and columns
*/
void WrappedArray::ToString( const v8::FunctionCallbackInfo<v8::Value>& args )
{
  Isolate* isolate = args.GetIsolate();
//  Local<Context> context = isolate->GetCurrentContext() ;

  WrappedArray* self = ObjectWrap::Unwrap<WrappedArray>(args.Holder());

  int mm = std::min( self->m_, self->maxPrint_ ) ;
  int nn = std::min( self->n_, self->maxPrint_ ) ;
  char *rc = new char[ 100 + (mm * nn * 50 * self->maxPrint_ ) ] ;   // allocate a big array. This can still overflow :( TODO: fix this crap
  int n = 0 ;
  if( self->name_ != NULL ) {
  	n = sprintf( rc, "[ %s ] ", self->name_ ) ;
  }
  n += sprintf( rc+n, "%d x %d %s\n", self->m_, self->n_, self->isVector?"Vector" : "" ) ;

  for( int r=0 ; r<mm ; r++ ) {
    for( int c=0 ; c<nn ; c++ ) {
      n += sprintf( rc+n, "% 6.2f ", self->data_[ c * self->m_ + r ] ) ;
    }
    if( self->n_ > nn ) {
      strcat( rc, " ..." ) ;
      n+=4 ;
    }
    strcat( rc, "\n" ) ;
    n++ ;
  }
  if( self->m_ > mm ) {
    for( int i=0 ; i<nn ; i++ ) {
      strcat( rc, "  ...  " ) ;
    }
    strcat( rc, "\n" ) ;
  }

  args.GetReturnValue().Set( String::NewFromUtf8( isolate, rc) );
  delete  rc ;
}

/** internally calls ToString. @see ToString */
void WrappedArray::Inspect( const v8::FunctionCallbackInfo<v8::Value>& args )
{
  ToString( args ) ;
}



/** 
	Get a value from the matrix

	Returns a value from the matrix at the X,Y. If a single index is given it is used an an absolute
	address into the underlying store. Negative indices count from the end.

	@param [in,default=0] M the row index ( or the absolute index is N is missing )
	@param [in,optional] N the column index
	@return the value in the matrix at required
*/
void WrappedArray::Get( const v8::FunctionCallbackInfo<v8::Value>& args )
{
  Isolate* isolate = args.GetIsolate();
//  Local<Context> context = isolate->GetCurrentContext() ;

  WrappedArray* self = ObjectWrap::Unwrap<WrappedArray>(args.Holder());

  int m = args[0]->IsUndefined() ? 0 : args[0]->NumberValue() ;
  if( args[1]->IsUndefined() ) {
    if( ::abs(m)>=(self->m_ * self->n_)) {
      char *msg = new char[1000] ;
      snprintf( msg, 1000, "Array index (%d) out of bounds, max length is %d", m, ((self->m_ * self->n_)-1) ) ;
      Local<String> err = String::NewFromUtf8(isolate, msg);
      isolate->ThrowException(Exception::TypeError( err ) );
      delete msg ;
    } else {
      if( m<0) m = (self->m_ * self->n_) - m - 1 ;
      args.GetReturnValue().Set( self->data_[m] );
    }
  } else {
    int n = args[1]->NumberValue() ;
    if( ::abs(m)>=self->m_ || ::abs(n)>=self->n_) {
      char *msg = new char[1000] ;
      snprintf( msg, 1000, "Array index (%d,%d) out of bounds  for matrix |%d x %d|", m, n, self->m_, self->n_ ) ;
      Local<String> err = String::NewFromUtf8(isolate, msg);
      isolate->ThrowException(Exception::TypeError( err ) );
      delete msg ;
    } else {
      if( m<0) m = self->m_ - m - 1 ;
      if( n<0) n = self->n_ - n - 1 ;
      args.GetReturnValue().Set( self->data_[m+n*self->m_] );
    }
  }
}



/** 
	Set a value in the matrix

	Overwrites a the value in the matrix at the X,Y. If the target is a vector
	only the first argument is used. If a single index is given it is used an an absolute
	address into the underlying store. Negative indices count from the end.

	@param [in] value to set into the matrix
	@param [in,default=0] M the row index ( or the absolute index is N is missing )
	@param [in,optional] N the column index
	@return the previous value in the matrix at location (m,n)
*/
void WrappedArray::Set( const v8::FunctionCallbackInfo<v8::Value>& args )
{
  Isolate* isolate = args.GetIsolate();
//  Local<Context> context = isolate->GetCurrentContext() ;

  WrappedArray* self = ObjectWrap::Unwrap<WrappedArray>(args.Holder());

  if( args[0]->IsUndefined() ) {
    Local<String> err = String::NewFromUtf8(isolate, "Missing value to set into a matrix");
    isolate->ThrowException(Exception::TypeError( err ) );
  } else {
    float newValue = args[0]->NumberValue() ;
    int m = args[1]->IsUndefined() ? 0 : args[1]->NumberValue() ;
    if( args[2]->IsUndefined() ) {
      if( ::abs(m)>=(self->m_ * self->n_)) {
        char *msg = new char[1000] ;
        snprintf( msg, 1000, "Array index (%d) out of bounds, max length is %d", m, ((self->m_ * self->n_)-1) ) ;
        Local<String> err = String::NewFromUtf8(isolate, msg);
        isolate->ThrowException(Exception::TypeError( err ) );
        delete msg ;
      } else {
	if( m<0) m = (self->m_ * self->n_) - m - 1 ;
        args.GetReturnValue().Set( self->data_[m] );
        self->data_[m] = newValue ;
      }
    } else {
      int n = args[2]->NumberValue() ;
      if( ::abs(m)>=self->m_ || ::abs(n)>=self->n_) {
        char *msg = new char[1000] ;
        snprintf( msg, 1000, "Array index (%d,%d) out of bounds  for matrix |%d x %d|", m, n, self->m_, self->n_ ) ;
        Local<String> err = String::NewFromUtf8(isolate, msg);
        isolate->ThrowException(Exception::TypeError( err ) );
        delete msg ;
      } else {
	if( m<0) m = self->m_ - m - 1 ;
	if( n<0) n = self->n_ - n - 1 ;
        args.GetReturnValue().Set( self->data_[m+n*self->m_] );
        self->data_[m+n*self->m_] = newValue ;
      }
    }
  }
}



/** 
	Duplicate a matrix

	Returns a new matrix which is an identical copy of the target
*/
void WrappedArray::Dup( const v8::FunctionCallbackInfo<v8::Value>& args )
{
  Isolate* isolate = args.GetIsolate();
  Local<Context> context = isolate->GetCurrentContext() ;

  WrappedArray* self = ObjectWrap::Unwrap<WrappedArray>(args.Holder());
  EscapableHandleScope scope(isolate) ; ;

  const unsigned argc = 2;
  Local<Value> argv[argc] = { Integer::New( isolate,self->m_ ), Integer::New( isolate,self->n_ ) };
  Local<Function> cons = Local<Function>::New(isolate, constructor);
  Local<Object> instance = cons->NewInstance(context, argc, argv).ToLocalChecked() ;

  scope.Escape( instance );

  WrappedArray* result = node::ObjectWrap::Unwrap<WrappedArray>( instance ) ;
  args.GetReturnValue().Set( instance );
  memcpy( result->data_, self->data_, sizeof(float) * self->m_ * self->n_ ) ;
}


/** 
	Negate a matrix

	Returns a new matrix which is an copy of the target with each element sign reversed

	It takes 0 args:
*/
void WrappedArray::Neg( const v8::FunctionCallbackInfo<v8::Value>& args )
{
  Isolate* isolate = args.GetIsolate();
  Local<Context> context = isolate->GetCurrentContext() ;

  WrappedArray* self = ObjectWrap::Unwrap<WrappedArray>(args.Holder());
  EscapableHandleScope scope(isolate) ; ;

  const unsigned argc = 2;
  Local<Value> argv[argc] = { Integer::New( isolate,self->m_ ), Integer::New( isolate,self->n_ ) };
  Local<Function> cons = Local<Function>::New(isolate, constructor);
  Local<Object> instance = cons->NewInstance(context, argc, argv).ToLocalChecked() ;

  scope.Escape( instance );

  WrappedArray* result = node::ObjectWrap::Unwrap<WrappedArray>( instance ) ;
  args.GetReturnValue().Set( instance );
  int sz = self->m_ * self->n_  ;
  for( int i=0 ; i<sz ; i++ ) {
	result->data_[i] = -self->data_[i] ;
  }
}


/** 
	Square root a matrix

	Returns a new matrix which is an copy of the target with each element 
	being the square root of the target. Negative target values will result
	in NaN entries.

	It takes 0 args:
*/
void WrappedArray::Sqrt( const v8::FunctionCallbackInfo<v8::Value>& args )
{
  Isolate* isolate = args.GetIsolate();
  Local<Context> context = isolate->GetCurrentContext() ;

  WrappedArray* self = ObjectWrap::Unwrap<WrappedArray>(args.Holder());
  EscapableHandleScope scope(isolate) ; ;

  const unsigned argc = 2;
  Local<Value> argv[argc] = { Integer::New( isolate,self->m_ ), Integer::New( isolate,self->n_ ) };
  Local<Function> cons = Local<Function>::New(isolate, constructor);
  Local<Object> instance = cons->NewInstance(context, argc, argv).ToLocalChecked() ;

  scope.Escape( instance );

  WrappedArray* result = node::ObjectWrap::Unwrap<WrappedArray>( instance ) ;
  args.GetReturnValue().Set( instance );
  int sz = self->m_ * self->n_  ;
  for( int i=0 ; i<sz ; i++ ) {
	result->data_[i] = ::sqrt( self->data_[i] ) ;
  }
}

/** 
	Log a matrix

	Returns a new matrix which is an copy of the target with each element being the 
	natural log of the target. Since we're not doing complex values (yet:o), negative
	and zero values will result in NaN

	It takes 0 args:
*/
void WrappedArray::Log( const v8::FunctionCallbackInfo<v8::Value>& args )
{
  Isolate* isolate = args.GetIsolate();
  Local<Context> context = isolate->GetCurrentContext() ;

  WrappedArray* self = ObjectWrap::Unwrap<WrappedArray>(args.Holder());
  EscapableHandleScope scope(isolate) ;

  const unsigned argc = 2;
  Local<Value> argv[argc] = { Integer::New( isolate,self->m_ ), Integer::New( isolate,self->n_ ) };
  Local<Function> cons = Local<Function>::New(isolate, constructor);
  Local<Object> instance = cons->NewInstance(context, argc, argv).ToLocalChecked() ;

  scope.Escape( instance );

  WrappedArray* result = node::ObjectWrap::Unwrap<WrappedArray>( instance ) ;
  args.GetReturnValue().Set( instance );
  int sz = self->m_ * self->n_  ;
  for( int i=0 ; i<sz ; i++ ) {
	result->data_[i] = ::log( self->data_[i] ) ;
  }
}


/** 
	Absolute value of matrix

	Returns a new matrix which is an copy of the target with each element set to positive

	It takes 0 args:
*/
void WrappedArray::Abs( const v8::FunctionCallbackInfo<v8::Value>& args )
{
  Isolate* isolate = args.GetIsolate();
  Local<Context> context = isolate->GetCurrentContext() ;

  WrappedArray* self = ObjectWrap::Unwrap<WrappedArray>(args.Holder());
  EscapableHandleScope scope(isolate) ; ;

  const unsigned argc = 2;
  Local<Value> argv[argc] = { Integer::New( isolate,self->m_ ), Integer::New( isolate,self->n_ ) };
  Local<Function> cons = Local<Function>::New(isolate, constructor);
  Local<Object> instance = cons->NewInstance(context, argc, argv).ToLocalChecked() ;

  scope.Escape( instance );

  WrappedArray* result = node::ObjectWrap::Unwrap<WrappedArray>( instance ) ;
  args.GetReturnValue().Set( instance );
  int sz = self->m_ * self->n_  ;
  for( int i=0 ; i<sz ; i++ ) {
	result->data_[i] = ::abs( self->data_[i] ) ;
  }
}


/** 
	Find matching values in a matrix

	Returns a new matrix where a matching element is set to the given parameter
	if the value matches the supplied number. An optional parameter defines the
	match accuracy ( since we are working with real numbers ). If the 2nd parameter
	is not given the orginal target element is dropped in.

	@param [in] the number to find ( default 1 )
	@param [in] the value to pop into the matching elements
	@param [in] the accuracy to find a match default 0.000001
*/
void WrappedArray::Find( const v8::FunctionCallbackInfo<v8::Value>& args )
{
  Isolate* isolate = args.GetIsolate();
  Local<Context> context = isolate->GetCurrentContext() ;

  WrappedArray* self = ObjectWrap::Unwrap<WrappedArray>(args.Holder());
  EscapableHandleScope scope(isolate) ; ;

  const unsigned argc = 2;
  Local<Value> argv[argc] = { Integer::New( isolate,self->m_ ), Integer::New( isolate,self->n_ ) };
  Local<Function> cons = Local<Function>::New(isolate, constructor);
  Local<Object> instance = cons->NewInstance(context, argc, argv).ToLocalChecked() ;

  scope.Escape( instance );

  float x = args[0]->IsUndefined() ? 1 : args[0]->NumberValue() ;
  float epsilon = (args[1]->IsUndefined() || args[1]->IsNull() ) ? 0.000001 : args[1]->NumberValue() ;
	
  WrappedArray* result = node::ObjectWrap::Unwrap<WrappedArray>( instance ) ;
  args.GetReturnValue().Set( instance );
  int sz = self->m_ * self->n_  ;

  if( args[2]->IsUndefined() ) {
    for( int i=0 ; i<sz ; i++ ) {
      result->data_[i] = ::abs( self->data_[i] - x ) < epsilon ? self->data_[i] : 0.0 ;
    }
  }  else {
    float v = args[2]->IsUndefined() ? 0 : args[2]->NumberValue() ;
    for( int i=0 ; i<sz ; i++ ) {
      result->data_[i] = ::abs( self->data_[i] - x ) < epsilon ? v : 0.0 ;
    }
  }
}


/** 
	Keep less (or same) values in a matrix

	Returns a new matrix where a matching element is set to the supplied value if an element 
	is greater than the supplied number, otherwise it's the target value. In other words
	keep the values that are <= than the input and zero the rest.
	If the 2nd parameter is not given (null) the orginal target element is dropped in.

	@param [in] the number to find ( default 1 )
	@param [in] the value to pop into the matching elements

*/
void WrappedArray::FindLessEqual( const v8::FunctionCallbackInfo<v8::Value>& args )
{
  Isolate* isolate = args.GetIsolate();
  Local<Context> context = isolate->GetCurrentContext() ;

  WrappedArray* self = ObjectWrap::Unwrap<WrappedArray>(args.Holder());
  EscapableHandleScope scope(isolate) ; ;

  const unsigned argc = 2;
  Local<Value> argv[argc] = { Integer::New( isolate,self->m_ ), Integer::New( isolate,self->n_ ) };
  Local<Function> cons = Local<Function>::New(isolate, constructor);
  Local<Object> instance = cons->NewInstance(context, argc, argv).ToLocalChecked() ;

  scope.Escape( instance );

  float x = args[0]->IsUndefined() ? 1 : args[0]->NumberValue() ;


  WrappedArray* result = node::ObjectWrap::Unwrap<WrappedArray>( instance ) ;
  args.GetReturnValue().Set( instance );
  int sz = self->m_ * self->n_  ;

  if( args[1]->IsUndefined() ) {
    for( int i=0 ; i<sz ; i++ ) {
      result->data_[i] = self->data_[i] <= x ? self->data_[i] : 0.0 ;
    }
  }  else {
    float v = args[1]->IsUndefined() ? 0 : args[1]->NumberValue() ;
    for( int i=0 ; i<sz ; i++ ) {
      result->data_[i] = self->data_[i] <= x ? v : 0.0 ;
    }
  }
}


/** 
	Keep greater values in a matrix

	Returns a new matrix where a matching element is set to supplied value if an element 
	is less than or equal to the supplied number, otherwise it's the target value. 
	In other words keep the values that are > than the input and zero the rest. 
	If the 2nd parameter is not given (null) the orginal target element is dropped in.

	@param [in] the number to find ( default 1 )
	@param [in] the value to pop into the matching elements

*/
void WrappedArray::FindGreater( const v8::FunctionCallbackInfo<v8::Value>& args )
{
  Isolate* isolate = args.GetIsolate();
  Local<Context> context = isolate->GetCurrentContext() ;

  WrappedArray* self = ObjectWrap::Unwrap<WrappedArray>(args.Holder());
  EscapableHandleScope scope(isolate) ; ;

  const unsigned argc = 2;
  Local<Value> argv[argc] = { Integer::New( isolate,self->m_ ), Integer::New( isolate,self->n_ ) };
  Local<Function> cons = Local<Function>::New(isolate, constructor);
  Local<Object> instance = cons->NewInstance(context, argc, argv).ToLocalChecked() ;

  scope.Escape( instance );

  float x = args[0]->IsUndefined() ? 1 : args[0]->NumberValue() ;

  WrappedArray* result = node::ObjectWrap::Unwrap<WrappedArray>( instance ) ;
  args.GetReturnValue().Set( instance );
  int sz = self->m_ * self->n_  ;
  if( args[1]->IsUndefined() ) {
    for( int i=0 ; i<sz ; i++ ) {
      result->data_[i] = self->data_[i] > x ? self->data_[i] : 0.0 ;
    }
  }  else {
    float v = args[1]->IsUndefined() ? 0 : args[1]->NumberValue() ;
    for( int i=0 ; i<sz ; i++ ) {
      result->data_[i] = self->data_[i] > x ? v : 0.0 ;
    }
  }
}

/** 
	Transpose a matrix

	Returns a new matrix which is an transpose of the target.

	It takes 0 args:
*/
void WrappedArray::Transpose( const v8::FunctionCallbackInfo<v8::Value>& args )
{
  Isolate* isolate = args.GetIsolate();
  Local<Context> context = isolate->GetCurrentContext() ;

  WrappedArray* self = ObjectWrap::Unwrap<WrappedArray>(args.Holder());

  EscapableHandleScope scope(isolate) ; ;

  // Create a new instance of ourself, with inverted dimensions. The data is uninitialized
  const unsigned argc = 2;
  Local<Value> argv[argc] = { Integer::New( isolate,self->n_ ), Integer::New( isolate,self->m_ ) };
  Local<Function> cons = Local<Function>::New(isolate, constructor);
  Local<Object> instance = cons->NewInstance(context, argc, argv).ToLocalChecked() ;

  scope.Escape( instance );
  WrappedArray* result = node::ObjectWrap::Unwrap<WrappedArray>( instance ) ;

// Now do the transpose. It's easy since we have a new memory buffer
// in palce transposition is difficult and slow
  if( !result->isVector ) {	// no need to transpose a vector's data
    float *data = result->data_ ;
    int ixa = 0 ;
    for( int r=0 ; r<self->m_ ; r++ ) {
      int ixb = r ;
      for( int c=0 ; c<self->n_ ; c++ ) {
        data[ixa++] = self->data_[ixb+(c*self->m_)] ;
      }
    }
  }
  else {
    memcpy( result->data_, self->data_, self->m_ * self->n_ * sizeof(float) ) ;
  }

// Set the resturn to be the new copy of ourself
  args.GetReturnValue().Set( instance );
}



/**
	Mul - multiply a matrix

	Perform matrix multiplication. The target must be MxK and the other must be KxN
	a new matrix of MxN is produced. This is fine for small matrices, if large matrices
	are to be multiplied, consider Mulp. If a scalar is passed in all elements in the
	array are multiplied by it.
	
	@see Mulp for a version which returns a promise
	@param the other matrix or a number
	@return a new matrix 
*/
void WrappedArray::Mul( const v8::FunctionCallbackInfo<v8::Value>& args )
{
/* The actual multiplication code is in MulpWorkAsync */
  WrappedArray::MulHelper( args, false, 1 ) ;
}




/**
	multiply a matrix in non-blocking mode

	Perform matrix multiplication. The target must be MxK and the other must be KxN
	a new matrix of MxN is produced. This will run in non-blocked
	mode. 

	There are two ways to use this, pass in an optional callback or accept a returned promise.
	Passing in a callback will prevent the Promise from being returned.
	
	@see Mul for a version which is blocking
	@param [in] the other matrix
	@param [in] a callback of prototype function(err,MATRIX){ }
	@return a promise which will resolve to a new Matrix

*/
void WrappedArray::Mulp( const v8::FunctionCallbackInfo<v8::Value>& args ) {
/* The actual multiplication code is in MulpWorkAsync */
  WrappedArray::MulHelper( args, true, 1 ) ;
}

/*
	This sets up the inputs and outputs (into a Work struct). That structure is
 	passed to the execution thread, which runs MulpWorkAsync.
*/
void WrappedArray::MulHelper( const v8::FunctionCallbackInfo<v8::Value>& args, bool block, int callbackIndex ) {
  Isolate* isolate = args.GetIsolate();
  Local<Context> context = isolate->GetCurrentContext() ;

  EscapableHandleScope scope(isolate) ;
  
// Get the 2 matrices to multiply
  WrappedArray *self = ObjectWrap::Unwrap<WrappedArray>(args.Holder());
// 2 choices - multiply by a scalar ( args[0] is a scalar)
// The matrix results may be different sizes depending on scalar or matrix multiply mode
  if( args[0]->IsNumber() ) { 
    const unsigned argc = 2;
    Local<Value> argv[argc] = { Integer::New( isolate,self->m_ ), Integer::New( isolate,self->n_ ) };
    Local<Function> cons = Local<Function>::New(isolate, constructor);
    Local<Object> instance = cons->NewInstance(context, argc, argv).ToLocalChecked() ;
    scope.Escape( instance ) ;
    WrappedArray::PrepareWork( args, block, callbackIndex, WrappedArray::MulpWorkAsync, instance ) ;
  } else {   // not a number other item is a matrix
    WrappedArray *other = ObjectWrap::Unwrap<WrappedArray>(args[0]->ToObject());    
    const unsigned argc = 2;
    Local<Value> argv[argc] = { Integer::New( isolate,self->m_ ), Integer::New( isolate,other->n_ ) };
    Local<Function> cons = Local<Function>::New(isolate, constructor);
    Local<Object> instance = cons->NewInstance(context, argc, argv).ToLocalChecked() ;
    scope.Escape( instance ) ;
    WrappedArray::PrepareWork( args, block, callbackIndex, WrappedArray::MulpWorkAsync, instance ) ;
  }
}


/*
	Handle the body of the multiply thread. Read the two matrices from
	the Work structure, do the multiply and return.
	This may, or may not, be called in a different thread that the caller's context
	Be aware of Local<...> and Persistent<...> data elements.
*/
void WrappedArray::MulpWorkAsync(uv_work_t *req) {
  Work *work = static_cast<Work *>(req->data);   

  WrappedArray* self = work->self ;	// setup in PrepareWork
  WrappedArray* result = work->result ;

  if( work->other == NULL ) {	// no other matrix? Then we'll use a scalar multiply
    float x = work->otherNumber ;
    int sz = self->m_ * self->n_ ;
    float *data = result->data_ ;
    float *a = self->data_ ;
    for( int i=0 ; i<sz ; i++ ) {
	*data++ = *a++ * x ;
    }	
  } else {
    WrappedArray* other = work->other ;
    if( self->n_ != other->m_ ) {
      work->err = new char[ 1000 ] ;	// set the error flag and abort
      snprintf( work->err, 1000, "Incompatible args: |%d x %d| x |%d x %d|", self->m_, self->n_, other->m_, other->n_ ) ;
    } else {
      cblas_sgemm(
          CblasColMajor,
          CblasNoTrans,
          CblasNoTrans,
          self->m_,
          other->n_,
          self->n_,
          1.f,
          self->data_,
          self->m_,
          other->data_,
          other->m_,
          0.f,
          result->data_,
          result->m_ );
    }
  }
}


/**
	sum all absolute value of each element
	@return the sum of all absolute values in the matrix
*/
void WrappedArray::Asum( const v8::FunctionCallbackInfo<v8::Value>& args )
{
  WrappedArray* self = ObjectWrap::Unwrap<WrappedArray>(args.Holder());
  float rc = cblas_sasum(
    self->m_ * self->n_ ,
    self->data_,
    1 ) ;
  args.GetReturnValue().Set( rc );
}

/**
	Sum the rows or columns of a matrix into a vector

	This sums the rows or columns of a matrix into a row or
	column vector. If the target is a vector this will just 
	sum all the elements into a single number.

	@param [in,default=0] the dimension to sum - 0 = sum columns, 1 = sum rows.
	@return the vector of the summed columns or rows or a number if the target is a vector.
*/
void WrappedArray::Sum( const v8::FunctionCallbackInfo<v8::Value>& args )
{
  Isolate* isolate = args.GetIsolate();
  Local<Context> context = isolate->GetCurrentContext() ;

  WrappedArray* self = ObjectWrap::Unwrap<WrappedArray>(args.Holder());

// If target is a vector ignore the dimensions
  if( self->isVector ) {
    float rc = 0 ;
    int l = self->m_ * self->n_ ;
    for( int i=0 ; i<l ; i++ ) {
      rc += self->data_[i] ;
    }
    args.GetReturnValue().Set( rc );
  }
  else {
    int dimension = args[0]->IsUndefined() ? 0 : args[0]->NumberValue() ;
    int m = ( dimension == 0 ) ? 1 : self->m_ ;
    int n = ( dimension == 0 ) ? self->n_ : 1 ;

    EscapableHandleScope scope(isolate) ; ;

    const unsigned argc = 2;
    Local<Value> argv[argc] = { Integer::New( isolate,m ), Integer::New( isolate,n ) };
    Local<Function> cons = Local<Function>::New(isolate, constructor);
    Local<Object> instance = cons->NewInstance(context, argc, argv).ToLocalChecked() ;

    scope.Escape(instance);
    WrappedArray* result = node::ObjectWrap::Unwrap<WrappedArray>( instance ) ;
    args.GetReturnValue().Set( instance );

    if( dimension == 0 ) {       // sum columns to 1xn vector
      int ix = 0 ;
      for( int c=0 ; c<self->n_ ; c++ ) {
        result->data_[c] = 0 ;
        for( int r=0 ; r<self->n_ ; r++ ) {
          result->data_[c] += self->data_[ix++] ;
        }
      }
    }

    if( dimension == 1 ) {       // sum rows to mx1 vector
      for( int r=0 ; r<self->m_ ; r++ ) {
        result->data_[r] = 0 ;
        int ix = r ;
        for( int c=0 ; c<self->n_ ; c++ ) {
          result->data_[r] += self->data_[ix] ;
          ix+=m ;
        }
      }
    }
  }
}


/**
	Euclidian norm of rows or columns of a matrix

	This calculates the norms of a row or column of a matrix into a row or
	column vector. If the target is a vector this will return the norm as a number.

	@param [in,default=0] the dimension to sum - 0 = sum columns, 1 = sum rows
	@return the vector of the norms of columns or rows or a number if the target is a vector.
*/
void WrappedArray::Norm( const v8::FunctionCallbackInfo<v8::Value>& args )
{
  Isolate* isolate = args.GetIsolate();
  Local<Context> context = isolate->GetCurrentContext() ;

  WrappedArray* self = ObjectWrap::Unwrap<WrappedArray>(args.Holder());

// If target is a vector ignore the dimensions
  if( self->isVector ) {
    float rc = 0 ;
    int l = self->m_ * self->n_ ;
    for( int i=0 ; i<l ; i++ ) {
      rc += self->data_[i] * self->data_[i] ;
    }
    args.GetReturnValue().Set( ::sqrt( rc ) );
  }
  else {
    int dimension = args[0]->IsUndefined() ? 0 : args[0]->NumberValue() ;
    int m = ( dimension == 0 ) ? 1 : self->m_ ;
    int n = ( dimension == 0 ) ? self->n_ : 1 ;

    EscapableHandleScope scope(isolate) ; ;

    const unsigned argc = 2;
    Local<Value> argv[argc] = { Integer::New( isolate,m ), Integer::New( isolate,n ) };
    Local<Function> cons = Local<Function>::New(isolate, constructor);
    Local<Object> instance = cons->NewInstance(context, argc, argv).ToLocalChecked() ;

    scope.Escape(instance);
    WrappedArray* result = node::ObjectWrap::Unwrap<WrappedArray>( instance ) ;
    args.GetReturnValue().Set( instance );

    if( dimension == 0 ) {       // sum columns to 1xn vector
      int ix = 0 ;
      for( int c=0 ; c<self->n_ ; c++ ) {
        result->data_[c] = 0 ;
        for( int r=0 ; r<self->n_ ; r++ ) {
          result->data_[c] += self->data_[ix] * self->data_[ix] ;
	  ix++ ;
        }
	result->data_[c] = ::sqrt( result->data_[c] ) ;
      }
    }

    if( dimension == 1 ) {       // sum rows to mx1 vector
      for( int r=0 ; r<self->m_ ; r++ ) {
        result->data_[r] = 0 ;
        int ix = r ;
        for( int c=0 ; c<self->n_ ; c++ ) {
          result->data_[r] += self->data_[ix] * self->data_[ix] ;
          ix+=m ;
        }
	result->data_[r] = ::sqrt( result->data_[r] ) ;
      }
    }
  }
}



/**
	mean of rows or columns of a matrix 

	This calculates the mean of rows or columns of a matrix into a row or
	column vector. If the target is a vector this will return the mean of
	all the elements into a single number.

	@param [in,default=0] the dimension to inspect - 0 = mean of columns, 1 = mean of rows.
	@return the vector of the mean of columns or rows. In the case that the target is a vector, this is a number
*/
void WrappedArray::Mean( const v8::FunctionCallbackInfo<v8::Value>& args )
{
  Isolate* isolate = args.GetIsolate();
  Local<Context> context = isolate->GetCurrentContext() ;

  WrappedArray* self = ObjectWrap::Unwrap<WrappedArray>(args.Holder());

  if( self->isVector ) {
    float rc = 0 ;
    int l = self->m_ * self->n_ ;
    for( int i=0 ; i<l ; i++ ) {
      rc += self->data_[i] ;
    }
    rc /= l ;
    args.GetReturnValue().Set( rc );
  }
  else {
    int dimension = args[0]->IsUndefined() ? 0 : args[0]->NumberValue() ;
    int m = ( dimension == 0 ) ? 1 : self->m_ ;
    int n = ( dimension == 0 ) ? self->n_ : 1 ;

    EscapableHandleScope scope(isolate) ; ;

    const unsigned argc = 2;
    Local<Value> argv[argc] = { Integer::New( isolate,m ), Integer::New( isolate,n ) };
    Local<Function> cons = Local<Function>::New(isolate, constructor);
    Local<Object> instance = cons->NewInstance(context, argc, argv).ToLocalChecked() ;

    scope.Escape(instance);
    WrappedArray* result = node::ObjectWrap::Unwrap<WrappedArray>( instance ) ;
    args.GetReturnValue().Set( instance );

    if( dimension == 0 ) {       // sum columns to 1xn vector
      int ix = 0 ;
      for( int c=0 ; c<self->n_ ; c++ ) {
        result->data_[c] = 0 ;
        for( int r=0 ; r<self->m_ ; r++ ) {
          result->data_[c] += self->data_[ix++] ;
        }
        result->data_[c] /= self->m_ ;
      }
    }

    if( dimension == 1 ) {       // sum rows to mx1 vector
      for( int r=0 ; r<self->m_ ; r++ ) {
        result->data_[r] = 0 ;
        int ix = r ;
        for( int c=0 ; c<self->n_ ; c++ ) {
          result->data_[r] += self->data_[ix] ;
          ix+=m ;
        }
        result->data_[r] /= self->n_ ;
      }
    }
  }
}







/**
	Copy rows from the target

	Build a new matrix containing some of the rows or the target. The target is
	unchanged. The input can be a single number or an array of rows to copy
	from the target. The reulting array will be KxN, where k is the number of
	rows requested and N is the number of columns in the target.

	\code{.js}

	var R = MATRIX.getRows( [ 1,2,2] ) ;
	
	\endcode

	@param [in,default=0] the rows to copy from the matrix, may be a number or an array of numbers
	@return a new matrix containing the copies of the requested rows.
*/
void WrappedArray::GetRows( const v8::FunctionCallbackInfo<v8::Value>& args )
{
  Isolate* isolate = args.GetIsolate();
  Local<Context> context = isolate->GetCurrentContext() ;

  WrappedArray* self = ObjectWrap::Unwrap<WrappedArray>(args.Holder());

  int numRows = 1 ;  
  int *m = new int[0] ;

  if( !args[0]->IsUndefined() && args[0]->IsArray() ) {
    Local<Array> rows = Local<Array>::Cast(args[0]->ToObject() );
    numRows = rows->Length() ;
    m = new int[ numRows ] ;
    for( int i=0 ; i<numRows ; i++ ) {
	m[i] = rows->Get( context, i ).ToLocalChecked()->NumberValue() ;
    }
  } else {
     m = new int[1] ;
     m[0] = args[0]->IsUndefined() ? 0 : args[0]->NumberValue() ;
  }

  EscapableHandleScope scope(isolate) ; ;

  // make a row vector: 1xn 
  const unsigned argc = 2;
  Local<Value> argv[argc] = { Integer::New( isolate,numRows ), Integer::New( isolate,self->n_ ) };
  Local<Function> cons = Local<Function>::New(isolate, constructor);
  Local<Object> instance = cons->NewInstance(context, argc, argv).ToLocalChecked() ;

  scope.Escape(instance);
  WrappedArray* result = node::ObjectWrap::Unwrap<WrappedArray>( instance ) ;
  args.GetReturnValue().Set( instance );

  int offsetResult = 0 ;
  int offsetSelf = 0 ;
  for( int c=0 ; c<self->n_ ; c++ ) {
    for( int r=0 ; r<numRows ; r++ ) {
      int ixa = m[r]+offsetSelf ;
      int ixb = r+offsetResult ; 
      result->data_[ixb] = self->data_[ixa] ;
    }
    offsetSelf += self->m_ ;
    offsetResult += result->m_ ;
  }

  delete m ;
}



/**
	Remove a row from a matrix

	Return a vector which is the requested row of the target. The target is 
	shrunk to have M-1 rows. This is not very efficient.

	\code{.js}

	var R = MATRIX.removeRow(4) ;
	
	\endcode

	@param [in] the row index (zero based) to remove from the array default=0
	@return a row vector containing the extracted row.
*/
void WrappedArray::RemoveRow( const v8::FunctionCallbackInfo<v8::Value>& args )
{
  Isolate* isolate = args.GetIsolate();
  Local<Context> context = isolate->GetCurrentContext() ;

  WrappedArray* self = ObjectWrap::Unwrap<WrappedArray>(args.Holder());

  int m = args[0]->IsUndefined() ? 0 : args[0]->NumberValue() ;

  EscapableHandleScope scope(isolate) ; ;

  // make a row vector: 1xn 
  const unsigned argc = 2;
  Local<Value> argv[argc] = { Integer::New( isolate,1 ), Integer::New( isolate,self->n_ ) };
  Local<Function> cons = Local<Function>::New(isolate, constructor);
  Local<Object> instance = cons->NewInstance(context, argc, argv).ToLocalChecked() ;

  scope.Escape(instance);
  WrappedArray* result = node::ObjectWrap::Unwrap<WrappedArray>( instance ) ;
  args.GetReturnValue().Set( instance );

  int ix = m ;
  int sz = self->m_ * self->n_ ;
  for( int c=0 ; c<self->n_ ; c++ ) {
    result->data_[c] = self->data_[ix] ;
    memmove( self->data_+ix, self->data_+ix+1, (sz-ix-1)*sizeof(float) ) ;
    ix += self->m_ - 1 ;
  }
  self->m_-- ;
}


/**
	Copy columns from the target

	Build a new matrix containing some of the columns of the target. The target is
	unchanged. The input can be a single number or an array of column indices to copy
	from the target. The reulting array will be MxK, where k is the number of
	columns requested and M is the number of rows in the target.

	\code{.js}

	var R = MATRIX.getColumns( [0,2] ) ;
	
	\endcode

	@param [in,default=0] the column indices to copy from the array , may be a number or an array of numbers
	@return a new matrix containing the copies of the requested columns.
*/
void WrappedArray::GetColumns( const v8::FunctionCallbackInfo<v8::Value>& args )
{
  Isolate* isolate = args.GetIsolate();
  Local<Context> context = isolate->GetCurrentContext() ;

  WrappedArray* self = ObjectWrap::Unwrap<WrappedArray>(args.Holder());

  int numCols = 1 ;  
  int *n = new int[0] ;

  if( !args[0]->IsUndefined() && args[0]->IsArray() ) {
    Local<Array> cols = Local<Array>::Cast(args[0]->ToObject() );
    numCols = cols->Length() ;
    n = new int[ numCols ] ;
    for( int i=0 ; i<numCols ; i++ ) {
	n[i] = cols->Get( context, i ).ToLocalChecked()->NumberValue() ;
    }
  } else {
     n = new int[1] ;
     n[0] = args[0]->IsUndefined() ? 0 : args[0]->NumberValue() ;
  }

  EscapableHandleScope scope(isolate) ; ;

  // make a row vector: mxnumCols
  const unsigned argc = 2;
  Local<Value> argv[argc] = { Integer::New( isolate,self->m_ ), Integer::New( isolate,numCols ) };
  Local<Function> cons = Local<Function>::New(isolate, constructor);
  Local<Object> instance = cons->NewInstance(context, argc, argv).ToLocalChecked() ;

  scope.Escape(instance);
  WrappedArray* result = node::ObjectWrap::Unwrap<WrappedArray>( instance ) ;
  args.GetReturnValue().Set( instance );


  for( int c=0 ; c<numCols ; c++ ) {
    int ixa = n[c] * self->m_ ;
    int ixb = c * self->m_ ;
    for( int r=0 ; r<self->m_ ; r++ ) {
      result->data_[ixb+r] = self->data_[ixa+r] ;
    }
  }
  delete n ;
}

/**
	Remove a column from a matrix

	Return a column vector which is the requested column of the target. The target is 
	shrunk to have N-1 columns.

	\code{.js}

	var R = MATRIX.removeColumn(4) ;
	
	\endcode

	@param [in,default=0] the column index (zero based) to remove from the array
	@return a column vector containing the extracted column.
*/
void WrappedArray::RemoveColumn( const v8::FunctionCallbackInfo<v8::Value>& args )
{
  Isolate* isolate = args.GetIsolate();
  Local<Context> context = isolate->GetCurrentContext() ;

  WrappedArray* self = ObjectWrap::Unwrap<WrappedArray>(args.Holder());

  int n = args[0]->IsUndefined() ? 0 : args[0]->NumberValue() ;

  EscapableHandleScope scope(isolate) ; ;

  // make a row vector: mx1
  const unsigned argc = 2;
  Local<Value> argv[argc] = { Integer::New( isolate,self->m_ ), Integer::New( isolate,1 ) };
  Local<Function> cons = Local<Function>::New(isolate, constructor);
  Local<Object> instance = cons->NewInstance(context, argc, argv).ToLocalChecked() ;

  scope.Escape(instance);
  WrappedArray* result = node::ObjectWrap::Unwrap<WrappedArray>( instance ) ;
  args.GetReturnValue().Set( instance );

  int ix = n * self->m_ ;
  int colsToMove = self->n_ - n - 1 ;
  float *start = self->data_ + ix ;
  memcpy( result->data_, start, self->m_*sizeof(float) ) ;
  memmove( start, start+self->m_, colsToMove*self->m_*sizeof(float) ) ;
  self->n_-- ;
}


/**
	Append columns to a matrix

	Append column vectors. The target is expanded to have N+K columns. The matrix should
	have the same length as the matrix.m ( rows ). K is the width of the added column matrix

	\code{.js}

        var MATRIX = lalg.rand(10,6) ;
        var V = lalg.rand(3,10) ;
	var R = MATRIX.appendColumns(V) ;
	
	\endcode

	@param [in] the column(s) to append to the matrix
	@return a new matrix containing the additional column.
*/
void WrappedArray::AppendColumns( const v8::FunctionCallbackInfo<v8::Value>& args )
{
  Isolate* isolate = args.GetIsolate();
  Local<Context> context = isolate->GetCurrentContext() ;

  WrappedArray* self = ObjectWrap::Unwrap<WrappedArray>(args.Holder());
  WrappedArray *other = ObjectWrap::Unwrap<WrappedArray>(args[0]->ToObject());    

  EscapableHandleScope scope(isolate) ; ;

// Matching sizes? same #rows - or any vector of length M
  if( other->m_!= self->m_ ) {
      char *msg = new char[ 1000 ] ;
      snprintf( msg, 1000, "Incompatible args: |%d x %d| append |%d x %d|", self->m_, self->n_, other->m_, other->n_ ) ;
      isolate->ThrowException(Exception::TypeError( String::NewFromUtf8(isolate, msg)));
      delete msg ;
      args.GetReturnValue().Set( Undefined(isolate) );
  }

  // make a new matrix M x N+K  ( K = other cols )
  const unsigned argc = 2;
  Local<Value> argv[argc] = { Integer::New( isolate,self->m_ ), Integer::New( isolate,self->n_ + other->n_ ) };
  Local<Function> cons = Local<Function>::New(isolate, constructor) ;
  Local<Object> instance = cons->NewInstance(context, argc, argv).ToLocalChecked() ;

  scope.Escape(instance);
  WrappedArray* result = node::ObjectWrap::Unwrap<WrappedArray>( instance ) ;
  args.GetReturnValue().Set( instance );

  // copy the original data 
  memcpy( result->data_, self->data_, self->n_ * self->m_*sizeof(float) ) ;
  // then append the vectors !
  memcpy( result->data_+(self->n_ * self->m_), other->data_, other->n_*other->m_*sizeof(float) ) ;
}




/**
	Rotate columns in a matrix

	Rotate column vectors. A positive number means rotate to the left, a negative number rotates
	to the right. The rotation count is internally and silently limited using modulus the 
	number of columns. 

	If there is a requirement to rotate rows, transpose the matrix then call this, the transpose back.

	\code{.js}

	B = A.rotateColumns( 1 ) ;
	C = A.rotateColumns( -2 ) ;
	
	\endcode

            | 1.00 2.00 3.00 4.00 5.00 |
        A = | 1.00 2.00 3.00 4.00 5.00 |
            | 1.00 2.00 3.00 4.00 5.00 |


            | 2.00 3.00 4.00 5.00 1.00 |
        B = | 2.00 3.00 4.00 5.00 1.00 |
            | 2.00 3.00 4.00 5.00 1.00 |


            | 4.00 5.00 1.00 2.00 3.00 |
        C = | 4.00 5.00 1.00 2.00 3.00 |
            | 4.00 5.00 1.00 2.00 3.00 |


	@param [in,default=1] the rotation count.
	@return a new matrix containing the rotated columns.
*/
void WrappedArray::RotateColumns( const v8::FunctionCallbackInfo<v8::Value>& args )
{
  Isolate* isolate = args.GetIsolate();
  Local<Context> context = isolate->GetCurrentContext() ;

  WrappedArray* self = ObjectWrap::Unwrap<WrappedArray>(args.Holder());
  int rotationCount = args[0]->IsUndefined() ? 1 : args[0]->NumberValue() ;

  EscapableHandleScope scope(isolate) ; ;

  // make a new matrix M x N 
  const unsigned argc = 2;
  Local<Value> argv[argc] = { Integer::New( isolate,self->m_ ), Integer::New( isolate,self->n_ ) };
  Local<Function> cons = Local<Function>::New(isolate, constructor) ;
  Local<Object> instance = cons->NewInstance(context, argc, argv).ToLocalChecked() ;

  scope.Escape(instance);
  WrappedArray* result = node::ObjectWrap::Unwrap<WrappedArray>( instance ) ;
  args.GetReturnValue().Set( instance );

  if( rotationCount < 0 ) {   
    rotationCount = self->n_ + rotationCount ;
  }
  rotationCount %= self->n_ ;

  // copy part 1 of the data
  memcpy( result->data_, self->data_+(self->m_*rotationCount), (self->n_-rotationCount) * self->m_*sizeof(float) ) ;
  // then append the vectors !
  memcpy( result->data_+(self->m_*(self->n_-rotationCount)), self->data_, rotationCount * self->m_*sizeof(float) ) ;
}






/**
	Reshape the matrix

	Change the shape of the matrix. If the shape is smaller ( new M x new N < M x N )
	the memory is left alone. Otherwise a new buffer is allocated, and the additional
	elements are initialized to zero. This operation is done in place - the target is
	changed. The default is to produce a (M*N)x1 vector is no paramters are given.

	@param [in,default=m*n] the number of rows to have in the new shape
	@param [in,default=1] the number of columns to have in the new shape
*/
void WrappedArray::Reshape( const v8::FunctionCallbackInfo<v8::Value>& args )
{
//  Isolate* isolate = args.GetIsolate();
//  Local<Context> context = isolate->GetCurrentContext() ;

  WrappedArray* self = ObjectWrap::Unwrap<WrappedArray>(args.Holder());

  int m = args[0]->IsUndefined() ? (self->m_*self->n_) : args[0]->NumberValue() ;
  int n = args[1]->IsUndefined() ? 1 : args[1]->NumberValue() ;

//  EscapableHandleScope scope(isolate) ; ;

  args.GetReturnValue().Set( args.Holder() );

// Only if we need to ... allocate new memory
  if( m*n > self->dataSize_ ) {
    float *tmp = self->data_ ;
    self->dataSize_ = m*n;
    self->data_ = new float[self->dataSize_];
    memcpy( self->data_, tmp, self->m_*self->n_*sizeof(float) ) ; 
    delete tmp  ;
  }
  self->m_ = m ;
  self->n_ = n ;
}

/**
	Add two matrices

	Performs one of
	- Element wise addition of two matrices. The other array to be added
	must be the same shape or vector matching shape
	- Row or column wise addition of a vector to the target  ( if adding a row vector to 
	a target the number of elements in the vector must match the number of columns
	in the matrix).
  	- Simple element wise addition - adds a single number to every element in the target

	@param [in] one of 
	- a matrix 
	- a vector 
	- a number
	@return a new matrix

*/
void WrappedArray::Add( const v8::FunctionCallbackInfo<v8::Value>& args )
{
  Isolate* isolate = args.GetIsolate();
  Local<Context> context = isolate->GetCurrentContext() ;
  EscapableHandleScope scope(isolate) ;

  WrappedArray* self = ObjectWrap::Unwrap<WrappedArray>(args.Holder());
  const unsigned argc = 2;
  Local<Value> argv[argc] = { Integer::New( isolate,self->m_ ), Integer::New( isolate,self->n_ ) };
  Local<Function> cons = Local<Function>::New(isolate, constructor);
  Local<Object> instance = cons->NewInstance(context, argc, argv).ToLocalChecked() ;
  scope.Escape(instance);
  WrappedArray* result = node::ObjectWrap::Unwrap<WrappedArray>( instance ) ;
  args.GetReturnValue().Set( instance );

  float *data = result->data_ ;
  float *a = self->data_ ;
  int sz = self->m_ * self->n_ ;

  if( args[0]->IsNumber() ) { 
	float x = args[0]->NumberValue() ;
	for( int i=0 ; i<sz ; i++ ) {
		*data++ = *a++ + x ;
	}		
  } else {
    WrappedArray* other = ObjectWrap::Unwrap<WrappedArray>( args[0]->ToObject() );
    float *b = other->data_ ;
    if( self->n_ == other->n_  &&  self->m_ == other->m_ ) {
      for( int i=0 ; i<sz ; i++ ) {
	*data++ = *a++ + *b++ ;
      }
    } else if( self->n_ == other->n_  &&  other->m_ == 1 ) { // add a row vector to each row
      int j = 0 ;
      int n = self->m_ ;
      for( int i=0 ; i<sz ; i++ ) {
        *data++ = *a++ + b[j] ;
        if( --n == 0 ) { j++ ; n = self->m_ ; }
      }
    } else if( self->m_ == other->m_  &&  other->n_ == 1 ) { // add a col vector to each col
      int j = 0 ;
      for( int i=0 ; i<sz ; i++ ) {
        data[i] = a[i] + b[j++] ;
        if( j>=self->m_ ) j=0 ;
      }
    } else { // incompatible types ...
      char *msg = new char[ 1000 ] ;
      snprintf( msg, 1000, "Incompatible args: |%d x %d| + |%d x %d|", self->m_, self->n_, other->m_, other->n_ ) ;
      isolate->ThrowException(Exception::TypeError( String::NewFromUtf8(isolate, msg)));
      delete msg ;
      args.GetReturnValue().Set( Undefined(isolate) );
    }
  }
}



/**
	Subtract two matrices

	Performs one of
	- Element wise subtraction of two matrices. The other array to be subtracted from target
	must be the same shape or vector matching shape
	- Row or column wise subtraction of a vector from the target  ( if subtracting a row vector from
	a target the number of elements in the vector must match the number of columns
	in the matrix).
  	- Simple element wise subtraction - subtracts a single number from every element in the target

	@param [in] one of 
	- a matrix 
	- a vector 
	- a number
	@return a new matrix

*/
void WrappedArray::Sub( const v8::FunctionCallbackInfo<v8::Value>& args )
{
  WrappedArray* self = ObjectWrap::Unwrap<WrappedArray>(args.Holder());

  Isolate* isolate = args.GetIsolate();
  Local<Context> context = isolate->GetCurrentContext() ;
  EscapableHandleScope scope(isolate) ;

  const unsigned argc = 2;
  Local<Value> argv[argc] = { Integer::New( isolate,self->m_ ), Integer::New( isolate,self->n_ ) };
  Local<Function> cons = Local<Function>::New(isolate, constructor);
  Local<Object> instance = cons->NewInstance(context, argc, argv).ToLocalChecked() ;
  scope.Escape(instance);
  args.GetReturnValue().Set( instance );

  WrappedArray* result = node::ObjectWrap::Unwrap<WrappedArray>( instance ) ;

  float *data = result->data_ ;
  float *a = self->data_ ;
  int sz = self->m_ * self->n_ ;

  if( args[0]->IsNumber() ) { 
	float x = args[0]->NumberValue() ;
	for( int i=0 ; i<sz ; i++ ) {
		*data++ = *a++ - x ;
	}		
  } else {
    WrappedArray* other = ObjectWrap::Unwrap<WrappedArray>( args[0]->ToObject() );
    float *b = other->data_ ;
    if( self->n_ == other->n_  &&  self->m_ == other->m_ ) {
      for( int i=0 ; i<sz ; i++ ) {
	*data++ = *a++ - *b++ ;
      }
    } else if( self->n_ == other->n_  &&  other->m_ == 1 ) { // add a row vector to each row
      int j = 0 ;
      int n = self->m_ ;
      for( int i=0 ; i<sz ; i++ ) {
        *data++ = *a++ - b[j] ;
        if( --n == 0 ) { j++ ; n = self->m_ ; }
      }
    } else if( self->m_ == other->m_  &&  other->n_ == 1 ) { // add a col vector to each col
      int j = 0 ;
      for( int i=0 ; i<sz ; i++ ) {
        data[i] = a[i] - b[j++] ;
        if( j>=self->m_ ) j=0 ;
      }
    } else { // incompatible types ...
      char *msg = new char[ 1000 ] ;
      snprintf( msg, 1000, "Incompatible args: |%d x %d| + |%d x %d|", self->m_, self->n_, other->m_, other->n_ ) ;
      isolate->ThrowException(Exception::TypeError( String::NewFromUtf8(isolate, msg)));
      delete msg ;
      args.GetReturnValue().Set( Undefined(isolate) );
    }
  }
}


/**
	Hadamard (Schur) multiply of two matrices

	This is (VERY) different to normal matrix multipliy.  Also the vector x matrix version just multiplies
	each column (or row) entry by the corresponding vector element.
	
	It is the .* operator in Matlab or Octave

	Performs one of
	- Element wise multiplication of two matrices. The other array to be added
	must be the same shape or vector matching shape
	- Row or column wise multiplication of a vector to the target  ( if adding a row vector to 
	a target the number of elements in the vector must match the number of columns
	in the matrix).
  	- Simple element wise multiplication - multiplication a single number by every element in the target

	@param [in] one of 
	- a matrix 
	- a vector 
	- a number
	@return a new matrix

*/
void WrappedArray::Hadamard( const v8::FunctionCallbackInfo<v8::Value>& args )
{
  Isolate* isolate = args.GetIsolate();
  Local<Context> context = isolate->GetCurrentContext() ;
  EscapableHandleScope scope(isolate) ;

  WrappedArray* self = ObjectWrap::Unwrap<WrappedArray>(args.Holder());
  const unsigned argc = 2;
  Local<Value> argv[argc] = { Integer::New( isolate,self->m_ ), Integer::New( isolate,self->n_ ) };
  Local<Function> cons = Local<Function>::New(isolate, constructor);
  Local<Object> instance = cons->NewInstance(context, argc, argv).ToLocalChecked() ;
  scope.Escape(instance);
  WrappedArray* result = node::ObjectWrap::Unwrap<WrappedArray>( instance ) ;
  args.GetReturnValue().Set( instance );

  float *data = result->data_ ;
  float *a = self->data_ ;
  int sz = self->m_ * self->n_ ;

  if( args[0]->IsNumber() ) { 
	float x = args[0]->NumberValue() ;
	for( int i=0 ; i<sz ; i++ ) {
		*data++ = *a++ * x ;
	}		
  } else {
    WrappedArray* other = ObjectWrap::Unwrap<WrappedArray>( args[0]->ToObject() );
    float *b = other->data_ ;
    if( self->n_ == other->n_  &&  self->m_ == other->m_ ) {
      for( int i=0 ; i<sz ; i++ ) {
	*data++ = *a++ * *b++ ;
      }
    } else if( self->n_ == other->n_  &&  other->m_ == 1 ) { // add a row vector to each row
      int j = 0 ;
      int n = self->m_ ;
      for( int i=0 ; i<sz ; i++ ) {
        *data++ = *a++ * b[j] ;
        if( --n == 0 ) { j++ ; n = self->m_ ; }
      }
    } else if( self->m_ == other->m_  &&  other->n_ == 1 ) { // add a col vector to each col
      int j = 0 ;
      for( int i=0 ; i<sz ; i++ ) {
        data[i] = a[i] * b[j++] ;
        if( j>=self->m_ ) j=0 ;
      }
    } else { // incompatible types ...
      char *msg = new char[ 1000 ] ;
      snprintf( msg, 1000, "Incompatible args: |%d x %d| + |%d x %d|", self->m_, self->n_, other->m_, other->n_ ) ;
      isolate->ThrowException(Exception::TypeError( String::NewFromUtf8(isolate, msg)));
      delete msg ;
      args.GetReturnValue().Set( Undefined(isolate) );
    }
  }
}




/**
	Matrix inverse

	Calculate the inverse of a matrix. A matrix inverse multiplied by itself
	is an identity matrix. The input matrix must be square. A newly created 
	inverse is returned, the original remains intact.

	@return the new matrix inverse of the target
*/
void WrappedArray::Inv( const v8::FunctionCallbackInfo<v8::Value>& args )
{
/* Actual work done in InvpWorkAsync */
  WrappedArray::InvHelper( args, false, 0 ) ;
}

/**
	Matrix inverse

	Calculate the inverse of a matrix. A matrix inverse multiplied by itself
	is an identity matrix. The input matrix must be square. The return is a
	promise ( no args ) or, if a callback function is provided undefined.
	A newly created inverse is resolved into the promise or the callback;
	the original remains intact.

	@param [in,optional] a callback function prototype = function(err,inv) { }
	@return a promise (if the callback function is not given)
*/
void WrappedArray::Invp( const v8::FunctionCallbackInfo<v8::Value>& args )
{
/* Actual work done in InvpWorkAsync */
  WrappedArray::InvHelper( args, true, 0 ) ;
}



void WrappedArray::InvHelper( const v8::FunctionCallbackInfo<v8::Value>& args, bool block, int callbackIndex ) {
  Isolate* isolate = args.GetIsolate();
  Local<Context> context = isolate->GetCurrentContext() ;

  EscapableHandleScope scope(isolate) ;
  
// Get the matrix to invert
  WrappedArray *self = ObjectWrap::Unwrap<WrappedArray>(args.Holder());

  const unsigned argc = 2;
  Local<Value> argv[argc] = { Integer::New( isolate,self->m_ ), Integer::New( isolate,self->n_ ) };
  Local<Function> cons = Local<Function>::New(isolate, constructor);
  Local<Object> instance = cons->NewInstance(context, argc, argv).ToLocalChecked() ;
  scope.Escape( instance ) ;
  WrappedArray::PrepareWork( args, block, callbackIndex, WrappedArray::InvpWorkAsync, instance ) ;
}




void WrappedArray::InvpWorkAsync( uv_work_t *req )
{
  Work *work = static_cast<Work *>(req->data);

  WrappedArray* self = work->self ;

  if( self->m_ != self->n_ ) {
    work->err = new char[ 1000 ] ;
    snprintf( work->err, 1000, "Incompatible args: |%d x %d| should be a square matrix for inv()", self->m_, self->n_ ) ;
  } else {
    WrappedArray* result = work->result ;

    int sz = result->m_ * result->n_ ;
    for( int i=0 ; i<sz ; i++ ) {
      result->data_[i] = self->data_[i] ;
    }

    int *ipiv = new int[ std::min( self->m_, self->n_) ] ;
    int rc = LAPACKE_sgetrf(
        CblasColMajor,
        result->m_,
        result->n_,
        result->data_,
        result->m_,
        ipiv ) ;
    if( rc != 0 ) {
      work->err = new char[ 1000 ] ;
      if( rc>0 ) {
        snprintf( work->err, 1000, "This matrix is singular and cannot be inverted" ) ;
      } else {
        snprintf( work->err, 1000, "Internal failure - sgetrf() failed with %d", rc ) ;
      }
    } else {
      rc = LAPACKE_sgetri(
          CblasColMajor,
          result->n_,
          result->data_,
          result->n_,
          ipiv ) ;
      if( rc != 0 ) {
        work->err = new char[ 1000 ] ;
        snprintf( work->err, 1000, "Internal failure - sgetri() failed with %d", rc ) ;
      }
    }
    delete ipiv ;
  }
}






/**
	Get the pseudo inverse of a matrix

	Calculate the pseudo inverse of a matrix. A matrix multiplied by its 
	pseudo inverse is an identity matrix. A newly created inverse is 
	returned, the original remains intact.

        This seems to have a problem with vectors - don't invert a vector
        until we know what it may mean.

	@return the new matrix pseudo inverse of the target
*/
void WrappedArray::Pinv( const v8::FunctionCallbackInfo<v8::Value>& args )
{
  Isolate* isolate = args.GetIsolate();
  Local<Context> context = isolate->GetCurrentContext() ;

  WrappedArray* self = ObjectWrap::Unwrap<WrappedArray>(args.Holder());

    EscapableHandleScope scope(isolate) ;

    const unsigned argc = 2;
// inverse matrix is NxM
    Local<Value> argv[argc] = { Integer::New( isolate,self->n_ ), Integer::New( isolate,self->m_ ) };
    Local<Function> cons = Local<Function>::New(isolate, constructor);
    Local<Object> instance = cons->NewInstance(context, argc, argv).ToLocalChecked() ;

    scope.Escape( instance );

    args.GetReturnValue().Set( instance );
    WrappedArray* result = node::ObjectWrap::Unwrap<WrappedArray>( instance ) ;

    int n = self->n_ ;
    int m = self->m_ ;

/*******************************
*  T A L L  inv(A' x A) x A'   *
*******************************/
if( m > n ) {	
// multiply A'A to get NxN covariance
  float *cov = new float[ n * n ] ;
	
  cblas_sgemm(
      CblasColMajor,	// always
      CblasTrans,	// A x A'
      CblasNoTrans,     
      n,		// A' rows ( N )
      n,		// B cols ( N )
      n,		// A cols		
      1.f,		// mpy by 1.0 
      self->data_,	// A data
      m,		// A rows  
      self->data_,	// B data
      m,		// B rows
      0.f,		// don't do anything with input cov
      cov,		// output
      n );		// output rows

// then invert cov
    int *ipiv = new int[ n ] ;  // cov matrix is n x n
    int rc = LAPACKE_sgetrf(
      CblasColMajor,
      n,
      n,
      cov,
      n,
      ipiv ) ;
    if( rc != 0 ) {
      char *msg = new char[ 1000 ] ;
      if( rc>0 ) {
        snprintf( msg, 1000, "This matrix is singular and cannot be inverted" ) ;
      } else {
        snprintf( msg, 1000, "Internal failure - sgetrf() failed with %d", rc ) ;
      }
      isolate->ThrowException(Exception::TypeError( String::NewFromUtf8(isolate, msg)));
      delete msg  ;
      args.GetReturnValue().Set( Undefined(isolate) );
    }
    else {
      rc = LAPACKE_sgetri(
        CblasColMajor,
        n,
        cov,
        n,
        ipiv ) ;
      if( rc != 0 ) {
        char *msg = new char[ 1000 ] ;
        snprintf( msg, 1000, "Internal failure - sgetri() failed with %d", rc ) ;
        isolate->ThrowException(Exception::TypeError( String::NewFromUtf8(isolate, msg)));
        delete msg ;
        args.GetReturnValue().Set( Undefined(isolate) );
      }
    }
    delete ipiv ;

// finally multiply the above inverse by A'
   cblas_sgemm(
      CblasColMajor,
      CblasNoTrans,	// COV x A'
      CblasTrans,
      n,		// COV rows
      n,		// A' cols = n
      n,		// COV cols
      1.f,		// no scaling of COV
      cov,		// COV
      n,		// COV rows
      self->data_,	// A
      m,		// A rows
      0.f,		// leave result alone
      result->data_,	// result
      m );	  	// rows in result
   delete cov ;

  } else {  	
/*******************************
*  S H O R T  A' x inv(A x A') *
*******************************/

// multiply AA' to get MxM covariance
  float *cov = new float[ m * m ] ;
	
  cblas_sgemm(
      CblasColMajor,	// always
      CblasNoTrans,	// A x A'
      CblasTrans,     
      m,		// A rows ( m )
      m,		// B cols ( A' cols = m )
      n,		// A cols ( n ) 
      1.f,		// mpy by 1.0 
      self->data_,	// A data
      m,		// A rows  
      self->data_,	// B data
      m,		// B rows
      0.f,		// don't do anything with input cov
      cov,		// output
      m );		// output rows

// then invert cov

    int *ipiv = new int[ n ] ;  // cov matrix is m x m
    int rc = LAPACKE_sgetrf(
      CblasColMajor,
      m,
      m,
      cov,
      m,
      ipiv ) ;
    if( rc != 0 ) {
      char *msg = new char[ 1000 ] ;
      if( rc>0 ) {
        snprintf( msg, 1000, "This matrix is singular and cannot be inverted" ) ;
      } else {
        snprintf( msg, 1000, "Internal failure - sgetrf() failed with %d", rc ) ;
      }
      isolate->ThrowException(Exception::TypeError( String::NewFromUtf8(isolate, msg)));
      delete msg  ;
      args.GetReturnValue().Set( Undefined(isolate) );
    } else {
      rc = LAPACKE_sgetri(
        CblasColMajor,
        m,
        cov,
        m,
        ipiv ) ;
      if( rc != 0 ) {
        char *msg = new char[ 1000 ] ;
        snprintf( msg, 1000, "Internal failure - sgetri() failed with %d", rc ) ;
        isolate->ThrowException(Exception::TypeError( String::NewFromUtf8(isolate, msg)));
        delete msg ;
        args.GetReturnValue().Set( Undefined(isolate) );
      }
    }
    delete ipiv ;

// finally multiply A' by the above inverse
   cblas_sgemm(
      CblasColMajor,
      CblasTrans,	// A' x COV 
      CblasNoTrans,
      n,		// A' rows
      m,		// A' cols 
      m,		// COV cols
      1.f,		// no scaling of COV
      self->data_,	// A'
      m,		// A rows
      cov,		// COV
      m,		// COV rows
      0.f,		// leave result alone
      result->data_,	// result
      n );		// rows in result

   delete cov ;
  }
}




/**
	Singular Value Decomposition of a matrix

	Calculate the SVD of a matrix. This will return an object
	with 3 attributes: U, S and VT. This factorizes the matrix to
	3 components.
	
	\code{.js}
	
	var lalg = require('lalg');

	var A = lalg.rand(10) ;	// a 10x10 random array

	var svd = A.svd()  ;
	var B = U x lalg.diag(S) x VT ;

	// B and A should be the same ! (math precision permitting)

	\endcode

	@return a JS object with 3 components
	- U the left singular vectors	
	- S a vector of the eigenvalues ( use Diag to conver to a matrix )
	- VT the transposed right singular vectors
	
	@see Diag
*/
void WrappedArray::Svd( const v8::FunctionCallbackInfo<v8::Value>& args )
{
  Isolate* isolate = args.GetIsolate();
  Local<Context> context = isolate->GetCurrentContext() ;

  WrappedArray* self = ObjectWrap::Unwrap<WrappedArray>(args.Holder());

  EscapableHandleScope scope(isolate) ;

  Local<Function> cons = Local<Function>::New(isolate, constructor);

  const unsigned argc = 2;
  Local<Value> argvu[argc] = { Integer::New( isolate,self->m_ ), Integer::New( isolate,self->m_ ) };
  Local<Object> U = cons->NewInstance(context, argc, argvu).ToLocalChecked() ;

  Local<Value> argvvt[argc] = { Integer::New( isolate,self->n_ ), Integer::New( isolate,self->n_ ) };
  Local<Object> VT = cons->NewInstance(context, argc, argvvt).ToLocalChecked() ;

  Local<Value> argvs[argc] = { Integer::New( isolate,self->n_ ), Integer::New( isolate,1 ) };
  Local<Object> S = cons->NewInstance(context, argc, argvs).ToLocalChecked() ;

  Local<Object> result = Object::New(isolate);
  result->Set(String::NewFromUtf8(isolate, "U"), U );
  result->Set(String::NewFromUtf8(isolate, "S"), S );
  result->Set(String::NewFromUtf8(isolate, "VT"), VT );
  scope.Escape( result ) ;

  WrappedArray* u = ObjectWrap::Unwrap<WrappedArray>(U);
  WrappedArray* vt = ObjectWrap::Unwrap<WrappedArray>(VT);
  WrappedArray* s = ObjectWrap::Unwrap<WrappedArray>(S);

  float *data = new float[ self->m_ * self->n_ ] ;
  memcpy( data, self->data_, sizeof(float) * self->m_ * self->n_ ) ; 

  float *superb = new float[ ::min(self->m_,self->n_) ] ;
  int rc = LAPACKE_sgesvd(
    CblasColMajor,
    'A', 'A',
    self->m_,
    self->n_,
    data,
    self->m_,
    s->data_,
    u->data_,
    self->m_,
    vt->data_,
    self->n_,
    superb ) ;

  delete superb ;
  delete data ;

  if( rc != 0 ) {
    char *msg = new char[ 1000 ] ;
    snprintf( msg, 1000, "Internal failure - sgesvd() failed with %d", rc ) ;
    isolate->ThrowException(Exception::TypeError( String::NewFromUtf8(isolate, msg)));
    delete msg ;
    args.GetReturnValue().Set( Undefined(isolate) );
  }
  else {
    args.GetReturnValue().Set( result );
  }
}

/**
	Principal component analysis

	Calculate the PCA factor of a matrix. The returned factor can be multiplies by any
	observation (row) or observations (the whole matrix) to get a reduced dimension
	set of features. The amount of information (variance) to keep is passed in as 
	an argument.

	IMPORTANT: the target features should be mean normalized for this to be accurate. 
	Mean normalized data has a zero mean for each feature ( each col mean = 0 ).
	


	The returned factor is a NxK array. Where K<M. We can reduce the dimensions of 
	the target from M to K by multiplying the observation(s) by the factor.
	
	In this example we extract the mean - in case we want to normalize unseen features.

	\code{.js}
	
	var lalg = require('lalg');

	var A = lalg.rand(10) ;	   // a 10x10 random array

	var mean = A.mean() ) ;    // keep the mean for later use

	// normalize the array befopre PCA
	var factor = A.sub( mean ).pca( 0.95 )  ; 	// keep 95% of the information

	// normalize any data we will reduce as well (need not be the same as the inputs )
	var ARD = A.sub(mean).mul( factor ) ;		// A's features mapped to a reduced set of dimensions

	\endcode

	@param the amount of variance to return (0 to 1.0). Default is 0.97
	@return a matrix to multiple an obeservation (matrix) by to reduce its dimensionality
	
	@see Diag
*/
void WrappedArray::Pca( const v8::FunctionCallbackInfo<v8::Value>& args )
{
  Isolate* isolate = args.GetIsolate();
  Local<Context> context = isolate->GetCurrentContext() ;

  float variance = args[0]->IsUndefined() ? 0.97f : args[0]->NumberValue();

  WrappedArray* self = ObjectWrap::Unwrap<WrappedArray>(args.Holder());

  int ls = std::min(self->n_,self->m_) ;
  float *vt = new float[ self->n_ * self->n_ ] ;
  float *s  = new float[ ls ] ;
  float *superb = new float[ self->m_ * self->n_ ] ;

  int rc = LAPACKE_sgesvd(
    CblasColMajor,
    'N', 'A',
    self->m_,
    self->n_,
    self->data_,
    self->m_,
    s,
    NULL,
    self->m_,
    vt,
    self->n_,
    superb ) ;

  float sum = 0 ;
  for( int i=0 ; i<ls ; i++ ) {
    sum += s[i] ;
  }
  float tot = 0 ;
  int k = ls ;
  for( int i=0 ; i<ls ; i++ ) {
    tot += s[i] / sum ;
    if( tot >= variance ) { k = (i+1) ; break ; }
  }

  if( rc != 0 ) {
    char *msg = new char[ 1000 ] ;
    snprintf( msg, 1000, "Internal failure - sgesvd() failed with %d", rc ) ;
    isolate->ThrowException(Exception::TypeError( String::NewFromUtf8(isolate, msg)));
    delete msg ;
    args.GetReturnValue().Set( Undefined(isolate) );
  }
  else {
    EscapableHandleScope scope(isolate) ;

    Local<Function> cons = Local<Function>::New(isolate, constructor);

    const unsigned argc = 2;
    Local<Value> argvu[argc] = { Integer::New( isolate,self->n_ ), Integer::New( isolate,k ) };
    Local<Object> instance = cons->NewInstance(context, argc, argvu).ToLocalChecked() ;
    scope.Escape( instance ) ;
    WrappedArray* result = node::ObjectWrap::Unwrap<WrappedArray>( instance ) ;
    int ixa = 0 ;
    for( int r=0 ; r<k ; r++ ) {
      int ixb = r ;
      for( int c=0 ; c<result->m_ ; c++ ) {
        result->data_[ixa++] = vt[ixb] ;
        ixb += self->n_ ;
      }
    }
    args.GetReturnValue().Set( instance );
  }
  delete superb ;
  delete vt ;
  delete s ;
}



/** 
	Returns a new square matrix, where the principal diagonal
	is formed from a vector. The size of the array is the vector
	length squared. All other elements are zero.

	@param a WrappedArray containing a vector to use as the diagonal
*/
void WrappedArray::Diag( const v8::FunctionCallbackInfo<v8::Value>& args )
{
  Isolate* isolate = args.GetIsolate();
  Local<Context> context = isolate->GetCurrentContext() ;
  EscapableHandleScope scope(isolate) ; ;

  WrappedArray* self = ObjectWrap::Unwrap<WrappedArray>( args[0]->ToObject() );
  int m = args[1]->IsUndefined() ? self->m_ : args[1]->NumberValue();

  const unsigned argc = 2;
  Local<Value> argv[argc] = { Integer::New( isolate,m), Integer::New( isolate,self->m_ ) }
  ;
  Local<Function> cons = Local<Function>::New(isolate, constructor);
  Local<Object> instance = cons->NewInstance(context, argc, argv).ToLocalChecked() ;

  scope.Escape(instance);
  WrappedArray* rc = node::ObjectWrap::Unwrap<WrappedArray>( instance ) ;
  int sz = rc->m_ * rc->n_ ;
  for( int i=0 ; i<sz ; i++ ) {
    rc->data_[i] = 0 ;
  }
  for( int i=0 ; i<self->m_ ; i++ ) {
    rc->data_[i*rc->m_+i] = self->data_[i] ;
  }
  args.GetReturnValue().Set( instance );
}


/** 
	Returns a new square identity matrix.

	An identity matrix is a leading diagonal of 1.0 and all
	other elements are 0. Any matrix multiplied by (a suitable shaped) identity
	will return itself.

	@param the number of rows (m) and columns (n)
	@return a new matrix, each element is set to 1.0
*/
void WrappedArray::Eye( const v8::FunctionCallbackInfo<v8::Value>& args )
{
  Isolate* isolate = args.GetIsolate();
  Local<Context> context = isolate->GetCurrentContext() ;
  EscapableHandleScope scope(isolate) ; ;

  const unsigned argc = 2;
  Local<Value> argv[argc] = { args[0], args[1]->IsUndefined() ? args[0] : args[1] };
  Local<Function> cons = Local<Function>::New(isolate, constructor);
  Local<Object> instance = cons->NewInstance(context, argc, argv).ToLocalChecked() ;

  scope.Escape(instance);

  WrappedArray* self = node::ObjectWrap::Unwrap<WrappedArray>( instance ) ;
  int sz = self->m_ * self->n_ ;
  for( int i=0 ; i<sz ; i++ ) {
    self->data_[i] = 0 ;
  }
  for( int i=0 ; i<self->m_ ; i++ ) {
    self->data_[i*self->m_+i] = 1.f ;
  }

  args.GetReturnValue().Set( instance );
}


/** 
	Returns a new matrix with all values set to .0

	@param the number of rows (m) defaults to 0
	@param the number of columns (n) defaults to m
	@return a new matrix, each element is set to 0.0
*/

void WrappedArray::Zeros( const v8::FunctionCallbackInfo<v8::Value>& args )
{
  Isolate* isolate = args.GetIsolate();
  Local<Context> context = isolate->GetCurrentContext() ;
  EscapableHandleScope scope(isolate) ; ;

  const unsigned argc = 2;
  Local<Value> argv[argc] = { args[0], args[1]->IsUndefined() ? args[0] : args[1] };
  Local<Function> cons = Local<Function>::New(isolate, constructor);
  Local<Object> instance = cons->NewInstance(context, argc, argv).ToLocalChecked() ;

  scope.Escape(instance);

  WrappedArray* self = node::ObjectWrap::Unwrap<WrappedArray>( instance ) ;
  int sz = self->m_ * self->n_ ;
  for( int i=0 ; i<sz ; i++ ) {
    self->data_[i] = 0 ;
  }

  args.GetReturnValue().Set( instance );
}




/** 
	Returns a new matrix with all values set to 1.0

	@param the number of rows (m) defaults to 0
	@param the number of columns (n) defaults to m
	@return a new matrix, each element is set to 1.0
*/
void WrappedArray::Ones( const FunctionCallbackInfo<v8::Value>& args )
{
  Isolate* isolate = args.GetIsolate();
  Local<Context> context = isolate->GetCurrentContext() ;
  EscapableHandleScope scope(isolate) ; ;

  const unsigned argc = 2;
  Local<Value> argv[argc] = { args[0], args[1]->IsUndefined() ? args[0] : args[1] };
  Local<Function> cons = Local<Function>::New(isolate, constructor);
  Local<Object> instance = cons->NewInstance(context, argc, argv).ToLocalChecked() ;

  scope.Escape(instance);

  WrappedArray* self = node::ObjectWrap::Unwrap<WrappedArray>( instance ) ;
  int sz = self->m_ * self->n_ ;
  for( int i=0 ; i<sz ; i++ ) {
    self->data_[i] = 1.f ;
  }

  args.GetReturnValue().Set( instance );
}


/** 
	Returns a new matrix with all values set to a random integer between -10 and 10 (inclusive)
	
	The randomness isn't very good, it's used for testing.

	@param[in,deafult=0] m the number of rows 
	@paramin,default=m] n the number of columns

	@return a new matrix, each element is set to a random integer
*/
void WrappedArray::Rand( const v8::FunctionCallbackInfo<v8::Value>& args )
{
  Isolate* isolate = args.GetIsolate();
  Local<Context> context = isolate->GetCurrentContext() ;
  EscapableHandleScope scope(isolate) ; ;

  const unsigned argc = 2;
  Local<Value> argv[argc] = { args[0], args[1]->IsUndefined() ? args[0] : args[1] };
  Local<Function> cons = Local<Function>::New(isolate, constructor);
  Local<Object> instance = cons->NewInstance(context, argc, argv).ToLocalChecked();

  scope.Escape(instance);

  srand( time(NULL) ) ;

  WrappedArray* self = node::ObjectWrap::Unwrap<WrappedArray>( instance ) ;
  int sz = self->m_ * self->n_ ;
  for( int i=0 ; i<sz ; i++ ) {
    self->data_[i] = ( rand() % 21 ) - 10 ;
  }
  args.GetReturnValue().Set( instance );
}



/** 
	Solves a function for its minimum
	
	Solves the function starting at the given Array based starting position. After
	this is finished the array is set to the global minimum.

	This requires a gradient function object to be passed as a parameter.
	The gradient function is an object with two methods:
	- value( lalg.Array x ) returns Number 
	- gradient( lalg.Array x ) returns lalg.Array gradients
	
	The solver name parameter should be one of
	- BFGS
	- CGD
	- NEWTON
	- NELDERMEAD
	- LBFGS	
	- CMAES

	\code{.js}

	var lalg = require( "lalg" ) ;
	var R = lalg.rand( 2,1 ) ;
	function gradFunction() {
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

	var f = new gradFunction() ;
	A = new lalg.Array( 2, 1, [ 0.1, 0.1] ) ;
	A.solve( f, "LBFGS" ) ;   
	console.log( "Actual Min:", f.value(A), "@", Array.from(A) ) ;
		
	\endcode
	
	
	@param[in] the function to solve
	@param[in,default='BFGS'] the solver name one of [ "BFGS", "CGD", "NEWTON","NELDERMEAD", "LBFGS","CMAES" ]
	
*/
void WrappedArray::Solve( const v8::FunctionCallbackInfo<v8::Value>& args )
{
	WrappedArray::SolveHelper( args, false, 2 ) ;
}


/*
    Do not use (yet)

	Solves a function for its minimum
	
	Solves the function starting at the given Array based starting position. After
	this is finished the array is set to the global minimum.

	@see Solve
	
	@param [in] the function to solve
	@param [in,default=BFGS"] the solver name one of [ "BFGS", "CGD", "NEWTON","NELDERMEAD", "LBFGS","CMAES" ]
	@Wparam[in, optional] callback of prototype function(err) {} 
	
*/
void WrappedArray::Solvep( const v8::FunctionCallbackInfo<v8::Value>& args )
{
	WrappedArray::SolveHelper( args, true, 2 ) ;
}

/*
	This handles the actual solving in an background thread
*/
void WrappedArray::SolveHelper( const v8::FunctionCallbackInfo<v8::Value>& args, bool block, int callbackIndex )
{
  Isolate* isolate = args.GetIsolate();
  Local<Context> context = isolate->GetCurrentContext() ;
  EscapableHandleScope scope(isolate) ; ;

  scope.Escape(args.Holder());

  if( args[0]->IsObject() ) {
      int solverIndex = args[1]->IsUndefined() ? 0 : args[1]->NumberValue() ;
      if( !args[1]->IsUndefined() && args[1]->IsString() ) {
	Local<String> solverName = args[1]->ToString() ;
    	char *c = new char[solverName->Utf8Length()] ;
	solverName->WriteUtf8( c, 32 ) ;
	if( !::strcasecmp( "BFGS", c ) ) solverIndex = 0 ;
	else if( !::strcasecmp( "CGD", c ) ) solverIndex = 1 ;
	else if( !::strcasecmp( "NEWTON", c ) ) solverIndex = 2 ;
	else if( !::strcasecmp( "NELDERMEAD", c ) ) solverIndex = 3 ;
	else if( !::strcasecmp( "LBFGS", c ) ) solverIndex = 4 ;
	else if( !::strcasecmp( "CMAES", c ) ) solverIndex = 5 ;
	delete c ;
      }

	Local<Object> ufg = args[0]->ToObject() ;
      
      WrappedArray::PrepareWork( args, block, callbackIndex, WrappedArray::SolveWorkAsync, 
				 args.Holder(), ufg, solverIndex ) ;
  }
}



/*
	This calls the solvers in, possibly, a background thread.
*/
void WrappedArray::SolveWorkAsync( uv_work_t *req )
{
  Work *work = static_cast<Work *>(req->data);

  Isolate *isolate = work->isolate ;

  WrappedArray* self = work->self ;

  Local<Object> selfObj = Local<Object>::New(isolate,work->selfObj) ;
  Local<Object> userFunctionHolder = Local<Object>::New(isolate,work->xtraObj) ;
  UserGradientFunction f( isolate, self, selfObj, userFunctionHolder ) ;

  // choose a starting point
  UserGradientFunction::TVector x(2); x << self->data_[0], self->data_[1] ;

  int solverIndex = work->xtraInt ;

  if( solverIndex==0 ) { cppoptlib::BfgsSolver<UserGradientFunction> solver ; solver.minimize(f, x); }
  if( solverIndex==1 ) { cppoptlib::ConjugatedGradientDescentSolver<UserGradientFunction> solver ; solver.minimize(f, x); }
  if( solverIndex==2 ) { cppoptlib::NewtonDescentSolver<UserGradientFunction> solver ; solver.minimize(f, x); }
  if( solverIndex==3 ) { cppoptlib::NelderMeadSolver<UserGradientFunction> solver ; solver.minimize(f, x); }
  if( solverIndex==4 ) { cppoptlib::LbfgsSolver<UserGradientFunction> solver ; solver.minimize(f, x); }
  if( solverIndex==5 ) { cppoptlib::CMAesSolver<UserGradientFunction> solver ; solver.minimize(f, x); }

}



/**
	get javascript iterator

	This creates an iterator object so we can iterate over the elements of a matrix. 
	
	\code{.js}

	var lalg = require( "lalg" ) ;
	var R = lalg.rand( 10,5 ) ;
	var arr = Array.from( R ) ;
	
	\endcode

	@return an object with a method next() that matches specs for an iterator
*/
void WrappedArray::MakeIterator( const FunctionCallbackInfo<v8::Value>& args  )
{
  Isolate* isolate = args.GetIsolate();
  EscapableHandleScope scope(isolate) ;

  Local<Object> xtra = Object::New(isolate) ;	// xtra stuff used by the iterator (how to get next)
  xtra->Set(String::NewFromUtf8(isolate, "index"), Integer::New( isolate,0 ) );
  xtra->Set(String::NewFromUtf8(isolate, "array"), args.Holder() ) ;

  // create an object (result) to return from the call to @@iterator 
  Local<Object> result = Object::New(isolate);   
  scope.Escape( result ) ;
  args.GetReturnValue().Set( result );

  // then set 1 function on the result prototype: 'next'
  Local<FunctionTemplate> tplNext = FunctionTemplate::New(isolate, WrappedArray::NextCallback, xtra );
  tplNext->SetClassName(String::NewFromUtf8(isolate, "next"));
  result->Set(String::NewFromUtf8(isolate, "next"), tplNext->GetFunction() );
}


/*
	This is the callback for the iterator, it uses a temporary data structure (xtra)
	to hold the index and a reference to the wrapped array. It's not meant to be called
	directly, it could be a stand-alone method.
*/
void WrappedArray::NextCallback( const v8::FunctionCallbackInfo<v8::Value>& args ) {

  Isolate* isolate = args.GetIsolate();
  Local<Context> context = isolate->GetCurrentContext() ;
  EscapableHandleScope scope(isolate) ;
  
  Local<Object> xtra = Local<Object>::Cast( args.Data() ) ;
  // index++
  Local<Value> indexValue = xtra->Get( context, String::NewFromUtf8(isolate, "index") ).ToLocalChecked() ;
  int index = indexValue->NumberValue() ;
  xtra->Set(String::NewFromUtf8(isolate, "index"), Integer::New( isolate, index+1 ) );

  // access the underlying matrix we are iterating over (from xtra params)
  Local<Value> array = xtra->Get( context, String::NewFromUtf8(isolate, "array") ).ToLocalChecked() ;  
  Local<Object> obj = Local<Object>::Cast( array ) ;
  WrappedArray* self = ObjectWrap::Unwrap<WrappedArray>( obj );

  Local<Object> result = Object::New(isolate);   // the return from next() as an object
  if( index>=(self->m_*self->n_) ) {
    result->Set(String::NewFromUtf8(isolate, "value"), Undefined(isolate) ) ;
    result->Set(String::NewFromUtf8(isolate, "done"),  Boolean::New( isolate, true ) ) ;
  } else {
    result->Set(String::NewFromUtf8(isolate, "value"), Number::New( isolate,self->data_[index] ) ) ;
    result->Set(String::NewFromUtf8(isolate, "done"),  Boolean::New( isolate, false ) ) ;
  }

  scope.Escape( result ) ;
  args.GetReturnValue().Set( result );
}


/**
	Read a matrix from a stream

	This takes a stream which emits data events for each row. One such stream is the fast-csv.

	\code{.js}
	
	var lalg = require('lalg');
	var fs = require('fs');
	var csv = require("fast-csv");

	const rr = fs.createReadStream('wine.csv');
	var csvStream = csv() ;
	rr.pipe(csvStream);

	lalg.read( csvStream )
	.then( function(X) {
	        console.log( X ) ;
	.catch( function(err) {
	        console.log( "Fail", err ) ;
	});

	\endcode

	@param a stream that presents data events with a single Array of numbers
	@return a Promise that will resolve to a matrix
*/
void WrappedArray::Read( const v8::FunctionCallbackInfo<v8::Value>& args )
{
  Isolate* isolate = args.GetIsolate();
  Local<Context> context = isolate->GetCurrentContext() ;
  EscapableHandleScope scope(isolate) ; ;
  
  const unsigned argc = 2;
  Local<Value> argv[argc] = { Integer::New( isolate, 0 ), Integer::New( isolate, 0 )  };
  Local<Function> cons = Local<Function>::New(isolate, constructor);
  Local<Object> instance = cons->NewInstance(context, argc, argv).ToLocalChecked();

  scope.Escape(instance);

  Local<Promise::Resolver> resolver = v8::Promise::Resolver::New( isolate ) ;
  args.GetReturnValue().Set( resolver->GetPromise() ) ;

  Local<Object> xtra = Object::New(isolate);
  xtra->Set(String::NewFromUtf8(isolate, "array"), instance ) ;
  xtra->Set(String::NewFromUtf8(isolate, "promise"), resolver ) ;

  Local<Object> readable = args[0]->ToObject() ;
  Local<Value> ontmp = readable->Get( context, String::NewFromUtf8(isolate, "on") ).ToLocalChecked() ;
  Local<Function> on = Local<Function>::Cast( ontmp ) ;

  Local<FunctionTemplate> tplData = FunctionTemplate::New(isolate, WrappedArray::DataCallback, xtra );
  Local<Function> fnData = tplData->GetFunction();
  Local<Value> argv2[2] = { String::NewFromUtf8(isolate, "data"), fnData };
  on->CallAsFunction( context, readable, 2, argv2 );

  Local<FunctionTemplate> tplDataEnd = FunctionTemplate::New(isolate, WrappedArray::DataEndCallback, xtra );
  Local<Function> fnDataEnd = tplDataEnd->GetFunction();
  Local<Value> argv3[2] = { String::NewFromUtf8(isolate, "end"), fnDataEnd };
  on->CallAsFunction( context, readable, 2, argv3 );

}







/*
	This is the callback function that is called when reading
	a new array from a stream.

	The callback is expected to receive an Array of numbers.
	Each data chunk is assumed to be a row in the array, however we
	store it as column based. Once the end event is processed we
	transpose the array. This way is not the most efficient, but often
	data is provided by text files in row based order (each row is an 
	observation). We don't know the number of rows we will read so 
	it's saved in column order then transposed.

	The first line (array) defines the number of columns. Each subsequent 
	array must contain that many values, or the 'missing' elements
	are undefined.

	@param [in] an array of numbers. 
*/
void WrappedArray::DataCallback(const FunctionCallbackInfo<Value>& args) {

  Isolate* isolate = args.GetIsolate();
  Local<Context> context = isolate->GetCurrentContext() ;
  Local<Object> xtra = Local<Object>::Cast( args.Data() ) ;
  Local<Value> array = xtra->Get( context, String::NewFromUtf8(isolate, "array") ).ToLocalChecked() ;
  Local<Object> obj = Local<Object>::Cast( array ) ;
  WrappedArray* self = ObjectWrap::Unwrap<WrappedArray>( obj );

// If we didn't get an array - we'll ignore it :(
  if( args[0]->IsArray() ) {

    Local<Array> cols = Local<Array>::Cast(args[0]->ToObject() );
    if( self->m_ == 0 ) self->m_ = cols->Length() ; // first time count cols in data to be row count
    int numColsToRead = ::min( (uint32_t)self->m_, cols->Length() ) ;  // #items to read

// If we've overflowed the current buffer - let's expand it
    if( self->dataSize_ <= (self->n_*self->m_) ) {
      float *tmp = self->data_ ;
      size_t oldDataSize = self->dataSize_ ;
      self->dataSize_ = ::max( (self->n_*2), self->n_+16 )*self->m_  ;
      self->data_ = new float[ self->dataSize_ ] ; 
      memcpy( self->data_, tmp, oldDataSize*sizeof(float) ) ;
      delete tmp ;
    }
// Then copy the array data to our local buffer
    int p = (self->n_*self->m_) ;
    for( int i=0 ; i<numColsToRead ; i++ ) {
	self->data_[p] = cols->Get( context, i ).ToLocalChecked()->NumberValue() ;
        p++ ;
    }
  }
// we just added a column
  self->n_++ ;
}

/*
	After the data is read this method is called. All we do here is to 
	transpose the array. As a side effect we do resize the array buffer.
*/
void WrappedArray::DataEndCallback(const FunctionCallbackInfo<Value>& args ) {

  Isolate* isolate = args.GetIsolate();
  Local<Context> context = isolate->GetCurrentContext() ;

  Local<Object> xtra = Local<Object>::Cast( args.Data() ) ;
  Local<Value> array = xtra->Get( context, String::NewFromUtf8(isolate, "array") ).ToLocalChecked() ;
  Local<Object> obj = Local<Object>::Cast( array ) ;
  WrappedArray* self = ObjectWrap::Unwrap<WrappedArray>( obj );

  if( !self->isVector ) { // don't transpose a vector - just switch m & n later
    self->dataSize_ = self->m_ * self->n_ ;
    float *data = new float[ self->dataSize_] ;
    int ixa = 0 ;
    for( int r=0 ; r<self->m_ ; r++ ) {
      int ixb = r ;
      for( int c=0 ; c<self->n_ ; c++ ) {
        data[ixa++] = self->data_[ixb+(c*self->m_)] ;
      }
    }
    delete self->data_ ;
    self->data_ = data ;
  }

// Swap rows & cols
  int tmp = self->m_ ;
  self->m_ = self->n_ ;
  self->n_ = tmp ;

// Then get the promise from the work data and resolve it
  Local<Value> promise = xtra->Get( context, String::NewFromUtf8(isolate, "promise") ).ToLocalChecked() ;
  Local<Promise::Resolver> resolver = Local<Promise::Resolver>::Cast( promise ) ;
  
  resolver->Resolve( obj ) ;
}






/*
	A generic method to be used for all non-blocking code. It should not have any instruction
	specific code in here. 
	This prepares the Work struct in a gneral way:
	* create the Work struct:
	* set the thread data in Work with itself
	* clear the err flag ( a char* for an error message, which will be passed back to caller )
	* if the 1st argument to the original caller is a number use it
	* if the 1st argument to the original caller exists assume it's another matrix
	* set the provided result to the work struct
	* if there's nothing provided at args[callbackIndex] create a promise
	* if a function is  provided at args[callbackIndex] use it as a callback

	Then if we're in non-blocking mode - create a new thread to run the provided work function. If
	we're in blocking mode just call the provided work function directly

	@param args, the original args to the function we are making into a promise
	@param block - will we run in blocking mode (need a callback or we create a promise )
	@param callbackIndex - which arg might contain a callback, if missing go to promise mode
	@param work_cb  this is the C++ function that executes the thread body (or direct in direct mode)
*/
void WrappedArray::PrepareWork( const v8::FunctionCallbackInfo<v8::Value>& args, bool block, int callbackIndex, uv_work_cb work_cb, Local<Object> instance ) {
  WrappedArray::PrepareWork( args, block, callbackIndex, work_cb, instance, Local<Object>(), 0 ) ;
}

void WrappedArray::PrepareWork( const v8::FunctionCallbackInfo<v8::Value>& args, bool block, int callbackIndex, uv_work_cb work_cb, Local<Object> instance, Local<Object> xtraObj, int xtraInt ) {
  Isolate* isolate = args.GetIsolate();

  EscapableHandleScope scope(isolate) ;

  WrappedArray *self = ObjectWrap::Unwrap<WrappedArray>(args.Holder());
// Work is used to pass info into our execution threda
  Work *work = new Work();
  work->request.data = work;   // 1st is to set the work so the thread can see our Work struct
  work->err = NULL ;
  work->isolate = isolate ;

  work->other = NULL ;
  if( args[0]->IsNumber() ) {
    work->otherNumber = args[0]->NumberValue() ;
  } else if( args[0]->IsObject() ) {   // is first arg. an array ?
    Local<Object> possibleOther = args[0]->ToObject() ;
    Local<String> className = possibleOther->GetConstructorName() ;
    char *c = new char[className->Utf8Length() + 10 ] ;
    className->WriteUtf8( c ) ;
    if( !::strcmp( "Array", c ) ) {
	    WrappedArray *other = ObjectWrap::Unwrap<WrappedArray>( possibleOther ) ;
	    work->other = other ;
    }
    delete c ;
  }

  work->self = self ;
  work->result = ObjectWrap::Unwrap<WrappedArray>( instance )  ;
// It seems to be best that we create the result in the caller's context
// So we do it here
  work->resultLocal.Reset( isolate, instance ) ;
  work->selfObj.Reset( isolate, instance ) ;

  if( !xtraObj.IsEmpty() ) {
    work->xtraObj.Reset( isolate, xtraObj ) ;
  }
  work->xtraInt = xtraInt ;

// If we have a second arg - it should be a callback
// So setup the Work struct in Promise or callback mode

  if( block ) {
    if( args[callbackIndex]->IsUndefined() ) {
      Local<Promise::Resolver> resolver = v8::Promise::Resolver::New( isolate ) ;
      work->resolver.Reset(isolate, resolver ) ;
      args.GetReturnValue().Set( resolver->GetPromise()  ) ;
    } else if( args[callbackIndex]->IsFunction() ) {
      Local<Function> callback = Local<Function>::Cast(args[callbackIndex]);
      work->callback.Reset(isolate, callback ) ;
      args.GetReturnValue().Set( Undefined(isolate) ) ;
    }
  }
// OK - if we don't have a callback or a promise we're in blocking mode
// execute in the current thread. This calls the 2 UV thread methods
// directly
  if( work->resolver.IsEmpty() && work->callback.IsEmpty() ) {
    work_cb( &work->request ) ;
    WrappedArray::WorkAsyncComplete( &work->request, -1 ) ;
    args.GetReturnValue().Set( instance ) ;
  } else {
// Otherwise create a new thread to do the work & return
// The proper return value (undefined for callback mode or a promise is already set)
    uv_queue_work(uv_default_loop(),&work->request, work_cb, WrappedArray::WorkAsyncComplete ) ;
  }
}



/*
	When the any work is done, get the result from the 'work'
	and call either the Promise or callback success methods.
	This handles the error case too.
*/
void WrappedArray::WorkAsyncComplete(uv_work_t *req,int status)
{    
    // read work from the uv thread handle
    Work *work = static_cast<Work *>(req->data);
    Isolate *isolate = work->isolate  ;
    HandleScope scope(isolate) ;

    if( work->err != NULL ) {
      if( !work->resolver.IsEmpty() ) {
        Local<Promise::Resolver> resolver = Local<Promise::Resolver>::New(isolate,work->resolver) ;
        resolver->Reject( String::NewFromUtf8(isolate, work->err) ) ;
        work->resolver.Reset();   // free the persistent storage 
      } else if( !work->callback.IsEmpty() ) {
        Handle<Value> argv[] = { String::NewFromUtf8(isolate, work->err), Null(work->isolate) };
        Local<Function>::New(isolate, work->callback)-> Call(isolate->GetCurrentContext()->Global(), 2, argv);
        work->callback.Reset();   // free the persistent storage 
      } else {
        isolate->ThrowException(Exception::TypeError( String::NewFromUtf8(isolate, work->err) ) );
      }
      delete work->err ;
    } else {
    // convert the persistent storage to Local - suitable for a return
      Local<Object> rc = Local<Object>::New(isolate,work->resultLocal) ;
      work->resultLocal.Reset();	// free the persistent storage

	// Then choose which return method (Promise or callback) to use to return the local<Object>
      if( !work->callback.IsEmpty() ) {
        Handle<Value> argv[] = { Null(isolate), rc };
        Local<Function>::New(isolate, work->callback)-> Call(isolate->GetCurrentContext()->Global(), 2, argv);
        work->callback.Reset();   // free the persistent storage 
      } else if( !work->resolver.IsEmpty() ) {
        Local<Promise::Resolver> resolver = Local<Promise::Resolver>::New(isolate,work->resolver) ;
        resolver->Resolve( rc ) ;
        work->resolver.Reset();  // free the persistent storage
      }
    }

    work->xtraObj.Reset() ;
    work->selfObj.Reset() ;

    delete work;	// finished
}





/*
	This is a nodejs defined method to get attributes. 
	It's pretty simple to follow - we return one of the following
	- m count of rows
	- n count of columns
	- length total size of the array MxN
*/
void WrappedArray::GetCoeff(Local<String> property, const PropertyCallbackInfo<Value>& info)
{
  Isolate* isolate = info.GetIsolate();
  //Local<Context> context = isolate->GetCurrentContext() ;
  WrappedArray* self = ObjectWrap::Unwrap<WrappedArray>(info.This());

  v8::String::Utf8Value s(property);
  std::string str(*s);

  if ( str == "m") {
    info.GetReturnValue().Set(Number::New(isolate, self->m_));
  } else if (str == "n") {
    info.GetReturnValue().Set(Number::New(isolate, self->n_));
  } else if (str == "length") {
    info.GetReturnValue().Set(Number::New(isolate, self->n_*self->m_ ));
  } else if (str == "maxPrint") {
    info.GetReturnValue().Set(Number::New(isolate, self->maxPrint_ ));
  } else if (str == "name" && self->name_ != NULL ) {
    info.GetReturnValue().Set( String::NewFromUtf8(isolate, self->name_) ) ;
  }
}




/*
	This is a nodejs defined method to set writeable attributes. 

	@see Reshape (to adjust m and n)
*/
void WrappedArray::SetCoeff(Local<String> property, Local<Value> value, const PropertyCallbackInfo<void>& info)
{
  WrappedArray* obj = ObjectWrap::Unwrap<WrappedArray>(info.This());

  v8::String::Utf8Value s(property);
  std::string str(*s);

  if ( str == "maxPrint") {
    obj->maxPrint_ = value->NumberValue();
  } else if (str == "name" ) {
    char *c = new char[value->ToString()->Utf8Length()] ;
    value->ToString()->WriteUtf8( c, 32 ) ;
    delete obj->name_ ;
    obj->name_ = c;
  }

}

/*
	Use for the new operators on the module. Not meant to be
	called directly.

*/
void CreateObject(const FunctionCallbackInfo<Value>& info)
{
  Isolate* isolate = info.GetIsolate();
  HandleScope scope(isolate);
  info.GetReturnValue().Set(WrappedArray::NewInstance(info) );
}

/*
	The module init script - called by nodejs at load time
*/
void InitArray(Local<Object> exports, Local<Object> module)
{
  WrappedArray::Init(exports, module);
}


NODE_MODULE(linalg, InitArray)
