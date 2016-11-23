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

using namespace std;
using namespace v8;

void CreateObject(const FunctionCallbackInfo<Value>& info) ;
/**
 This is the main class to represent a matrix. It is a nodejs compatible
 object.

 Arrays are stored in column order, this is to simplify access to (some)
 libraries (esp. cuda).
  
 See the README.md for details on how to use this.

*/
class WrappedArray : public node::ObjectWrap
{
  public:
    /**
       Initialize the prototype of class Array. Called when the module is loaded.
       ALl the methods and attributes are defined here.
    */
    static void Init(v8::Local<v8::Object> exports, Local<Object> module) {
      Isolate* isolate = exports->GetIsolate();

      // Prepare constructor template and name of the class
      Local<FunctionTemplate> tpl = FunctionTemplate::New(isolate, New);
      tpl->SetClassName(String::NewFromUtf8(isolate, "Array"));
      tpl->InstanceTemplate()->SetInternalFieldCount(4);

      // Prototype - methods. These can be called from javascript
      NODE_SET_PROTOTYPE_METHOD(tpl, "toString", ToString);
      NODE_SET_PROTOTYPE_METHOD(tpl, "inspect", Inspect);
      NODE_SET_PROTOTYPE_METHOD(tpl, "dup", Dup);
      NODE_SET_PROTOTYPE_METHOD(tpl, "transpose", Transpose);
      NODE_SET_PROTOTYPE_METHOD(tpl, "transposei", Transposei);
      NODE_SET_PROTOTYPE_METHOD(tpl, "muli", Muli);
      NODE_SET_PROTOTYPE_METHOD(tpl, "mul", Mul);
      NODE_SET_PROTOTYPE_METHOD(tpl, "mmul", Mmul);
      NODE_SET_PROTOTYPE_METHOD(tpl, "mmulp", Mmulp);
      NODE_SET_PROTOTYPE_METHOD(tpl, "asum", Asum);
      NODE_SET_PROTOTYPE_METHOD(tpl, "sum", Sum);
      NODE_SET_PROTOTYPE_METHOD(tpl, "mean", Mean);
      NODE_SET_PROTOTYPE_METHOD(tpl, "add", Add);
      NODE_SET_PROTOTYPE_METHOD(tpl, "addi", Addi);
      NODE_SET_PROTOTYPE_METHOD(tpl, "sub", Sub);
      NODE_SET_PROTOTYPE_METHOD(tpl, "subi", Subi);
      NODE_SET_PROTOTYPE_METHOD(tpl, "inv", Inv);
      NODE_SET_PROTOTYPE_METHOD(tpl, "invi", Invi);
      NODE_SET_PROTOTYPE_METHOD(tpl, "svd", Svd);
      NODE_SET_PROTOTYPE_METHOD(tpl, "pca", Pca);
      NODE_SET_PROTOTYPE_METHOD(tpl, "getRows", GetRows);
      NODE_SET_PROTOTYPE_METHOD(tpl, "removeRow", RemoveRow);
      NODE_SET_PROTOTYPE_METHOD(tpl, "getColumns", GetCols);
      NODE_SET_PROTOTYPE_METHOD(tpl, "removeColumn", RemoveCol);
      NODE_SET_PROTOTYPE_METHOD(tpl, "reshape", Reshape);


      // Factories - call these on the module - to create a new array
      NODE_SET_METHOD(exports, "eye", Eye);
      NODE_SET_METHOD(exports, "ones", Ones);
      NODE_SET_METHOD(exports, "zeros", Zeros);
      NODE_SET_METHOD(exports, "rand", Rand);
      NODE_SET_METHOD(exports, "diag", Diag);
      NODE_SET_METHOD(exports, "read", Read);

      // define how we access the attributes
      tpl->InstanceTemplate()->SetAccessor(String::NewFromUtf8(isolate, "m"), GetCoeff, SetCoeff);
      tpl->InstanceTemplate()->SetAccessor(String::NewFromUtf8(isolate, "n"), GetCoeff, SetCoeff);
      tpl->InstanceTemplate()->SetAccessor(String::NewFromUtf8(isolate, "length"), GetCoeff);

      constructor.Reset(isolate, tpl->GetFunction());
      exports->Set(String::NewFromUtf8(isolate, "Array"), tpl->GetFunction());
    }
    /**
	The nodejs constructor. 
	@see New
    */
    static Local<Object> NewInstance(const FunctionCallbackInfo<Value>& args);
  private:
   /**
	The C++ constructor, creates an mxn array.
   */
    explicit WrappedArray(int m=0, int n=0) : m_(m), n_(n) {
      dataSize_ = m*n ;
      data_ = new float[dataSize_] ;
      isVector = m==1 || n== 1 ;
    }
    /**
	The destructor needs to free the data buffer
    */
    ~WrappedArray() { 
	delete data_ ;
    }

    /**
	The nodejs constructor 
	It takes up to 3 args: 
		** the number of rows (m) defaults to 0
		** the number of columns (n) defaults to m
		** an array of data to use for the array (column major order)
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
        }

        if( self != NULL ) {  // double check we managed to create something
          self->Wrap(args.This());
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

/** ToString returns a string representation of the matrix */
    static void ToString(const FunctionCallbackInfo<Value>& args);
    static void Inspect(const FunctionCallbackInfo<Value>& args);
    static void Ones(const FunctionCallbackInfo<Value>& args);
    static void Zeros(const FunctionCallbackInfo<Value>& args);
    static void Eye(const FunctionCallbackInfo<Value>& args);
    static void Diag(const FunctionCallbackInfo<Value>& args);
    static void Read(const FunctionCallbackInfo<Value>& args);
    static void Rand(const FunctionCallbackInfo<Value>& args);
    static void Dup(const FunctionCallbackInfo<Value>& args);
    static void Transpose(const FunctionCallbackInfo<Value>& args);
    static void Transposei(const FunctionCallbackInfo<Value>& args);
    static void Mul(const FunctionCallbackInfo<Value>& args);
    static void Muli(const FunctionCallbackInfo<Value>& args);
    static void Mmul(const FunctionCallbackInfo<Value>& args);
    static void Mmulp(const FunctionCallbackInfo<Value>& args);
    static void Asum( const v8::FunctionCallbackInfo<v8::Value>& args ) ;
    static void Sum( const v8::FunctionCallbackInfo<v8::Value>& args ) ;
    static void Mean( const v8::FunctionCallbackInfo<v8::Value>& args ) ;
    static void Add( const v8::FunctionCallbackInfo<v8::Value>& args ) ;
    static void Addi( const v8::FunctionCallbackInfo<v8::Value>& args ) ;
    static void Sub( const v8::FunctionCallbackInfo<v8::Value>& args ) ;
    static void Subi( const v8::FunctionCallbackInfo<v8::Value>& args ) ;
    static void Inv( const v8::FunctionCallbackInfo<v8::Value>& args ) ;
    static void Invi( const v8::FunctionCallbackInfo<v8::Value>& args ) ;
    static void Svd( const v8::FunctionCallbackInfo<v8::Value>& args ) ;
    static void Pca( const v8::FunctionCallbackInfo<v8::Value>& args ) ;
    static void GetRows( const v8::FunctionCallbackInfo<v8::Value>& args ) ;
    static void RemoveRow( const v8::FunctionCallbackInfo<v8::Value>& args ) ;
    static void GetCols( const v8::FunctionCallbackInfo<v8::Value>& args ) ;
    static void RemoveCol( const v8::FunctionCallbackInfo<v8::Value>& args ) ;
    static void Reshape( const v8::FunctionCallbackInfo<v8::Value>& args ) ;

    static void GetCoeff(Local<String> property, const PropertyCallbackInfo<Value>& info);
    static void SetCoeff(Local<String> property, Local<Value> value, const PropertyCallbackInfo<void>& info);

    static v8::Persistent<v8::Function> constructor;
    int m_;
    int n_;
    float *data_ ;
    bool isVector ;
    bool isDiag ;
    int dataSize_ ;

    static void DataCallback(const FunctionCallbackInfo<Value>& args) ;
    static void DataEndCallback(const FunctionCallbackInfo<Value>& args) ;

    static void WorkAsyncComplete(uv_work_t *req,int status) ;
    static void MmulpWorkAsync(uv_work_t *req) ;
    static void ReadWorkAsync(uv_work_t *req) ;

};

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


void WrappedArray::ToString( const v8::FunctionCallbackInfo<v8::Value>& args )
{
  Isolate* isolate = args.GetIsolate();
  WrappedArray* self = ObjectWrap::Unwrap<WrappedArray>(args.Holder());

  int mm = std::min( self->m_, 10 ) ;
  int nn = std::min( self->n_, 10 ) ;
  char *rc = new char[ 100 + (mm * nn * 50) ] ;
  int n = sprintf( rc, "%d x %d %s\n", self->m_, self->n_, self->isVector?"Vector" : "" ) ;

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
    strcat( rc, "... ... " ) ;
    n+=3 ;
  }

  args.GetReturnValue().Set( String::NewFromUtf8( isolate, rc) );
  delete  rc ;
}


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


void WrappedArray::Transposei( const v8::FunctionCallbackInfo<v8::Value>& args )
{

  WrappedArray* self = ObjectWrap::Unwrap<WrappedArray>(args.Holder());
  if( !self->isVector ) {
    float *data = new float[ self->m_ * self->n_ ];
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

  int tmp = self->m_ ;
  self->m_ = self->n_ ;
  self->n_ = tmp ;
  args.GetReturnValue().Set( args.Holder() );
}


void WrappedArray::Transpose( const v8::FunctionCallbackInfo<v8::Value>& args )
{
  Isolate* isolate = args.GetIsolate();
  Local<Context> context = isolate->GetCurrentContext() ;

  WrappedArray* self = ObjectWrap::Unwrap<WrappedArray>(args.Holder());

  EscapableHandleScope scope(isolate) ; ;

  const unsigned argc = 2;
  Local<Value> argv[argc] = { Integer::New( isolate,self->n_ ), Integer::New( isolate,self->m_ ) };
  Local<Function> cons = Local<Function>::New(isolate, constructor);
  Local<Object> instance = cons->NewInstance(context, argc, argv).ToLocalChecked() ;

  scope.Escape( instance );
  WrappedArray* result = node::ObjectWrap::Unwrap<WrappedArray>( instance ) ;

  if( !result->isVector ) {
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

  args.GetReturnValue().Set( instance );
}


void WrappedArray::Muli( const v8::FunctionCallbackInfo<v8::Value>& args )
{
  WrappedArray* self = ObjectWrap::Unwrap<WrappedArray>(args.Holder());

  args.GetReturnValue().Set( args.Holder() ) ;

  float f = args[0]->IsUndefined() ? 1.0 : args[0]->NumberValue() ;
  int sz = self->m_ * self->n_ ;

  for( int i=0 ; i<sz ; i++ ) {
    self->data_[i] *= f ;
  }
}




void WrappedArray::Mul( const v8::FunctionCallbackInfo<v8::Value>& args )
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

  float f = args[0]->IsUndefined() ? 1.0 : args[0]->NumberValue() ;
  int sz = self->m_ * self->n_ ;

  for( int i=0 ; i<sz ; i++ ) {
    result->data_[i] = self->data_[i] * f ;
  }

}



void WrappedArray::Mmul( const v8::FunctionCallbackInfo<v8::Value>& args )
{
  Isolate* isolate = args.GetIsolate();
  Local<Context> context = isolate->GetCurrentContext() ;

  WrappedArray* self = ObjectWrap::Unwrap<WrappedArray>(args.Holder());
  WrappedArray* other = ObjectWrap::Unwrap<WrappedArray>( args[0]->ToObject() );

  if( self->n_ != other->m_ ) {
    char *msg = new char[ 1000  ];
    snprintf( msg, 1000, "Incompatible args: |%d x %d| x |%d x %d|", self->m_, self->n_, other->m_, other->n_ ) ;
    isolate->ThrowException(Exception::TypeError( String::NewFromUtf8(isolate, msg)));
    delete msg ;
    args.GetReturnValue().Set( Undefined(isolate) );
  }
  else {
    EscapableHandleScope scope(isolate) ; ;

    const unsigned argc = 2;
    Local<Value> argv[argc] = { Integer::New( isolate,self->m_ ), Integer::New( isolate,other->n_ ) };
    Local<Function> cons = Local<Function>::New(isolate, constructor);
    Local<Object> instance = cons->NewInstance(context, argc, argv).ToLocalChecked() ;

    scope.Escape( instance );

    WrappedArray* result = node::ObjectWrap::Unwrap<WrappedArray>( instance ) ;
    args.GetReturnValue().Set( instance );

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



struct Work {
  uv_work_t  request;
  Persistent<Function> callback;
  Persistent<Promise::Resolver> resolver ;
  WrappedArray* self ;
  WrappedArray* other;
  WrappedArray* result ;
  Persistent<Object> resultLocal;
};



void WrappedArray::Mmulp( const v8::FunctionCallbackInfo<v8::Value>& args )
{
  Isolate* isolate = args.GetIsolate();

  EscapableHandleScope scope(isolate) ;

  Local<Context> context = isolate->GetCurrentContext() ;

  WrappedArray *self = ObjectWrap::Unwrap<WrappedArray>(args.Holder());
  WrappedArray *other = ObjectWrap::Unwrap<WrappedArray>(args[0]->ToObject());

  const unsigned argc = 2;
  Local<Value> argv[argc] = { Integer::New( isolate,self->m_ ), Integer::New( isolate,other->n_ ) };
  Local<Function> cons = Local<Function>::New(isolate, constructor);
  Local<Object> instance = cons->NewInstance(context, argc, argv).ToLocalChecked() ;
  scope.Escape( instance );


  Work *work = new Work();
  work->request.data = work;

  work->self = self ;
  work->other = other ;
  work->result = ObjectWrap::Unwrap<WrappedArray>( instance )  ;
  work->resultLocal.Reset( isolate, instance ) ;

  if( args[1]->IsUndefined() ) {
    Local<Promise::Resolver> resolver = v8::Promise::Resolver::New( isolate ) ;
    work->resolver.Reset(isolate, resolver ) ;
    args.GetReturnValue().Set( resolver->GetPromise()  ) ;
  } else {
    Local<Function> callback = Local<Function>::Cast(args[1]);
    work->callback.Reset(isolate, callback ) ;
    args.GetReturnValue().Set( Undefined(isolate) ) ;
  }
  if( work->self->n_ != work->other->m_ ) {
    char *msg = new char[ 1000 ] ;
    snprintf( msg, 1000, "Incompatible args: |%d x %d| x |%d x %d|", self->m_, self->n_, other->m_, other->n_ ) ;
    Local<String> err = String::NewFromUtf8(isolate, msg);
    delete msg ;
    if( !work->resolver.IsEmpty() ) {
      Local<Promise::Resolver> resolver = Local<Promise::Resolver>::New(isolate,work->resolver) ;
      resolver->Reject( context, err ) ;
      work->resolver.Reset();   
    }
    if( !work->callback.IsEmpty() ) {
      Handle<Value> argv[] = { err, Null(isolate) };
      Local<Function>::New(isolate, work->callback)-> Call(isolate->GetCurrentContext()->Global(), 2, argv);
      work->callback.Reset();    
    }
    work->resultLocal.Reset(); 
  } else {
    uv_queue_work(uv_default_loop(),&work->request, WrappedArray::MmulpWorkAsync, WrappedArray::WorkAsyncComplete ) ;
  }
}


void WrappedArray::MmulpWorkAsync(uv_work_t *req) {
  Work *work = static_cast<Work *>(req->data);

  WrappedArray* self = work->self ;
  WrappedArray* other = work->other ;
  WrappedArray* result = work->result ;

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


void WrappedArray::WorkAsyncComplete(uv_work_t *req,int status)
{
    Isolate * isolate = Isolate::GetCurrent();
    HandleScope scope(isolate) ;

    Work *work = static_cast<Work *>(req->data);

    Local<Object> rc = Local<Object>::New(isolate,work->resultLocal) ;
    work->resultLocal.Reset();

    if( !work->callback.IsEmpty() ) {
      Handle<Value> argv[] = { Null(isolate), rc };
      Local<Function>::New(isolate, work->callback)-> Call(isolate->GetCurrentContext()->Global(), 2, argv);
      work->callback.Reset();    
    }
    if( !work->resolver.IsEmpty() ) {
      Local<Promise::Resolver> resolver = Local<Promise::Resolver>::New(isolate,work->resolver) ;
      resolver->Resolve( rc ) ;
      work->resolver.Reset();
    }

    delete work;
}


void WrappedArray::Asum( const v8::FunctionCallbackInfo<v8::Value>& args )
{
  WrappedArray* self = ObjectWrap::Unwrap<WrappedArray>(args.Holder());
  float rc = cblas_sasum(
    self->m_ * self->n_ ,
    self->data_,
    1 ) ;
  args.GetReturnValue().Set( rc );
}


void WrappedArray::Sum( const v8::FunctionCallbackInfo<v8::Value>& args )
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

void WrappedArray::GetCols( const v8::FunctionCallbackInfo<v8::Value>& args )
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

  // make a row vector: mx1
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

void WrappedArray::RemoveCol( const v8::FunctionCallbackInfo<v8::Value>& args )
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



void WrappedArray::Reshape( const v8::FunctionCallbackInfo<v8::Value>& args )
{
  Isolate* isolate = args.GetIsolate();
//  Local<Context> context = isolate->GetCurrentContext() ;

  WrappedArray* self = ObjectWrap::Unwrap<WrappedArray>(args.Holder());

  int m = args[0]->IsUndefined() ? 1 : args[0]->NumberValue() ;
  int n = args[1]->IsUndefined() ? 1 : args[1]->NumberValue() ;

  EscapableHandleScope scope(isolate) ; ;

  args.GetReturnValue().Set( args.Holder() );

  if( m*n > self->m_*self->n_ ) {
    float *tmp = self->data_ ;
    self->data_ = new float[m*n];
    memcpy( self->data_, tmp, self->m_*self->n_*sizeof(float) ) ; 
    delete tmp  ;
  }
  self->m_ = m ;
  self->n_ = n ;
}


void WrappedArray::Add( const v8::FunctionCallbackInfo<v8::Value>& args )
{
  Isolate* isolate = args.GetIsolate();
  Local<Context> context = isolate->GetCurrentContext() ;

  WrappedArray* self = ObjectWrap::Unwrap<WrappedArray>(args.Holder());
  WrappedArray* other = ObjectWrap::Unwrap<WrappedArray>( args[0]->ToObject() );

  if( self->n_ == other->n_  &&  self->m_ == other->m_ ) {
    EscapableHandleScope scope(isolate) ; ;

    const unsigned argc = 2;
    Local<Value> argv[argc] = { Integer::New( isolate,self->m_ ), Integer::New( isolate,self->n_ ) };
    Local<Function> cons = Local<Function>::New(isolate, constructor);
    Local<Object> instance = cons->NewInstance(context, argc, argv).ToLocalChecked() ;

    scope.Escape(instance);
    WrappedArray* result = node::ObjectWrap::Unwrap<WrappedArray>( instance ) ;
    args.GetReturnValue().Set( instance );

    int sz = self->m_ * self->n_ ;
    float *data = result->data_ ;
    float *a = self->data_ ;
    float *b = other->data_ ;
    for( int i=0 ; i<sz ; i++ ) {
      data[i] = a[i] + b[i] ;
    }
  }                              // add a row vector to each row
  else if( self->n_ == other->n_  &&  other->m_ == 1 ) {
    EscapableHandleScope scope(isolate) ; ;

    const unsigned argc = 2;
    Local<Value> argv[argc] = { Integer::New( isolate,self->m_ ), Integer::New( isolate,self->n_ ) };
    Local<Function> cons = Local<Function>::New(isolate, constructor);
    Local<Object> instance = cons->NewInstance(context, argc, argv).ToLocalChecked() ;

    scope.Escape(instance);
    WrappedArray* result = node::ObjectWrap::Unwrap<WrappedArray>( instance ) ;
    args.GetReturnValue().Set( instance );

    int sz = self->m_ * self->n_ ;
    float *data = result->data_ ;
    float *a = self->data_ ;
    float *b = other->data_ ;
    int j = 0 ;
    int n = self->m_ ;
    for( int i=0 ; i<sz ; i++ ) {
      data[i] = a[i] + b[j] ;
      if( --n == 0 ) { j++ ; n = self->m_ ; }
    }
  }                              // add a col vector to each col
  else if( self->m_ == other->m_  &&  other->n_ == 1 ) {
    EscapableHandleScope scope(isolate) ; ;

    const unsigned argc = 2;
    Local<Value> argv[argc] = { Integer::New( isolate,self->m_ ), Integer::New( isolate,self->n_ ) };
    Local<Function> cons = Local<Function>::New(isolate, constructor);
    Local<Object> instance = cons->NewInstance(context, argc, argv).ToLocalChecked() ;

    scope.Escape(instance);
    WrappedArray* result = node::ObjectWrap::Unwrap<WrappedArray>( instance ) ;
    args.GetReturnValue().Set( instance );

    int sz = self->m_ * self->n_ ;
    float *data = result->data_ ;
    float *a = self->data_ ;
    float *b = other->data_ ;
    int j = 0 ;
    for( int i=0 ; i<sz ; i++ ) {
      data[i] = a[i] + b[j++] ;
      if( j>=self->m_ ) j=0 ;
    }
  }                              // incompatible types ...
  else {
    char *msg = new char[ 1000 ] ;
    snprintf( msg, 1000, "Incompatible args: |%d x %d| + |%d x %d|", self->m_, self->n_, other->m_, other->n_ ) ;
    isolate->ThrowException(Exception::TypeError( String::NewFromUtf8(isolate, msg)));
    delete msg ;
    args.GetReturnValue().Set( Undefined(isolate) );
  }

}


void WrappedArray::Addi( const v8::FunctionCallbackInfo<v8::Value>& args )
{

  WrappedArray* self = ObjectWrap::Unwrap<WrappedArray>(args.Holder());
  WrappedArray* other = ObjectWrap::Unwrap<WrappedArray>( args[0]->ToObject() );

                                 // this always returns self
  args.GetReturnValue().Set( args.Holder() ) ;
  if( self->n_ == other->n_  &&  self->m_ == other->m_ ) {
    int sz = self->m_ * self->n_ ;
    float *data = self->data_ ;
    float *a = self->data_ ;
    float *b = other->data_ ;
    for( int i=0 ; i<sz ; i++ ) {
      data[i] = a[i] + b[i] ;
    }
  }                              // add a row vector to each row
  else if( self->n_ == other->n_  &&  other->m_ == 1 ) {
    int sz = self->m_ * self->n_ ;
    float *data = self->data_ ;
    float *a = self->data_ ;
    float *b = other->data_ ;
    int j = 0 ;
    int n = self->m_ ;
    for( int i=0 ; i<sz ; i++ ) {
      data[i] = a[i] + b[j] ;
      if( --n == 0 ) { j++ ; n = self->m_ ; }
    }
  }                              // add a col vector to each col
  else if( self->m_ == other->m_  &&  other->n_ == 1 ) {
    int sz = self->m_ * self->n_ ;
    float *data = self->data_ ;
    float *a = self->data_ ;
    float *b = other->data_ ;
    int j = 0 ;
    for( int i=0 ; i<sz ; i++ ) {
      data[i] = a[i] + b[j++] ;
      if( j>=self->m_ ) j=0 ;
    }
  }                              // incompatible types ...
  else {
    Isolate* isolate = args.GetIsolate();
    char *msg = new char[ 1000  ];
    snprintf( msg, 1000, "Incompatible args: |%d x %d| + |%d x %d|", self->m_, self->n_, other->m_, other->n_ ) ;
    isolate->ThrowException(Exception::TypeError( String::NewFromUtf8(isolate, msg)));
    delete msg ;
  }
}


void WrappedArray::Sub( const v8::FunctionCallbackInfo<v8::Value>& args )
{
  Isolate* isolate = args.GetIsolate();
  Local<Context> context = isolate->GetCurrentContext() ;

  WrappedArray* self = ObjectWrap::Unwrap<WrappedArray>(args.Holder());
  WrappedArray* other = ObjectWrap::Unwrap<WrappedArray>( args[0]->ToObject() );

  if( self->n_ == other->n_  &&  self->m_ == other->m_ ) {
    EscapableHandleScope scope(isolate) ; ;

    const unsigned argc = 2;
    Local<Value> argv[argc] = { Integer::New( isolate,self->m_ ), Integer::New( isolate,self->n_ ) };
    Local<Function> cons = Local<Function>::New(isolate, constructor);
    Local<Object> instance = cons->NewInstance(context, argc, argv).ToLocalChecked() ;

    scope.Escape(instance);
    WrappedArray* result = node::ObjectWrap::Unwrap<WrappedArray>( instance ) ;
    args.GetReturnValue().Set( instance );

    int sz = self->m_ * self->n_ ;
    float *data = result->data_ ;
    float *a = self->data_ ;
    float *b = other->data_ ;
    for( int i=0 ; i<sz ; i++ ) {
      data[i] = a[i] - b[i] ;
    }
  }                              // add a row vector to each row
  else if( self->n_ == other->n_  &&  other->m_ == 1 ) {
    EscapableHandleScope scope(isolate) ; ;

    const unsigned argc = 2;
    Local<Value> argv[argc] = { Integer::New( isolate,self->m_ ), Integer::New( isolate,self->n_ ) };
    Local<Function> cons = Local<Function>::New(isolate, constructor);
    Local<Object> instance = cons->NewInstance(context, argc, argv).ToLocalChecked() ;

    scope.Escape(instance);
    WrappedArray* result = node::ObjectWrap::Unwrap<WrappedArray>( instance ) ;
    args.GetReturnValue().Set( instance );

    int sz = self->m_ * self->n_ ;
    float *data = result->data_ ;
    float *a = self->data_ ;
    float *b = other->data_ ;
    int j = 0 ;
    int n = self->m_ ;
    for( int i=0 ; i<sz ; i++ ) {
      data[i] = a[i] - b[j] ;
      if( --n == 0 ) { j++ ; n = self->m_ ; }
    }
  }                              // add a col vector to each col
  else if( self->m_ == other->m_  &&  other->n_ == 1 ) {
    EscapableHandleScope scope(isolate) ; ;

    const unsigned argc = 2;
    Local<Value> argv[argc] = { Integer::New( isolate,self->m_ ), Integer::New( isolate,self->n_ ) };
    Local<Function> cons = Local<Function>::New(isolate, constructor);
    Local<Object> instance = cons->NewInstance(context, argc, argv).ToLocalChecked() ;

    scope.Escape(instance);
    WrappedArray* result = node::ObjectWrap::Unwrap<WrappedArray>( instance ) ;
    args.GetReturnValue().Set( instance );

    int sz = self->m_ * self->n_ ;
    float *data = result->data_ ;
    float *a = self->data_ ;
    float *b = other->data_ ;
    int j = 0 ;
    for( int i=0 ; i<sz ; i++ ) {
      data[i] = a[i] - b[j++] ;
      if( j>=self->m_ ) j=0 ;
    }
  }                              // incompatible types ...
  else {
    char *msg = new char[ 1000 ] ;
    snprintf( msg, 1000, "Incompatible args: |%d x %d| + |%d x %d|", self->m_, self->n_, other->m_, other->n_ ) ;
    isolate->ThrowException(Exception::TypeError( String::NewFromUtf8(isolate, msg)));
    delete msg ;
    args.GetReturnValue().Set( Undefined(isolate) );
  }

}


void WrappedArray::Subi( const v8::FunctionCallbackInfo<v8::Value>& args )
{

  WrappedArray* self = ObjectWrap::Unwrap<WrappedArray>(args.Holder());
  WrappedArray* other = ObjectWrap::Unwrap<WrappedArray>( args[0]->ToObject() );

                                 // this always returns self
  args.GetReturnValue().Set( args.Holder() ) ;
  if( self->n_ == other->n_  &&  self->m_ == other->m_ ) {
    int sz = self->m_ * self->n_ ;
    float *data = self->data_ ;
    float *a = self->data_ ;
    float *b = other->data_ ;
    for( int i=0 ; i<sz ; i++ ) {
      data[i] = a[i] - b[i] ;
    }
  }                              // add a row vector to each row
  else if( self->n_ == other->n_  &&  other->m_ == 1 ) {
    int sz = self->m_ * self->n_ ;
    float *data = self->data_ ;
    float *a = self->data_ ;
    float *b = other->data_ ;
    int j = 0 ;
    int n = self->m_ ;
    for( int i=0 ; i<sz ; i++ ) {
      data[i] = a[i] - b[j] ;
      if( --n == 0 ) { j++ ; n = self->m_ ; }
    }
  }                              // add a col vector to each col
  else if( self->m_ == other->m_  &&  other->n_ == 1 ) {
    int sz = self->m_ * self->n_ ;
    float *data = self->data_ ;
    float *a = self->data_ ;
    float *b = other->data_ ;
    int j = 0 ;
    for( int i=0 ; i<sz ; i++ ) {
      data[i] = a[i] - b[j++] ;
      if( j>=self->m_ ) j=0 ;
    }
  }                              // incompatible types ...
  else {
    Isolate* isolate = args.GetIsolate();
    char *msg = new char[ 1000 ] ;
    snprintf( msg, 1000, "Incompatible args: |%d x %d| + |%d x %d|", self->m_, self->n_, other->m_, other->n_ ) ;
    isolate->ThrowException(Exception::TypeError( String::NewFromUtf8(isolate, msg)));
    delete msg  ;
  }
}


void WrappedArray::Inv( const v8::FunctionCallbackInfo<v8::Value>& args )
{
  Isolate* isolate = args.GetIsolate();
  Local<Context> context = isolate->GetCurrentContext() ;

  WrappedArray* self = ObjectWrap::Unwrap<WrappedArray>(args.Holder());

  if( self->n_ != self->m_ ) {
    char *msg = new char[ 1000 ] ;
    snprintf( msg, 1000, "Incompatible args - inv() requires a square matrix not |%d x %d|", self->m_, self->n_ ) ;
    isolate->ThrowException(Exception::TypeError( String::NewFromUtf8(isolate, msg)));
    delete msg ;
    args.GetReturnValue().Set( Undefined(isolate) );
  }
  else {
    EscapableHandleScope scope(isolate) ;

    const unsigned argc = 2;
    Local<Value> argv[argc] = { Integer::New( isolate,self->m_ ), Integer::New( isolate,self->n_ ) };
    Local<Function> cons = Local<Function>::New(isolate, constructor);
    Local<Object> instance = cons->NewInstance(context, argc, argv).ToLocalChecked() ;

    scope.Escape( instance );

    WrappedArray* result = node::ObjectWrap::Unwrap<WrappedArray>( instance ) ;
    int sz = result->m_ * result->n_ ;
    for( int i=0 ; i<sz ; i++ ) {
      result->data_[i] = self->data_[i] ;
    }

    args.GetReturnValue().Set( instance );

    int *ipiv = new int[ std::min( self->m_, self->n_) ] ;
    int rc = LAPACKE_sgetrf(
      CblasColMajor,
      result->m_,
      result->n_,
      result->data_,
      result->m_,
      ipiv ) ;
    if( rc != 0 ) {
      char *msg = new char[ 1000 ] ;
      snprintf( msg, 1000, "Internal failure - sgetrf() failed with %d", rc ) ;
      isolate->ThrowException(Exception::TypeError( String::NewFromUtf8(isolate, msg)));
      delete msg  ;
      args.GetReturnValue().Set( Undefined(isolate) );
    }
    else {
      rc = LAPACKE_sgetri(
        CblasColMajor,
        result->n_,
        result->data_,
        result->n_,
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
  }
}


void WrappedArray::Invi( const v8::FunctionCallbackInfo<v8::Value>& args )
{
  Isolate* isolate = args.GetIsolate();

  WrappedArray* self = ObjectWrap::Unwrap<WrappedArray>(args.Holder());

  if( self->n_ != self->m_ ) {
    char *msg = new char[ 1000 ] ;
    snprintf( msg, 1000, "Incompatible args - inv() requires a square matrix not |%d x %d|", self->m_, self->n_ ) ;
    isolate->ThrowException(Exception::TypeError( String::NewFromUtf8(isolate, msg)));
    delete msg ;
    args.GetReturnValue().Set( Undefined(isolate) );
  }
  else {
    args.GetReturnValue().Set( args.Holder() );

    int *ipiv = new int[ std::min( self->m_, self->n_) ] ;
    int rc = LAPACKE_sgetrf(
      CblasColMajor,
      self->m_,
      self->n_,
      self->data_,
      self->m_,
      ipiv ) ;
    if( rc < 0 ) {
      char *msg = new char[ 1000 ] ;
      snprintf( msg, 1000, "Internal failure - sgetrf() failed with %d", rc ) ;
      isolate->ThrowException(Exception::TypeError( String::NewFromUtf8(isolate, msg)));
      delete msg ;
      args.GetReturnValue().Set( Undefined(isolate) );
    }
    else {
      rc = LAPACKE_sgetri(
        CblasColMajor,
        self->n_,
        self->data_,
        self->n_,
        ipiv ) ;

      if( rc < 0 ) {
        char *msg = new char[ 1000 ] ;
        snprintf( msg, 1000, "Internal failure - sgetri() failed with %d", rc ) ;
        isolate->ThrowException(Exception::TypeError( String::NewFromUtf8(isolate, msg)));
        delete msg ;
        args.GetReturnValue().Set( Undefined(isolate) );
      }
    }
    delete ipiv ;
  }
}


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

  float *superb = new float[ self->m_ * self->n_ ] ;
  int rc = LAPACKE_sgesvd(
    CblasColMajor,
    'A', 'A',
    self->m_,
    self->n_,
    self->data_,
    self->m_,
    s->data_,
    u->data_,
    self->m_,
    vt->data_,
    self->n_,
    superb ) ;

  delete superb ;

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


void WrappedArray::Pca( const v8::FunctionCallbackInfo<v8::Value>& args )
{
  Isolate* isolate = args.GetIsolate();
  Local<Context> context = isolate->GetCurrentContext() ;

  float variance = args[0]->IsUndefined() ? 0.98f : args[0]->NumberValue();

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


void WrappedArray::Inspect( const v8::FunctionCallbackInfo<v8::Value>& args )
{
  ToString( args ) ;
}


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





void WrappedArray::Ones( const v8::FunctionCallbackInfo<v8::Value>& args )
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
    self->data_[i] = ( rand() % 10 ) - 5 ;
  }
  args.GetReturnValue().Set( instance );
}


void WrappedArray::DataCallback(const FunctionCallbackInfo<Value>& args) {

  Isolate* isolate = args.GetIsolate();
  Local<Context> context = isolate->GetCurrentContext() ;
  Local<Object> xtra = Local<Object>::Cast( args.Data() ) ;
  Local<Value> array = xtra->Get( context, String::NewFromUtf8(isolate, "array") ).ToLocalChecked() ;
  Local<Object> obj = Local<Object>::Cast( array ) ;
  WrappedArray* self = ObjectWrap::Unwrap<WrappedArray>( obj );

  if( args[0]->IsArray() ) {
    Local<Array> cols = Local<Array>::Cast(args[0]->ToObject() );
    if( self->m_ == 0 ) self->m_ = cols->Length() ; // first time count cols in data to be row count
    int numColsToRead = ::min( (uint32_t)self->m_, cols->Length() ) ;

    if( self->dataSize_ <= (self->n_*self->m_) ) {
      float *tmp = self->data_ ;
      size_t oldDataSize = self->dataSize_ ;
      self->dataSize_ = ::max( (self->n_*2), self->n_+16 )*self->m_  ;
      self->data_ = new float[ self->dataSize_ ] ; 
      memcpy( self->data_, tmp, oldDataSize*sizeof(float) ) ;
      delete tmp ;
    }

    int p = (self->n_*self->m_) ;
    for( int i=0 ; i<numColsToRead ; i++ ) {
	self->data_[p] = cols->Get( context, i ).ToLocalChecked()->NumberValue() ;
        p++ ;
    }
  }

  self->n_++ ;
}

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

  int tmp = self->m_ ;
  self->m_ = self->n_ ;
  self->n_ = tmp ;

  Local<Value> promise = xtra->Get( context, String::NewFromUtf8(isolate, "promise") ).ToLocalChecked() ;
  Local<Promise::Resolver> resolver = Local<Promise::Resolver>::Cast( promise ) ;

  resolver->Resolve( obj ) ;

}



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


void WrappedArray::GetCoeff(Local<String> property, const PropertyCallbackInfo<Value>& info)
{
  Isolate* isolate = info.GetIsolate();
  WrappedArray* obj = ObjectWrap::Unwrap<WrappedArray>(info.This());

  v8::String::Utf8Value s(property);
  std::string str(*s);

  if ( str == "m") {
    info.GetReturnValue().Set(Number::New(isolate, obj->m_));
  }
  else if (str == "n") {
    info.GetReturnValue().Set(Number::New(isolate, obj->n_));
  }
  else if (str == "length") {
    info.GetReturnValue().Set(Number::New(isolate, obj->n_*obj->m_ ));
  }
}


void WrappedArray::SetCoeff(Local<String> property, Local<Value> value, const PropertyCallbackInfo<void>& info)
{
  WrappedArray* obj = ObjectWrap::Unwrap<WrappedArray>(info.This());

  v8::String::Utf8Value s(property);
  std::string str(*s);

  if ( str == "m") {
    obj->m_ = value->NumberValue();
  }
  else if (str == "n") {
    obj->n_ = value->NumberValue();
  }
}


void CreateObject(const FunctionCallbackInfo<Value>& info)
{
  Isolate* isolate = info.GetIsolate();
  HandleScope scope(isolate);
  info.GetReturnValue().Set(WrappedArray::NewInstance(info) );
}


void InitArray(Local<Object> exports, Local<Object> module)
{
  WrappedArray::Init(exports, module);
}


NODE_MODULE(linalg, InitArray)
