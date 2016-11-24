// hello.cc
#include <node.h>
#include <uv.h>
#include <stdio.h>
#include <string.h>
#include <string>
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <thread>

namespace linalg {

using namespace v8;



struct Work {
  uv_work_t  request;
  Persistent<Function> callback;
  char *result ;
  int sleepTime ;
};

static void WorkAsync(uv_work_t *req)
{
    Work *work = static_cast<Work *>(req->data);

    work->result = (char*)malloc( 100 ) ;
    strncpy( work->result, "Boo Ya", 100 ) ;

    std::this_thread::sleep_for(std::chrono::seconds(work->sleepTime));
}



static void WorkAsyncComplete(uv_work_t *req,int status)
{
    Isolate * isolate = Isolate::GetCurrent();
    v8::HandleScope handleScope(isolate); // Required for Node 4.x

    Work *work = static_cast<Work *>(req->data);

    Local<Array> result_list = Array::New(isolate);
    Local<String> result = String::NewFromUtf8(isolate, work->result) ;

    result_list->Set(0, result);

    // set up return arguments
    Handle<Value> argv[] = { Null(isolate) , result_list };

    // execute the callback
    Local<Function>::New(isolate, work->callback)-> Call(isolate->GetCurrentContext()->Global(), 2, argv);

   // Free up the persistent function callback
    work->callback.Reset();

    free( work->result );
    delete work;
}


void Method(const FunctionCallbackInfo<Value>& args) {
  Isolate* isolate = args.GetIsolate();
  
  Work * work = new Work();
  work->request.data = work;
  work->sleepTime = 2 ;
  Local<Function> callback = Local<Function>::Cast(args[1]);
  work->callback.Reset(isolate, callback);


  // kick of the worker thread
  uv_queue_work(uv_default_loop(),&work->request, WorkAsync, WorkAsyncComplete );

  args.GetReturnValue().Set(Undefined(isolate));
}

void init(Local<Object> exports) {
  NODE_SET_METHOD(exports, "mmul", Method);
}

NODE_MODULE(addon, init)

}  // namespace demo

