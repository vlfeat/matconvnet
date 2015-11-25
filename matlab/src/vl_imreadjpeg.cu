/** @file vl_imreadjpeg.cu
 ** @brief Load images asynchronously
 ** @author Andrea Vedaldi
 **/

/*
Copyright (C) 2014-15 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "bits/impl/tinythread.h"
#include "bits/imread.hpp"

#include <vector>
#include <string>
#include <algorithm>

#include "bits/mexutils.h"

/* option codes */
enum {
  opt_num_threads = 0,
  opt_prefetch,
  opt_preallocate,
  opt_verbose,
} ;

/* options */
vlmxOption  options [] = {
  {"NumThreads",       1,   opt_num_threads        },
  {"Prefetch",         0,   opt_prefetch           },
  {"Preallocate",      1,   opt_preallocate        },
  {"Verbose",          0,   opt_verbose            },
  {0,                  0,   0                      }
} ;

enum {
  IN_FILENAMES = 0, IN_END
} ;

enum {
  OUT_IMAGES = 0, OUT_END
} ;

/* ---------------------------------------------------------------- */
/*                                                           Caches */
/* ---------------------------------------------------------------- */

struct task_t {
  std::string name ;
  bool done ;
  bool hasMatlabMemory ;
  vl::Image image ;
} ;

typedef std::vector<task_t> tasks_t ;
tasks_t tasks ;
tthread::mutex tasksMutex ;
tthread::condition_variable tasksCondition ;
tthread::condition_variable completedCondition ;
int nextTaskIndex = 0 ;
int numTasksCompleted = 0 ;

typedef std::pair<tthread::thread*,vl::ImageReader*> reader_t ;
typedef std::vector<reader_t> readers_t ;
readers_t readers ;
bool terminateReaders = true ;

/* ---------------------------------------------------------------- */
/*                                                Tasks and readers */
/* ---------------------------------------------------------------- */

void reader_function(void* reader_)
{
  vl::ImageReader* reader = (vl::ImageReader*) reader_ ;
  int taskIndex ;

  tasksMutex.lock() ;
  while (true) {
    // wait for next task
    while ((nextTaskIndex >= tasks.size()) && ! terminateReaders) {
      tasksCondition.wait(tasksMutex);
    }
    if (terminateReaders) {
      break ;
    }
    taskIndex = nextTaskIndex++ ;
    task_t & thisTask = tasks[taskIndex] ;

    tasksMutex.unlock() ;
    thisTask.image = reader->read(thisTask.name.c_str(), thisTask.image.memory) ;

    tasksMutex.lock() ;
    thisTask.done = true ;
    numTasksCompleted ++ ;
    completedCondition.notify_all() ;
  }
  tasksMutex.unlock() ;
}

void delete_readers()
{
  tasksMutex.lock() ;
  terminateReaders = true ;
  tasksMutex.unlock() ;
  tasksCondition.notify_all() ;
  for (int r = 0 ; r < readers.size() ; ++r) {
    readers[r].first->join() ;
    delete readers[r].first ;
    delete readers[r].second ;
  }
  readers.clear() ;
}

void create_readers(int num, int verbosity)
{
  if (num <= 0) {
    num = (std::max)(1, (int)readers.size()) ;
  }
  if (readers.size() == num) {
    return ;
  }
  if (verbosity > 1) { mexPrintf("vl_imreadjpeg: flushing reader threads\n") ; }
  delete_readers() ;

  terminateReaders = false ;
  for (int r = 0 ; r < num ; ++r) {
    vl::ImageReader * reader = new vl::ImageReader() ;
    tthread::thread * readerThread = new tthread::thread(reader_function, reader) ;
    readers.push_back(reader_t(readerThread, reader)) ;
  }
  if (verbosity > 1) { mexPrintf("vl_imreadjpeg: created %d reader threads\n", readers.size()) ; }
}

void flush_tasks() {
  // wait until all tasks in the current list are complete
  tasksMutex.lock() ;
  while (numTasksCompleted < tasks.size()) {
    completedCondition.wait(tasksMutex);
  }

  // now delete them
  for (int t = 0 ; t < tasks.size() ; ++t) {
    if (tasks[t].image.memory) {
      if (tasks[t].hasMatlabMemory) {
        mxFree(tasks[t].image.memory) ;
      } else {
        free(tasks[t].image.memory) ;
      }
    }
  }
  tasks.clear() ;
  numTasksCompleted = 0 ;
  nextTaskIndex = 0 ;
  tasksMutex.unlock() ;
}

void atExit()
{
  delete_readers() ;
}

/* ---------------------------------------------------------------- */
/*                                                            Cache */
/* ---------------------------------------------------------------- */

void mexFunction(int nout, mxArray *out[],
                 int nin, mxArray const *in[])
{
  bool prefetch = false ;
  int requestedNumThreads = -1 ;
  int verbosity = 0 ;
  int opt ;
  int next = IN_END ;
  mxArray const *optarg ;
  int i ;
  bool preallocate = true ;

  /* -------------------------------------------------------------- */
  /*                                            Check the arguments */
  /* -------------------------------------------------------------- */

  mexAtExit(atExit) ;

  if (nin < 1) {
    mexErrMsgTxt("There are less than one argument.") ;
  }

  while ((opt = vlmxNextOption (in, nin, options, &next, &optarg)) >= 0) {
    switch (opt) {
      case opt_verbose :
        ++ verbosity ;
        break ;

      case opt_prefetch :
        prefetch = true ;
        break ;

      case opt_preallocate :
        if (!mxIsLogicalScalar(optarg)) {
          mexErrMsgTxt("PREALLOCATE is not a logical scalar.") ;
        }
        preallocate = mxIsLogicalScalarTrue(optarg) ;
        break ;

      case opt_num_threads :
        requestedNumThreads = mxGetScalar(optarg) ;
        break ;
    }
  }

  if (!mxIsCell(in[IN_FILENAMES])) {
    mexErrMsgTxt("FILENAMES is not a cell array of strings.") ;
  }

  // prepare reader tasks
  create_readers(requestedNumThreads, verbosity) ;

  if (verbosity) {
    mexPrintf("vl_imreadjpeg: numThreads = %d, prefetch = %d, preallocate = %d\n",
              readers.size(), prefetch, preallocate) ;
  }

  // extract filenames as strings
  std::vector<std::string> filenames ;
  for (i = 0 ; i < mxGetNumberOfElements(in[IN_FILENAMES]) ; ++i) {
    mxArray* filename_array = mxGetCell(in[IN_FILENAMES], i) ;
    if (!vlmxIsString(filename_array,-1)) {
      mexErrMsgTxt("FILENAMES contains an entry that is not a string.") ;
    }
    char filename [4096] ;
    mxGetString (filename_array, filename, sizeof(filename)/sizeof(char)) ;
    filenames.push_back(std::string(filename)) ;
  }

  // check if the cached tasks match the new ones
  bool match = true ;
  for (int t = 0 ; match & (t < filenames.size()) ; ++t) {
    if (t >= tasks.size()) {
      match = false ;
      break ;
    }
    match &= (tasks[t].name == filenames[t]) ;
  }

  // if there is no match, then flush tasks and start over
  if (!match) {
    if (verbosity > 1) {
      mexPrintf("vl_imreadjpeg: flushing tasks\n") ;
    }
    flush_tasks() ;
    tasksMutex.lock() ;
    for (int t = 0 ; t < filenames.size() ; ++t) {
      task_t newTask ;
      newTask.name = filenames[t] ;
      newTask.done = false ;
      if (preallocate) {
        newTask.image = readers[0].second->readDimensions(filenames[t].c_str()) ;
        if (newTask.image.error == 0) {
          newTask.image.memory = (float*)mxMalloc(sizeof(float)*
                                                  newTask.image.width*
                                                  newTask.image.height*
                                                  newTask.image.depth) ;
          mexMakeMemoryPersistent(newTask.image.memory) ;
          newTask.hasMatlabMemory = true ;
        }
      } else {
        newTask.image = vl::Image() ;
        newTask.hasMatlabMemory = false ;
      }
      tasks.push_back(newTask) ;
    }
    tasksMutex.unlock() ;
    tasksCondition.notify_all() ;
  }

  // done if prefetching only
  if (prefetch) { return ; }

  // return
  out[OUT_IMAGES] = mxCreateCellArray(mxGetNumberOfDimensions(in[IN_FILENAMES]),
                                      mxGetDimensions(in[IN_FILENAMES])) ;
  for (int t = 0 ; t < tasks.size() ; ++t) {
    tasksMutex.lock() ;
    while (!tasks[t].done) {
      completedCondition.wait(tasksMutex);
    }
    vl::Image & image = tasks[t].image ;
    tasksMutex.unlock() ;

    if (!image.error) {
      mwSize dimensions [3] = {
        (mwSize)image.height,
        (mwSize)image.width,
        (mwSize)image.depth} ;
      mwSize dimensions_ [3] = {0} ;
      mxArray * image_array = mxCreateNumericArray(3, dimensions_, mxSINGLE_CLASS, mxREAL) ;
      mxSetDimensions(image_array, dimensions, 3) ;
      if (tasks[t].hasMatlabMemory) {
        mxSetData(image_array, image.memory) ;
        image.memory = NULL ;
      } else {
        float * matlabMemory = (float*)mxMalloc(dimensions[0]*dimensions[1]*dimensions[2]*sizeof(float)) ;
        mxSetData(image_array, matlabMemory) ;
        memcpy(matlabMemory,
               image.memory,
               image.height*image.width*image.depth*sizeof(float)) ;
      }
      mxSetCell(out[OUT_IMAGES], t, image_array) ;
    } else {
      char message [1024*4] ;
      snprintf(message, sizeof(message)/sizeof(char),
               "could not read image '%s'", tasks[t].name.c_str()) ;
      mexWarnMsgTxt(message) ;
    }
  }
  flush_tasks() ;
}
