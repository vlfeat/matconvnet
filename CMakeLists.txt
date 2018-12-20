cmake_minimum_required (VERSION 3.5)
project (matconvnet)

set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
      STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

find_package(Matlab COMPONENTS MX_LIBRARY REQUIRED)
find_package(JPEG REQUIRED)
find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

add_definitions(-DENABLE_DOUBLE)
include_directories(${JPEG_INCLUDE_DIR})
include_directories(${Matlab_INCLUDE_DIRS})

if(MSVC)
  set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
  add_definitions(-D_CRT_SECURE_NO_DEPRECATE)
  if(CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
    string(REGEX REPLACE "/W[0-4]" "/W1" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  endif()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4068")
endif()

# Snippet to locate the matlab-blas library to link
if(NOT Matlab_BLAS_LIBRARY)
  include(${CMAKE_ROOT}/Modules/SelectLibraryConfigurations.cmake)
  find_library(Matlab_BLAS_LIBRARY_RELEASE NAMES mwblas  libmwblas  PATHS ${Matlab_ROOT_DIR} PATH_SUFFIXES extern/lib/win64/microsoft bin/maci64 bin/glnxa64)
  find_library(Matlab_BLAS_LIBRARY_DEBUG   NAMES mwblasd libmwblasd PATHS ${Matlab_ROOT_DIR} PATH_SUFFIXES extern/lib/win64/microsoft bin/maci64 bin/glnxa64)
  select_library_configurations(Matlab_BLAS)
endif()

add_library(matconvnet SHARED
 matlab/src/bits/data.cpp
 matlab/src/bits/datamex.cpp
 matlab/src/bits/nnconv.cpp
 matlab/src/bits/nnbias.cpp
 matlab/src/bits/nnfullyconnected.cpp
 matlab/src/bits/nnsubsample.cpp
 matlab/src/bits/nnpooling.cpp
 matlab/src/bits/nnnormalize.cpp
 matlab/src/bits/nnnormalizelp.cpp
 matlab/src/bits/nnbnorm.cpp
 matlab/src/bits/nnbilinearsampler.cpp
 matlab/src/bits/nnroipooling.cpp
 matlab/src/bits/impl/im2row_cpu.cpp
 matlab/src/bits/impl/copy_cpu.cpp
 matlab/src/bits/impl/tinythread.cpp
 matlab/src/bits/impl/imread_libjpeg.cpp
 matlab/src/bits/imread.cpp
)
set_property(TARGET matconvnet PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(matconvnet ${Matlab_BLAS_LIBRARY} ${JPEG_LIBRARY} ${Matlab_MEX_LIBRARY} ${Matlab_MX_LIBRARY})
install(TARGETS matconvnet DESTINATION ${CMAKE_SOURCE_DIR}/matlab/)

if(NOT MSVC)
  set(VL_TMOVE "vl_tmove" "matlab/src/vl_tmove.cpp")
endif()

list(APPEND vl_files
vl_imreadjpeg        matlab/src/vl_imreadjpeg.cpp
vl_imreadjpeg_old    matlab/src/vl_imreadjpeg_old.cpp
vl_nnbilinearsampler matlab/src/vl_nnbilinearsampler.cpp
vl_nnbnorm           matlab/src/vl_nnbnorm.cpp
vl_nnconv            matlab/src/vl_nnconv.cpp
vl_nnconvt           matlab/src/vl_nnconvt.cpp
vl_nnnormalize       matlab/src/vl_nnnormalize.cpp
vl_nnnormalizelp     matlab/src/vl_nnnormalizelp.cpp
vl_nnpool            matlab/src/vl_nnpool.cpp
vl_nnroipool         matlab/src/vl_nnroipool.cpp
vl_taccummex         matlab/src/vl_taccummex.cpp
${VL_TMOVE}
)

list(LENGTH vl_files size)
math(EXPR size "${size} - 1")
foreach(val RANGE 0 ${size} 2)
  math(EXPR nextval "${val} + 1")
  list(GET vl_files ${val} mex)
  list(GET vl_files ${nextval} src)
  matlab_add_mex(NAME ${mex} SRC ${src} LINK_TO ${JPEG_LIBRARY} matconvnet)
  install(TARGETS ${mex} DESTINATION ${CMAKE_SOURCE_DIR}/matlab/)
endforeach()
