/** @file    mexutils.h
 ** @brief   MEX utilities
 ** @author  Andrea Vedaldi
 **/

/*
Copyright (C) 2007-15 Andrea Vedaldi and Brian Fulkerson.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef MEXUTILS_H
#define MEXUTILS_H

#include"mex.h"
#include<ctype.h>
#include<string.h>
#include<stdio.h>
#include<stdarg.h>
#include<assert.h>

#ifdef _MSC_VER
#define snprintf _snprintf
#define vsnprintf _vsnprintf
#endif

typedef mwSize vl_size ;
typedef int unsigned  vl_uint  ;
typedef int vl_bool ;
typedef ptrdiff_t vl_index ;
typedef size_t vl_uindex ;
#define VL_INLINE static __inline
#define VL_TRUE 1
#define VL_FALSE 0

/** @brief Access MEX input argument */
#undef IN
#define IN(x) (in[IN_ ## x])

/** @brief Access MEX output argument */
#undef OUT
#define OUT(x) (out[OUT_ ## x])

#define MEXUTILS_RAISE_HELPER_A \
char const * errorString ; \
char formattedErrorId [512] ; \
char formattedErrorMessage [1024] ; \
\
switch (errorId) { \
case vlmxErrAlloc : errorString = "outOfMemory" ; break ; \
case vlmxErrInvalidArgument : errorString = "invalidArgument" ; break ; \
case vlmxErrNotEnoughInputArguments : errorString = "notEnoughInputArguments" ; break ; \
case vlmxErrTooManyInputArguments : errorString = "tooManyInputArguments" ; break ; \
case vlmxErrNotEnoughOutputArguments : errorString = "notEnoughOutputArguments" ; break ; \
case vlmxErrTooManyOutputArguments : errorString = "tooManyOutputArguments" ; break ; \
case vlmxErrInvalidOption : errorString = "invalidOption" ; break ; \
case vlmxErrInconsistentData : errorString = "inconsistentData" ; break ; \
case vlmxErrExecution: errorString = "execution" ; break ; \
default : errorString = "undefinedError" ; break ; \
} \
\
if (! errorMessage) { \
switch (errorId) { \
case vlmxErrAlloc: errorMessage = "Out of memory." ; break ; \
case vlmxErrInvalidArgument: errorMessage = "Invalid argument." ; break ; \
case vlmxErrNotEnoughInputArguments: errorMessage = "Not enough input arguments." ; break ; \
case vlmxErrTooManyInputArguments: errorMessage = "Too many input arguments." ; break ; \
case vlmxErrNotEnoughOutputArguments: errorMessage = "Not enough output arguments." ; break ; \
case vlmxErrTooManyOutputArguments: errorMessage = "Too many output arguments." ; break ; \
case vlmxErrInconsistentData: errorMessage = "Inconsistent data." ; break ; \
case vlmxErrInvalidOption: errorMessage = "Invalid option." ; break ; \
case vlmxErrExecution: errorMessage = "Execution error." ; break ; \
default: errorMessage = "Undefined error message." ; \
} \
}

#ifdef VL_COMPILER_LCC
#define MEXUTILS_RAISE_HELPER_B \
{ \
va_list args ; \
va_start(args, errorMessage) ; \
sprintf(formattedErrorId, \
"vl:%s", errorString) ; \
vsprintf(formattedErrorMessage, \
errorMessage, args) ; \
va_end(args) ; \
}
#else
#define MEXUTILS_RAISE_HELPER_B \
{ \
va_list args ; \
va_start(args, errorMessage) ; \
snprintf(formattedErrorId, \
sizeof(formattedErrorId)/sizeof(char), \
"vl:%s", errorString) ; \
vsnprintf(formattedErrorMessage, \
sizeof(formattedErrorMessage)/sizeof(char), \
errorMessage, args) ; \
va_end(args) ; \
}
#endif

#define MEXUTILS_RAISE_HELPER MEXUTILS_RAISE_HELPER_A MEXUTILS_RAISE_HELPER_B

/** @{
 ** @name Error handling
 **/

/** @brief VLFeat MEX errors */
typedef enum _VlmxErrorId {
  vlmxErrAlloc = 1,
  vlmxErrInvalidArgument,
  vlmxErrNotEnoughInputArguments,
  vlmxErrTooManyInputArguments,
  vlmxErrNotEnoughOutputArguments,
  vlmxErrTooManyOutputArguments,
  vlmxErrInvalidOption,
  vlmxErrInconsistentData,
  vlmxErrExecution
} VlmxErrorId ;


/** @brief Raise a MEX error with VLFeat format
 ** @param errorId error ID string.
 ** @param errorMessage error message C-style format string.
 ** @param ... format string arguments.
 **
 ** The function internally calls @c mxErrMsgTxtAndId, which causes
 ** the MEX file to abort.
 **/

#if defined(VL_COMPILER_GNUC) & ! defined(__DOXYGEN__)
static void __attribute__((noreturn))
#else
static void
#endif
vlmxError (VlmxErrorId errorId, char const * errorMessage, ...)
{
  MEXUTILS_RAISE_HELPER ;
  mexErrMsgIdAndTxt (formattedErrorId, formattedErrorMessage) ;
}

/** @brief Raise a MEX warning with VLFeat format
 ** @param errorId error ID string.
 ** @param errorMessage error message C-style format string.
 ** @param ... format string arguments.
 **
 ** The function internally calls @c mxWarnMsgTxtAndId.
 **/

static void
vlmxWarning (VlmxErrorId errorId, char const * errorMessage, ...)
{
  MEXUTILS_RAISE_HELPER ;
  mexWarnMsgIdAndTxt (formattedErrorId, formattedErrorMessage) ;
}

/** @} */

/** @name Check for array attributes
 ** @{ */

/** ------------------------------------------------------------------
 ** @brief Check if a MATLAB array is of a prescribed class
 ** @param array MATLAB array.
 ** @param classId prescribed class of the array.
 ** @return ::VL_TRUE if the class is of the array is of the prescribed class.
 ** @sa @ref mexutils-array-test
 **/

VL_INLINE vl_bool
vlmxIsOfClass (mxArray const * array, mxClassID classId)
{
  return mxGetClassID (array) == classId ;
}

/** ------------------------------------------------------------------
 ** @brief Check if a MATLAB array is real
 ** @param array MATLAB array.
 ** @return ::VL_TRUE if the array is real.
 ** @sa @ref mexutils-array-test
 **/

VL_INLINE vl_bool
vlmxIsReal (mxArray const * array)
{
  return mxIsNumeric (array) && ! mxIsComplex (array) ;
}

/** @} */

/** @name Check for scalar, vector and matrix arrays
 ** @{ */

/** ------------------------------------------------------------------
 ** @brief Check if a MATLAB array is scalar
 ** @param array MATLAB array.
 ** @return ::VL_TRUE if the array is scalar.
 ** @sa @ref mexutils-array-test
 **/

VL_INLINE vl_bool
vlmxIsScalar (mxArray const * array)
{
  return (! mxIsSparse (array)) && (mxGetNumberOfElements (array) == 1)  ;
}

/** ------------------------------------------------------------------
 ** @brief Check if a MATLAB array is a vector.
 ** @param array MATLAB array.
 ** @param numElements number of elements (negative for any).
 ** @return ::VL_TRUE if the array is a vecotr of the prescribed size.
 ** @sa @ref mexutils-array-test
 **/

static vl_bool
vlmxIsVector (mxArray const * array, vl_index numElements)
{
  vl_size numDimensions = (unsigned) mxGetNumberOfDimensions (array) ;
  mwSize const * dimensions = mxGetDimensions (array) ;
  vl_uindex di ;

  /* check that it is not sparse */
  if (mxIsSparse (array)) {
    return VL_FALSE ;
  }

  /* check that the number of elements is the prescribed one */
  if ((numElements >= 0) && ((unsigned) mxGetNumberOfElements (array) !=
                             (unsigned) numElements)) {
    return VL_FALSE ;
  }

  /* check that all but at most one dimension is singleton */
  for (di = 0 ;  di < numDimensions ; ++ di) {
    if (dimensions[di] != 1) break ;
  }
  for (++ di ; di < numDimensions ; ++di) {
    if (dimensions[di] != 1) return VL_FALSE ;
  }
  return VL_TRUE ;
}

/** ------------------------------------------------------------------
 ** @brief Check if a MATLAB array is a matrix.
 ** @param array MATLAB array.
 ** @param M number of rows (negative for any).
 ** @param N number of columns (negative for any).
 ** @return ::VL_TRUE if the array is a matrix of the prescribed size.
 ** @sa @ref mexutils-array-test
 **/

static vl_bool
vlmxIsMatrix (mxArray const * array, vl_index M, vl_index N)
{
  vl_size numDimensions = (unsigned) mxGetNumberOfDimensions (array) ;
  mwSize const * dimensions = mxGetDimensions (array) ;
  vl_uindex di ;

  /* check that it is not sparse */
  if (mxIsSparse (array)) {
    return VL_FALSE ;
  }

  /* check that the number of elements is the prescribed one */
  if ((M >= 0) && ((unsigned) mxGetM (array) != (unsigned) M)) {
    return VL_FALSE;
  }
  if ((N >= 0) && ((unsigned) mxGetN (array) != (unsigned) N)) {
    return VL_FALSE;
  }

  /* ok if empty and either M = 0 or N = 0 */
  if ((mxGetNumberOfElements (array) == 0) && (mxGetM (array) == 0 || mxGetN (array) == 0)) {
    return VL_TRUE ;
  }

  /* ok if any dimension beyond the first two is singleton */
  for (di = 2 ; ((unsigned)dimensions[di] == 1) && di < numDimensions ; ++ di) ;
  return di == numDimensions ;
}


/** ------------------------------------------------------------------
 ** @brief Check if the MATLAB array has the specified dimensions.
 ** @param array array to check.
 ** @param numDimensions number of dimensions.
 ** @param dimensions dimensions.
 ** @return true the test succeeds.
 **
 ** The test is true if @a numDimensions < 0. If not, it is false if
 ** the array has not @a numDimensions. Otherwise it is true is @a
 ** dimensions is @c NULL or if each entry of @a dimensions is
 ** either negative or equal to the corresponding array dimension.
 **/

static vl_bool
vlmxIsArray (mxArray const * array, vl_index numDimensions, vl_index* dimensions)
{
  if (numDimensions >= 0) {
    vl_index d ;
    mwSize const * actualDimensions = mxGetDimensions (array) ;

    if ((unsigned) mxGetNumberOfDimensions (array) != (unsigned) numDimensions) {
      return VL_FALSE ;
    }

    if(dimensions != NULL) {
      for(d = 0 ; d < numDimensions ; ++d) {
        if (dimensions[d] >= 0 && (unsigned) dimensions[d] != (unsigned) actualDimensions[d])
          return VL_FALSE ;
      }
    }
  }
  return VL_TRUE ;
}

/** @} */

/** @name Check for plain arrays
 ** @{ */

/** ------------------------------------------------------------------
 ** @brief Check if a MATLAB array is plain
 ** @param array MATLAB array.
 ** @return ::VL_TRUE if the array is plain.
 ** @sa @ref mexutils-array-test
 **/

VL_INLINE vl_bool
vlmxIsPlain (mxArray const * array)
{
  return
  vlmxIsReal (array) &&
  vlmxIsOfClass (array, mxDOUBLE_CLASS) ;
}


/** ------------------------------------------------------------------
 ** @brief Check if a MATLAB array is plain scalar
 ** @param array MATLAB array.
 ** @return ::VL_TRUE if the array is plain scalar.
 ** @sa @ref mexutils-array-test
 **/

VL_INLINE vl_bool
vlmxIsPlainScalar (mxArray const * array)
{
  return vlmxIsPlain (array) && vlmxIsScalar (array) ;
}

/** ------------------------------------------------------------------
 ** @brief Check if a MATLAB array is a plain vector.
 ** @param array MATLAB array.
 ** @param numElements number of elements (negative for any).
 ** @return ::VL_TRUE if the array is a plain vecotr of the prescribed size.
 ** @sa @ref mexutils-array-test
 **/

VL_INLINE vl_bool
vlmxIsPlainVector (mxArray const * array, vl_index numElements)
{
  return vlmxIsPlain (array) && vlmxIsVector (array, numElements) ;
}


/** ------------------------------------------------------------------
 ** @brief Check if a MATLAB array is a plain matrix.
 ** @param array MATLAB array.
 ** @param M number of rows (negative for any).
 ** @param N number of columns (negative for any).
 ** @return ::VL_TRUE if the array is a plain matrix of the prescribed size.
 ** @sa @ref mexutils-array-test
 **/

VL_INLINE vl_bool
vlmxIsPlainMatrix (mxArray const * array, vl_index M, vl_index N)
{
  return vlmxIsPlain (array) && vlmxIsMatrix (array, M, N) ;
}

/** ------------------------------------------------------------------
 ** @brief Check if the array is a string
 ** @param array array to test.
 ** @param length string length.
 ** @return true if the array is a string of the specified length
 **
 ** The array @a array satisfies the test if:
 ** - its storage class is CHAR;
 ** - it has two dimensions but only one row;
 ** - @a length < 0 or the array has @a length columns.
 **/

static int
vlmxIsString (const mxArray* array, vl_index length)
{
  mwSize M = (mwSize) mxGetM (array) ;
  mwSize N = (mwSize) mxGetN (array) ;

  return
  mxIsChar(array) &&
  mxGetNumberOfDimensions(array) == 2 &&
  (M == 1 || (M == 0 && N == 0)) &&
  (length < 0 || (signed)N == length) ;
}


/** @} */

/** ------------------------------------------------------------------
 ** @brief Create a MATLAB array which is a plain scalar
 ** @param x scalar value.
 ** @return the new array.
 **/

static mxArray *
vlmxCreatePlainScalar (double x)
{
  mxArray * array = mxCreateDoubleMatrix (1,1,mxREAL) ;
  *mxGetPr(array) = x ;
  return array ;
}

/** ------------------------------------------------------------------
 ** @brief Case insensitive string comparison
 ** @param s1 first string.
 ** @param s2 second string.
 ** @return comparison result.
 **
 ** The comparison result is equal to 0 if the strings are equal, >0
 ** if the first string is greater than the second (in lexicographical
 ** order), and <0 otherwise.
 **/

static int
vlmxCompareStringsI(const char *s1, const char *s2)
{
  /*
   Since tolower has an int argument, characters must be unsigned
   otherwise will be sign-extended when converted to int.
   */
  while (tolower((unsigned char)*s1) == tolower((unsigned char)*s2))
  {
    if (*s1 == 0) return 0 ; /* implies *s2 == 0 */
    s1++;
    s2++;
  }
  return tolower((unsigned char)*s1) - tolower((unsigned char)*s2) ;
}

/** ------------------------------------------------------------------
 ** @brief Case insensitive string comparison with array
 ** @param array first string (as a MATLAB array).
 ** @param string second string.
 ** @return comparison result.
 **
 ** The comparison result is equal to 0 if the strings are equal, >0
 ** if the first string is greater than the second (in lexicographical
 ** order), and <0 otherwise.
 **/

static int
vlmxCompareToStringI(mxArray const * array, char const  * string)
{
  mxChar const * s1 = (mxChar const *) mxGetData(array) ;
  char unsigned const * s2 = (char unsigned const*) string ;
  vl_size n = mxGetNumberOfElements(array) ;

  /*
   Since tolower has an int argument, characters must be unsigned
   otherwise will be sign-extended when converted to int.
   */
  while (n && tolower((unsigned)*s1) == tolower(*s2)) {
    if (*s2 == 0) return 1 ; /* s2 terminated on 0, but s1 did not terminate yet */
    s1 ++ ;
    s2 ++ ;
    n -- ;
  }
  return tolower(n ? (unsigned)*s1 : 0) - tolower(*s2) ;
}

/** ------------------------------------------------------------------
 ** @brief Case insensitive string equality test with array
 ** @param array first string (as a MATLAB array).
 ** @param string second string.
 ** @return true if the strings are equal.
 **/

static int
vlmxIsEqualToStringI(mxArray const * array, char const  * string)
{
  return vlmxCompareToStringI(array, string) == 0 ;
}

/* ---------------------------------------------------------------- */
/*                        Options handling                          */
/* ---------------------------------------------------------------- */

/** @brief MEX option */

struct _vlmxOption
{
  const char *name ; /**< option name */
  int has_arg ;      /**< has argument? */
  int val ;          /**< value to return */
} ;

/** @brief MEX option type */

typedef struct _vlmxOption vlmxOption  ;

/** ------------------------------------------------------------------
 ** @brief Parse the next option
 ** @param args     MEX argument array.
 ** @param nargs    MEX argument array length.
 ** @param options  List of option definitions.
 ** @param next     Pointer to the next option (input and output).
 ** @param optarg   Pointer to the option optional argument (output).
 ** @return the code of the next option, or -1 if there are no more options.
 **
 ** The function parses the array @a args for options. @a args is
 ** expected to be a sequence alternating option names and option
 ** values, in the form of @a nargs instances of @c mxArray. The
 ** function then scans the option starting at position @a next in the
 ** array.  The option name is matched (case insensitive) to the table
 ** of options @a options, a pointer to the option value is stored in
 ** @a optarg, @a next is advanced to the next option, and the option
 ** code is returned.
 **
 ** The function is typically used in a loop to parse all the available
 ** options. @a next is initialized to zero, and then the function
 ** is called until the special code -1 is returned.
 **
 ** If the option name cannot be matched to the available options,
 ** either because the option name is not a string array or because
 ** the name is unknown, the function exits the MEX file with an
 ** error.
 **/

static int
vlmxNextOption (mxArray const *args[], int nargs,
                vlmxOption  const *options,
                int *next,
                mxArray const **optarg)
{
  char name [1024] ;
  int opt = -1, i;

  if (*next >= nargs) {
    return opt ;
  }

  /* check the array is a string */
  if (! vlmxIsString (args [*next], -1)) {
    vlmxError (vlmxErrInvalidOption,
               "The option name is not a string (argument number %d)",
               *next + 1) ;
  }

  /* retrieve option name */
  if (mxGetString (args [*next], name, sizeof(name))) {
    vlmxError (vlmxErrInvalidOption,
               "The option name is too long (argument number %d)",
               *next + 1) ;
  }

  /* advance argument list */
  ++ (*next) ;

  /* now lookup the string in the option table */
  for (i = 0 ; options[i].name != 0 ; ++i) {
    if (vlmxCompareStringsI(name, options[i].name) == 0) {
      opt = options[i].val ;
      break ;
    }
  }

  /* unknown argument */
  if (opt < 0) {
    vlmxError (vlmxErrInvalidOption, "Unknown option '%s'.", name) ;
  }

  /* no argument */
  if (! options [i].has_arg) {
    if (optarg) *optarg = 0 ;
    return opt ;
  }

  /* argument */
  if (*next >= nargs) {
    vlmxError(vlmxErrInvalidOption,
              "Option '%s' requires an argument.", options[i].name) ;
  }

  if (optarg) *optarg = args [*next] ;
  ++ (*next) ;
  return opt ;
}


/* -------------------------------------------------------------------
 *                                                       VlEnumeration
 * ---------------------------------------------------------------- */

/** @name String enumerations
 ** @{ */

/** @brief Member of an enumeration */
typedef struct _VlEnumerator
{
  char const *name ; /**< enumeration member name. */
  vl_index value ;   /**< enumeration member value. */
} VlEnumerator ;

/** @brief Get a member of an enumeration by name
 ** @param enumeration array of ::VlEnumerator objects.
 ** @param name the name of the desired member.
 ** @return enumerator matching @a name.
 **
 ** If @a name is not found in the enumeration, then the value
 ** @c NULL is returned.
 **
 ** @sa vl-stringop-enumeration
 **/

static VlEnumerator*
vl_enumeration_get (VlEnumerator const *enumeration, char const *name)
{
  assert(enumeration) ;
  while (enumeration->name) {
    if (strcmp(name, enumeration->name) == 0) return (VlEnumerator*)enumeration ;
    enumeration ++ ;
  }
  return NULL ;
}

/** @brief Get a member of an enumeration by name (case insensitive)
 ** @param enumeration array of ::VlEnumerator objects.
 ** @param name the name of the desired member.
 ** @return enumerator matching @a name.
 **
 ** If @a name is not found in the enumeration, then the value
 ** @c NULL is returned. @a string is matched case insensitive.
 **
 **  @sa vl-stringop-enumeration
 **/

static VlEnumerator*
vl_enumeration_get_casei (VlEnumerator const *enumeration, char const *name)
{
  assert(enumeration) ;
  while (enumeration->name) {
    if (vlmxCompareStringsI(name, enumeration->name) == 0) return (VlEnumerator*)enumeration ;
    enumeration ++ ;
  }
  return NULL ;
}

/** @brief Get a member of an enumeration by value
 ** @param enumeration array of ::VlEnumerator objects.
 ** @param value value of the desired member.
 ** @return enumerator matching @a value.
 **
 ** If @a value is not found in the enumeration, then the value
 ** @c NULL is returned.
 **
 ** @sa vl-stringop-enumeration
 **/

static VlEnumerator *
vl_enumeration_get_by_value (VlEnumerator const *enumeration, vl_index value)
{
  assert(enumeration) ;
  while (enumeration->name) {
    if (enumeration->value == value) return (VlEnumerator*)enumeration ;
    enumeration ++ ;
  }
  return NULL ;
}

/** @brief Get an emumeration member by name
 ** @param enumeration the enumeration to decode.
 ** @param name_array member name as a MATLAB string array.
 ** @param caseInsensitive if @c true match the string case-insensitive.
 ** @return the corresponding enumeration member, or @c NULL if any.
 **/

static VlEnumerator *
vlmxDecodeEnumeration (mxArray const *name_array,
                       VlEnumerator const *enumeration,
                       vl_bool caseInsensitive)
{
  char name [1024] ;

  /* check the array is a string */
  if (! vlmxIsString (name_array, -1)) {
    vlmxError (vlmxErrInvalidArgument, "The array is not a string.") ;
  }

  /* retrieve option name */
  if (mxGetString (name_array, name, sizeof(name))) {
    vlmxError (vlmxErrInvalidArgument, "The string array is too long.") ;
  }

  if (caseInsensitive) {
    return vl_enumeration_get_casei(enumeration, name) ;
  } else {
    return vl_enumeration_get(enumeration, name) ;
  }
}

/* MEXUTILS_H */
#endif
