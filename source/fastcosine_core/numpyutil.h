#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL fastcosine_ARRAY_API
#include <numpy/arrayobject.h>

static PyArrayObject *pyvector(PyObject *objin) {
	return (PyArrayObject *)PyArray_ContiguousFromObject(objin, NPY_FLOAT, 1, 1);
}

static PyArrayObject *convertToUIntegerArray(PyObject *object, int minDepth, int maxDepth) {
	int flags = NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED;
	return PyArray_FromAny(
		object, PyArray_DescrFromType(NPY_UINT64), minDepth, maxDepth, flags, NULL);
}

static PyArrayObject *convertToIntegerArray(PyObject *object, int minDepth, int maxDepth) {
	int flags = NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED;
	return PyArray_FromAny(
		object, PyArray_DescrFromType(NPY_INT64), minDepth, maxDepth, flags, NULL);
}

static PyArrayObject *convertToDoubleArray(PyObject *object, int minDepth, int maxDepth) {
	int flags = NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED;
	return PyArray_FromAny(
		object, PyArray_DescrFromType(NPY_FLOAT64), minDepth, maxDepth, flags, NULL);
}

static PyArrayObject *convertToFloatArray(PyObject *object, int minDepth, int maxDepth) {
	int flags = NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED;
	return PyArray_FromAny(
		object, PyArray_DescrFromType(NPY_FLOAT32), minDepth, maxDepth, flags, NULL);
}

/* ==== Create 1D Carray from PyArray ======================
Assumes PyArray
is contiguous in memory.             */
static void *pyvector_to_Carrayptrs(PyArrayObject *arrayin) {
	int i, n;

	n = PyArray_DIM(arrayin, 0);
	return PyArray_DATA(arrayin); /* pointer to arrayin data as double */
}

/* ==== Check that PyArrayObject is a double (Float) type and a vector
============== return 1 if an error and raise exception */
static int not_floatvector(PyArrayObject *vec) {
	if (PyArray_TYPE(vec) != NPY_FLOAT) {
		PyErr_SetString(PyExc_ValueError, "In not_floatvector: array must be of "
										  "type Float and 1 dimensional (n).");
		return 1;
	}
	return 0;
}

/* ==== Check that PyArrayObject is a double (Float) type and a vector
============== return 1 if an error and raise exception */
// FIXME: make it work for 32bits
static int not_intvector(PyArrayObject *vec) {
	if (PyArray_TYPE(vec) != NPY_UINT64) {
		PyErr_SetString(PyExc_ValueError, "In not_intvector: array must be of "
										  "type Long and 1 dimensional (n).");
		return 1;
	}
	return 0;
}
