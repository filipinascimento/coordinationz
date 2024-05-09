/*
Fast Cosine Similarity Core
Fast cosine similarity calculation for bipartite graphs in Python using C.

This module is part of the CoordinationZ project.

MIT License

Copyright (c) 2024 Filipi Nascimento Silva

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/




#include <Python.h>
#include "cxnetwork/CVCommons.h"
#include "cxnetwork/CVNetwork.h"

// #define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL fastcosine_core_ARRAY_API
#include <numpy/arrayobject.h>

// Function to free the allocated memory when the capsule is garbage collected
void capsule_cleanup(PyObject *capsule) {
    void *memory = PyCapsule_GetPointer(capsule, NULL);
    free(memory);
}

static PyObject *cosine(PyObject *self, PyObject *args, PyObject *kwds) {
	
	static char *kwlist[] = {
		"sortedEdges", "weights", "leftCount","rightCount",
		"threshold","leftEdges","returnDictionary",
		"updateCallback","callbackUpdateInterval", NULL};
	// sortedEdges: list or nparray (they are sorted by left and then right node indices)
	// leftCount: int
	// rightCount: int
	// threshold: float/double
	PyObject* edgesObject = NULL;
	PyArrayObject *edgesArray = NULL;
	
	PyObject* leftEdgesObject = NULL;
	PyArrayObject *leftEdgesArray = NULL;

	PyObject* weightsObject = NULL;
	PyArrayObject *weightsArray = NULL;

	Py_ssize_t leftCount = 0;
	Py_ssize_t rightCount = 0;
	double threshold = 0.0;
	int returnDictionary = 0;
	PyObject *updateCallback = NULL;
	Py_ssize_t updateInterval = 0;
	
	if (!PyArg_ParseTupleAndKeywords(
			args, kwds, "O|OnndOpOn", kwlist, &edgesObject,&weightsObject, &leftCount, &rightCount, &threshold, &leftEdgesObject,&returnDictionary, &updateCallback, &updateInterval)) {

		return NULL;
	}
	
	if (leftCount <= 0 || rightCount<=0) {
		PyErr_SetString(
			PyExc_TypeError, "The number of ndoes (leftCount or rightCount) must be a positive integer.");
		return NULL;
	}
	
	if(updateCallback == NULL || updateCallback == Py_None){
		updateCallback = NULL;
	}

	// check if updateCallback is a callable object
	if (updateCallback != NULL && !PyCallable_Check(updateCallback)) {
		PyErr_SetString(PyExc_TypeError, "The updateCallback must be a callable object.");
		return NULL;
	}

	// convert edgesObject to numpy array of integers if needed (will be readonly)
	// lists are also supported
	// Check if edgesObject is a numpy array
	if (PyArray_Check(edgesObject)) {
		// Check if the numpy array is of type int64 and contiguous
		if (PyArray_TYPE(edgesObject) == NPY_INT64 && PyArray_IS_C_CONTIGUOUS(edgesObject)) {
			edgesArray = (PyArrayObject *)edgesObject;
			Py_INCREF(edgesArray);
			// printf("edgesArray is a numpy array of int64 and contiguous...\n");
		} else {
			// Convert edgesObject to numpy array of int64
			edgesArray = (PyArrayObject *)PyArray_FROM_OTF(edgesObject, NPY_INT64, NPY_ARRAY_C_CONTIGUOUS);
			// printf("Had to convert to array...\n");
		}
	} else {
		// Convert edgesObject to numpy array of int64
		edgesArray = (PyArrayObject *)PyArray_FROM_OTF(edgesObject, NPY_INT64, NPY_ARRAY_C_CONTIGUOUS);
		// printf("Not NP array, converting...\n");
	}

	size_t edgeCount = (size_t)PyArray_SIZE(edgesArray) / 2;
	npy_int64 *edges = PyArray_DATA(edgesArray);


	npy_float64 *weights = NULL;

	if(weightsObject!=NULL && weightsObject!=Py_None){
		// printf("Weights provided!\n");
		// // print using python weightsObject
		PyObject *weightsRepr = PyObject_Repr(weightsObject);

		// convert weightsObject to numpy array of doubles if needed (will be readonly)
		if (PyArray_Check(weightsObject)) {
			// Check if the numpy array is of type double and contiguous
			if (PyArray_TYPE(weightsObject) == NPY_FLOAT64 && PyArray_IS_C_CONTIGUOUS(weightsObject)) {
				weightsArray = (PyArrayObject *)weightsObject;
				Py_INCREF(weightsArray);
				// printf("edgesArray is a numpy array of double and contiguous...\n");
			} else {
				// Convert edgesObject to numpy array of int64
				weightsArray = (PyArrayObject *)PyArray_FROM_OTF(weightsObject, NPY_FLOAT64, NPY_ARRAY_C_CONTIGUOUS);
				// printf("Had to convert to array...\n");
			}
		} else {
			// Convert edgesObject to numpy array of int64
			weightsArray = (PyArrayObject *)PyArray_FROM_OTF(weightsObject, NPY_FLOAT64, NPY_ARRAY_C_CONTIGUOUS);
			// printf("Not NP array, converting...\n");
		}


		

		size_t weightsCount = (size_t)PyArray_SIZE(weightsArray);
		weights = PyArray_DATA(weightsArray);
		weightsRepr = PyObject_Repr(weightsArray);
		// printf("Weights ARRAY: %s\n", PyUnicode_AsUTF8(weightsRepr));
		
		// // print weights
		// printf("\n----------\n");
		// for (size_t i = 0; i < weightsCount; i++) {
		// 	// edge: weight
		// 	printf("|%ld,%ld : ",edges[2 * i], edges[2 * i + 1]);
		// 	printf("%g", weights[i]);
		// 	printf("\n");
		// }

		if(weightsCount!=edgeCount){
			PyErr_SetString(PyExc_TypeError, "The number of weights must be equal to the number of edges.");
			return NULL;
		}
	}
	
	// same for leftEdges but check if it is not NULL, if it is NULL keep it NULL


	if (leftEdgesObject != NULL && leftEdgesObject != Py_None) {
		if (PyArray_Check(leftEdgesObject)) {
			// Check if the numpy array is of type int64 and contiguous
			if (PyArray_TYPE(leftEdgesObject) == NPY_INT64 && PyArray_ISCONTIGUOUS(leftEdgesObject)) {
				leftEdgesArray = (PyArrayObject *)leftEdgesObject;
				Py_INCREF(leftEdgesArray);
				// printf("leftEdgesArray is a numpy array of int64 and contiguous...\n");
			} else {
				// Convert leftEdgesObject to numpy array of int64
				leftEdgesArray = (PyArrayObject *)PyArray_FROM_OTF(leftEdgesObject, NPY_INT64, NPY_ARRAY_C_CONTIGUOUS);
				// printf("Had to convert to array...\n");
			}
		} else {
			// Convert leftEdgesObject to numpy array of int64
			leftEdgesArray = (PyArrayObject *)PyArray_FROM_OTF(leftEdgesObject, NPY_INT64, NPY_ARRAY_C_CONTIGUOUS);
			// printf("Not NP array, converting...\n");
		}
	}else{
		leftEdgesObject = NULL;
	}


	size_t leftEdgeCount = 0;
	npy_int64 *leftEdges = NULL;
	if (leftEdgesArray != NULL) {
		leftEdgeCount = (size_t)PyArray_SIZE(leftEdgesArray) / 2;
		leftEdges = PyArray_DATA(leftEdgesArray);
	}

	npy_int64 *leftNodeStartEndIndices = (npy_int64 *)calloc((leftCount+1),sizeof(npy_int64));

	// create multiplicity array countaining for each edge, the number of repetitions
	npy_int64 *uniqueEdges = calloc((edgeCount*2),sizeof(npy_int64));
	npy_double *uniqueEdgesWeight = calloc((edgeCount),sizeof(npy_double));
	size_t uniqueEdgeCount = 0;
	npy_int64 previousLeftNode = -1;
	npy_int64 previousRightNode = -1;
	npy_double previousWeight = -1;
	size_t multiplicity = 1;

	for (size_t i = 0; i < edgeCount; i++) {
		npy_int64 leftNode = edges[2 * i];
		npy_int64 rightNode = edges[2 * i + 1];
		npy_double weight = 1.0;
		if(weights){
			weight = weights[i];
		}
		if (leftNode == previousLeftNode && rightNode == previousRightNode) {
			multiplicity++;
		} else {
			if (CVLikely(i != 0)) {
				uniqueEdges[2 * uniqueEdgeCount] = previousLeftNode;
				uniqueEdges[2 * uniqueEdgeCount + 1] = previousRightNode;
				if(weights){
					uniqueEdgesWeight[uniqueEdgeCount] = previousWeight*(double)multiplicity;
					// printf("!weight: %g, %ld\n", previousWeight,multiplicity);
				}else{
					uniqueEdgesWeight[uniqueEdgeCount] = (double)multiplicity;
				}
				uniqueEdgeCount++;
			}
			previousLeftNode = leftNode;
			previousRightNode = rightNode;
			previousWeight = weight;
			multiplicity = 1;
		}
		if(CVUnlikely(i == edgeCount-1)) {
			uniqueEdges[2 * uniqueEdgeCount] = leftNode;
			uniqueEdges[2 * uniqueEdgeCount + 1] = rightNode;
			if(weights){
				uniqueEdgesWeight[uniqueEdgeCount] = weight*(double)multiplicity;
			}else{
				uniqueEdgesWeight[uniqueEdgeCount] = multiplicity;
			}
			uniqueEdgeCount++;
		}
	}

	


	// printf("repetitions:\n");
	// for (size_t i = 0; i < uniqueEdgeCount; i++) {
	// 	npy_int64 leftNode = uniqueEdges[2 * i];
	// 	npy_int64 rightNode = uniqueEdges[2 * i + 1];
	// 	printf("%ld %ld: %g\n", leftNode, rightNode, uniqueEdgesWeight[i]);
	// }
	
	// loop over edges and find the start and end indices of each left node
	// if not found then set to start=end=previous index
	// For instance:
	// 0-5
	// 5-7
	// 7-7
	// 7-9
	// meaning that node 2 has no edges
	previousLeftNode = -1;
	for (size_t i = 0; i < uniqueEdgeCount; i++) {
		npy_int64 leftNode = uniqueEdges[2 * i];
		if (leftNode != previousLeftNode) {
			for (size_t j = previousLeftNode + 1; j <= leftNode; j++) {
				leftNodeStartEndIndices[j] = i;
			}
			previousLeftNode = leftNode;
		}
	}

	// fix the last node
	for (size_t j = previousLeftNode + 1; j <= leftCount; j++) {
		leftNodeStartEndIndices[j] = uniqueEdgeCount;
	}




	// // print edges

	// printf("edges: \n");
	// for (size_t i = 0; i < edgeCount; i++) {
	// 	printf("%ld: %ld %ld\n",i, edges[2 * i], edges[2 * i + 1]);
	// }

	// printf("start-end indices\n");
	// // print all the start-end indices write NO if not found
	// for (size_t i = 0; i < leftCount; i++) {
	// 	if (leftNodeStartEndIndices[i] == -1) {
	// 		printf("%ld: NO\n", i);
	// 	} else {
	// 		printf("%ld: %ld-%ld\n", i, leftNodeStartEndIndices[i], leftNodeStartEndIndices[i + 1]);
	// 	}
	// }

	// printf("node degrees\n");
	// // print all the node degrees
	// for (size_t i = 0; i < leftCount; i++) {
	// 	if (leftNodeStartEndIndices[i] == -1) {
	// 		printf("%ld: 0\n", i);
	// 	} else {
	// 		printf("%ld: %ld\n", i, leftNodeStartEndIndices[i + 1] - leftNodeStartEndIndices[i]);
	// 	}
	// }
	
	// printf("Normalization factors:\n");
	double *normalizationFactor = calloc(leftCount, sizeof(double));
	// calculate the cosine similarity normalization factor for each left node
	// based on unique edges and their multiplicity
	for (size_t i = 0; i < leftCount; i++) {
		size_t leftNodeStart = leftNodeStartEndIndices[i];
		size_t leftNodeEnd = leftNodeStartEndIndices[i + 1];
		double normalizationFactorValue = 0.0;
		for (size_t j = leftNodeStart; j < leftNodeEnd; j++) {
			normalizationFactorValue += uniqueEdgesWeight[j] * uniqueEdgesWeight[j];
		}
		normalizationFactor[i] = 1.0 / sqrt(normalizationFactorValue);
		// printf("%ld: %f\n", i, normalizationFactor[i]);
	}


	size_t similaritiesCapacity = 1000;
	size_t similaritiesCount = 0;
	npy_int64 *aboveThresholdEdges = NULL;
	double *similarities = NULL;
	PyObject *similaritiesDict = NULL;

	if(returnDictionary){
		similaritiesDict = PyDict_New();
	}else{
		aboveThresholdEdges = calloc(similaritiesCapacity*2, sizeof(npy_int64)); 
		similarities = calloc(similaritiesCapacity, sizeof(double));
	}

	// calculate the cosine similarity among all pairs of left nodes
	CVBool errorOccured = CVFalse;

	if(updateInterval==0){
		updateInterval = leftCount*10;
	}
	size_t totalIterations = leftCount*(leftCount-1)/2;
	if(leftEdgesArray == NULL) {
		npy_int64 iterations = -1;
		for(size_t i = 0; i < leftCount; i++) {
			if(CVUnlikely(errorOccured)){
				break;
			}
			for(size_t j = i+1; j < leftCount; j++) {
				iterations++;
				if(iterations % updateInterval == 0) {
					// printf("\r%ld/%ld", i, leftCount);
					// fflush(stdout);
					// if defined, call updateCallback with 2 parameters, current and total
					if (updateCallback != NULL) {
						PyObject *result = PyObject_CallFunction(updateCallback, "nn", iterations, totalIterations);
						if (result == NULL) {
							errorOccured = CVTrue;
							break;
						}
						Py_DECREF(result);
					}
					// python checks to see if the process is interrupted by error
					if (PyErr_CheckSignals() != 0) {
						errorOccured = CVTrue;
						break;
					}
				}


				size_t leftNode1Start = leftNodeStartEndIndices[i];
				size_t leftNode1End = leftNodeStartEndIndices[i+1];
				size_t leftNode1Degree = leftNode1End - leftNode1Start;
				// calculate the dot product
				double dotProduct = 0.0;
				size_t leftNode2Start = leftNodeStartEndIndices[j];
				size_t leftNode2End = leftNodeStartEndIndices[j+1];
				size_t leftNode2Degree = leftNode2End - leftNode2Start;

				size_t leftNode1Index = leftNode1Start;
				size_t leftNode2Index = leftNode2Start;
				while (leftNode1Index < leftNode1End && leftNode2Index < leftNode2End) {
					npy_int64 rightNode1 = uniqueEdges[2 * leftNode1Index + 1];
					npy_int64 rightNode2 = uniqueEdges[2 * leftNode2Index + 1];
					if (rightNode1 == rightNode2) { 
						dotProduct += uniqueEdgesWeight[leftNode1Index] * uniqueEdgesWeight[leftNode2Index];
						leftNode1Index++;
						leftNode2Index++;
					} else if (rightNode1 < rightNode2) {
						leftNode1Index++;
					} else {
						leftNode2Index++;
					}
				}
				double similarity = dotProduct*normalizationFactor[i]*normalizationFactor[j];
				if (similarity > threshold) {
					if(returnDictionary){
						PyObject *key = PyTuple_Pack(2, PyLong_FromLong(i), PyLong_FromLong(j));
						PyObject *value = PyFloat_FromDouble(similarity);
						PyDict_SetItem(similaritiesDict, key, value);
						Py_DECREF(key);
						Py_DECREF(value);
					}else{
						if (CVUnlikely(similaritiesCount == similaritiesCapacity)) {
							similaritiesCapacity *= 2;
							aboveThresholdEdges = (npy_int64 *)realloc(aboveThresholdEdges, similaritiesCapacity * 2 * sizeof(npy_int64));
							similarities = (double *)realloc(similarities, similaritiesCapacity * sizeof(double));
						}
						aboveThresholdEdges[2 * similaritiesCount] = i;
						aboveThresholdEdges[2 * similaritiesCount + 1] = j;
						similarities[similaritiesCount] = similarity;
						similaritiesCount++;
					}
				}
			}
		}
	}else{
		totalIterations = leftEdgeCount;
		npy_int64 iterations = -1;
		for(size_t leftEdgeIndex = 0; leftEdgeIndex < leftEdgeCount; leftEdgeIndex++) {
			size_t i = leftEdges[2 * leftEdgeIndex];
			size_t j = leftEdges[2 * leftEdgeIndex + 1];
			{
				iterations++;
				if(iterations % updateInterval == 0) {
					// printf("\r%ld/%ld", i, leftCount);
					// fflush(stdout);
					// if defined, call updateCallback with 2 parameters, current and total
					if (updateCallback != NULL) {
						PyObject *result = PyObject_CallFunction(updateCallback, "nn", iterations, totalIterations);
						if (result == NULL) {
							errorOccured = CVTrue;
							break;
						}
						Py_DECREF(result);
					}
					// python checks to see if the process is interrupted
					if (PyErr_CheckSignals() != 0) {
						errorOccured = CVTrue;
						break;
					}
				}


				size_t leftNode1Start = leftNodeStartEndIndices[i];
				size_t leftNode1End = leftNodeStartEndIndices[i+1];
				size_t leftNode1Degree = leftNode1End - leftNode1Start;
				// calculate the dot product
				double dotProduct = 0.0;
				size_t leftNode2Start = leftNodeStartEndIndices[j];
				size_t leftNode2End = leftNodeStartEndIndices[j+1];
				size_t leftNode2Degree = leftNode2End - leftNode2Start;

				size_t leftNode1Index = leftNode1Start;
				size_t leftNode2Index = leftNode2Start;
				while (leftNode1Index < leftNode1End && leftNode2Index < leftNode2End) {
					npy_int64 rightNode1 = uniqueEdges[2 * leftNode1Index + 1];
					npy_int64 rightNode2 = uniqueEdges[2 * leftNode2Index + 1];
					if (rightNode1 == rightNode2) { 
						dotProduct += uniqueEdgesWeight[leftNode1Index] * uniqueEdgesWeight[leftNode2Index];
						leftNode1Index++;
						leftNode2Index++;
					} else if (rightNode1 < rightNode2) {
						leftNode1Index++;
					} else {
						leftNode2Index++;
					}
				}
				double similarity = dotProduct*normalizationFactor[i]*normalizationFactor[j];
				if (similarity > threshold) {
					if(returnDictionary){
						PyObject *key = PyTuple_Pack(2, PyLong_FromLong(i), PyLong_FromLong(j));
						PyObject *value = PyFloat_FromDouble(similarity);
						PyDict_SetItem(similaritiesDict, key, value);
						Py_DECREF(key);
						Py_DECREF(value);
					}else{
						if (CVUnlikely(similaritiesCount == similaritiesCapacity)) {
							similaritiesCapacity *= 2;
							aboveThresholdEdges = (npy_int64 *)realloc(aboveThresholdEdges, similaritiesCapacity * 2 * sizeof(npy_int64));
							similarities = (double *)realloc(similarities, similaritiesCapacity * sizeof(double));
						}
						aboveThresholdEdges[2 * similaritiesCount] = i;
						aboveThresholdEdges[2 * similaritiesCount + 1] = j;
						similarities[similaritiesCount] = similarity;
						similaritiesCount++;
					}
				}
			}
		}
	}


	Py_XDECREF(edgesArray);

	if(leftEdgesArray != NULL) {
		Py_XDECREF(leftEdgesArray);
	}

	if(weightsArray != NULL) {
		Py_XDECREF(weightsArray);
	}

	free(normalizationFactor);
	free(uniqueEdges);
	free(uniqueEdgesWeight);
	free(leftNodeStartEndIndices);

	if(errorOccured) {
		// clean up
		free(aboveThresholdEdges);
		free(similarities);
		
		return NULL;
	}

	// final update callback
	if (updateCallback != NULL) {
		PyObject *result = PyObject_CallFunction(updateCallback, "nn", totalIterations, totalIterations);
		if (result == NULL) {
			errorOccured = CVTrue;
		}
		Py_DECREF(result);
	}


	// // print the similarities as node, node: similarity
	// printf("\n");
	// for (size_t i = 0; i < similaritiesCount; i++) {
	// 	printf("%ld, %ld: %f\n", aboveThresholdEdges[2 * i], aboveThresholdEdges[2 * i + 1], similarities[i]);
	// }

	PyObject *result;
	if(returnDictionary){
		result = similaritiesDict;
	}else{
		// return a tuple of 2 numpy arrays based aboveThresholdEdges (dimension 2) and similarities
		npy_intp dims[2] = {similaritiesCount, 2};
		PyObject *aboveThresholdEdgesArray = PyArray_SimpleNewFromData(2, dims, NPY_INT64, aboveThresholdEdges);
		PyObject *similaritiesArray = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, similarities);

		PyObject *aboveThresholdEdgesCapsule = PyCapsule_New(aboveThresholdEdges, NULL, capsule_cleanup);
		PyObject *similaritiesCapsule = PyCapsule_New(similarities, NULL, capsule_cleanup);

		if (!aboveThresholdEdgesCapsule || !similaritiesCapsule) {
			// Cleanup if capsule creation fails
			Py_XDECREF(aboveThresholdEdgesCapsule);
			Py_XDECREF(similaritiesCapsule);
			free(aboveThresholdEdges);
			free(similarities);
			return NULL;
		}

		PyArray_SetBaseObject((PyArrayObject *)aboveThresholdEdgesArray, aboveThresholdEdgesCapsule);
		PyArray_SetBaseObject((PyArrayObject *)similaritiesArray, similaritiesCapsule);

		result = PyTuple_Pack(2, aboveThresholdEdgesArray, similaritiesArray);
		
		// free(aboveThresholdEdges);
		// free(similarities);
		Py_DECREF(aboveThresholdEdgesArray);
		Py_DECREF(similaritiesArray);
	}
	return result;
	// Py_RETURN_NONE;
}

static PyMethodDef methods[] = {
	{"cosine", (PyCFunction)cosine, METH_VARARGS | METH_KEYWORDS, "Calculate cosine similarity with threshold."},
	{NULL, NULL, 0, NULL},
};

static struct PyModuleDef module = {
	PyModuleDef_HEAD_INIT,
	"fastcosine_core",
	NULL,
	-1,
	methods,
};

PyMODINIT_FUNC PyInit_fastcosine_core(void) {
	import_array();
	return PyModule_Create(&module);
}
