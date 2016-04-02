/*
Copyright © INRIA 2010-2011. 
Authors: Matthijs Douze & Herve Jegou 
Contact: matthijs.douze@inria.fr  herve.jegou@inria.fr

This software is a computer program whose purpose is to provide 
efficient tools for basic yet computationally demanding tasks, 
such as find k-nearest neighbors using exhaustive search 
and kmeans clustering. 

This software is governed by the CeCILL license under French law and
abiding by the rules of distribution of free software.  You can  use, 
modify and/ or redistribute the software under the terms of the CeCILL
license as circulated by CEA, CNRS and INRIA at the following URL
"http://www.cecill.info". 

As a counterpart to the access to the source code and  rights to copy,
modify and redistribute granted by the license, users are provided only
with a limited warranty  and the software's author,  the holder of the
economic rights,  and the successive licensors  have only  limited
liability. 

In this respect, the user's attention is drawn to the risks associated
with loading,  using,  modifying and/or developing or reproducing the
software by the user in light of its specific status of free software,
that may mean  that it is complicated to manipulate,  and  that  also
therefore means  that it is reserved for developers  and  experienced
professionals having in-depth computer knowledge. Users are therefore
encouraged to load and test the software's suitability as regards their
requirements in conditions enabling the security of their systems and/or 
data to be ensured and,  more generally, to use and operate it in the 
same conditions as regards security. 

The fact that you are presently reading this means that you have had
knowledge of the CeCILL license and that you accept its terms.
*/


/* Hamming distances. The binary vector length should be a power of 8 */
#ifndef __hamming_h
#define __hamming_h

typedef unsigned char uint8;
typedef unsigned short uint16;
typedef unsigned int uint32;
typedef unsigned long long uint64;

typedef long long int64;


/* matching elements (those returned) */
typedef struct hammatch_s {
  int qid;        /* query id */
  int bid;        /* base id */
  uint16 score;   /* Hamming distance */
} hammatch_t;


/* Define individual Hamming distance for various sizes.
   ncodes is given in bytes, therefore the actual number of bits is 8*ncodes.
   The generic one is slow while optimization is available for specific sizes */
uint16 hamming (const uint8 *bs1, const uint8 * bs2, int ncodes);



/* Compute a set of Hamming distances between na and nb binary vectors */
void compute_hamming (uint16 * dis, const uint8 * a, const uint8 * b, 
                      int na, int nb, int ncodes);




/* Counting the number of matches or of cross-matches (with actually returning them)
   Useful to be used with function that assume pre-allocated memory                  */
void match_hamming_count (const uint8 * bs1, const uint8 * bs2, int n1, int n2, 
                          int ht, int ncodes, size_t * nptr);

void crossmatch_hamming_count (const uint8 * dbs, int n, int ht, 
                               int ncodes, size_t * nptr);


/* For 1 query signature, compute the hamming distance and report those below a given 
   threshold in a structure array */
void match_hamming_thres (const uint8 * bs1, const uint8 * bs2, 
                          int n1, int n2, int ht, int ncodes, size_t bufsize, 
                          hammatch_t ** hmptr, size_t * nptr);


/* The same but with pre-allocation (typically used with match_hamming_count) */
size_t match_hamming_thres_prealloc (const uint8 * bs1, const uint8 * bs2, 
                                     int n1, int n2, int ht, int ncodes, 
                                     int * idx, uint16 * hams);

                                   /* Compute all cross-distances between two sets of binary vectors */
void crossmatch_hamming (const uint8 * dbs, long n, int ht, int ncodes, 
                         long bufsize, hammatch_t ** hmptr, size_t * nptr);

/* alternative variant with pre-allocated external memory.
   return number of elements for safety check. 
   Typical usage is to first invoke crossmatch_hamming_count, allocate memory,
   and then invoke crossmatch_hamming_prealloc */

size_t crossmatch_hamming_prealloc (const uint8 * dbs, long n, int ht, int ncodes,  
                                    int * idx, uint16 * hams);

/* Threaded versions, when OpenMP is available */
#ifdef _OPENMP
void compute_hamming_thread (uint16 * dis, const uint8 * a, const uint8 * b, 
                             int na, int nb, int ncodes);

size_t match_hamming_thres_nt (const uint8 * bs1, const uint8 * bs2, int n1, int n2, 
                              int ht, int ncodes, int nt, int ** keys, uint16 ** ham);
#endif /* _OPENMP */


#endif /* __hamming_h */

 