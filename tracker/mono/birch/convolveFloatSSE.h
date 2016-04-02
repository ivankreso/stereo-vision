#ifndef convolveFloatSSE_h
#define convolveFloatSSE_h

void convolveFloatSSEHor(
  float* imgin, int nrows, int ncols,
  float* kernel, int szker,
  float* imgout);
void convolveFloatSSEVert(
  float* imgin, int nrows, int ncols,
  float* kernel, int szker,
  float* imgout);

#endif

