/*********************************************************************
 * convolve.c
 *********************************************************************/

/* Standard includes */
#include <assert.h>
#include <math.h>
#include <stdlib.h>   /* malloc(), realloc() */

/* Our includes */
#include "base.h"
#include "error.h"
#include "convolve.h"
#include "klt_util.h"   /* printing */

#include "convolveFloatSSE.h"

#define MAX_KERNEL_WIDTH 	71
int useSSE=1; //Use sse acceleration

typedef struct  {
  int width;
  float data[MAX_KERNEL_WIDTH];
}  ConvolutionKernel;

/* Kernels */
static ConvolutionKernel gauss_kernel;
static ConvolutionKernel gaussderiv_kernel;
static float sigma_last = -10.0;


/*********************************************************************
 * _KLTToFloatImage
 *
 * Given a pointer to image data (probably unsigned chars), copy
 * data to a float image.
 */

void _KLTToFloatImage(
  KLT_PixelType *img,
  int ncols, int nrows,
  _KLT_FloatImage floatimg)
{
  KLT_PixelType *ptrend = img + ncols*nrows;
  float *ptrout = floatimg->data;

  /* Output image must be large enough to hold result */
  assert(floatimg->ncols >= ncols);
  assert(floatimg->nrows >= nrows);

  floatimg->ncols = ncols;
  floatimg->nrows = nrows;

  while (img < ptrend)  *ptrout++ = (float) *img++;
}


/*********************************************************************
 * _computeKernels
 */

static void _computeKernels(
  float sigma,
  ConvolutionKernel *gauss,
  ConvolutionKernel *gaussderiv)
{
  const float factor = 0.01f;   /* for truncating tail */
  int i;

  assert(MAX_KERNEL_WIDTH % 2 == 1);
  assert(sigma >= 0.0);

  /* Compute kernels, and automatically determine widths */
  {
    const int hw = MAX_KERNEL_WIDTH / 2;
    float max_gauss = 1.0f, max_gaussderiv = (float) (sigma*exp(-0.5f));
	
    /* Compute gauss and deriv */
    for (i = -hw ; i <= hw ; i++)  {
      gauss->data[i+hw]      = (float) exp(-i*i / (2*sigma*sigma));
      gaussderiv->data[i+hw] = -i * gauss->data[i+hw];
    }

    /* Compute widths */
    gauss->width = MAX_KERNEL_WIDTH;
    for (i = -hw ; fabs(gauss->data[i+hw] / max_gauss) < factor ; 
         i++, gauss->width -= 2);
    gaussderiv->width = MAX_KERNEL_WIDTH;
    for (i = -hw ; i<0 && fabs(gaussderiv->data[i+hw] / max_gaussderiv) < factor; 
         i++, gaussderiv->width -= 2);
    if (gauss->width == MAX_KERNEL_WIDTH || 
        gaussderiv->width == MAX_KERNEL_WIDTH)
      KLTError("(_computeKernels) MAX_KERNEL_WIDTH %d is too small for "
               "a sigma of %f", MAX_KERNEL_WIDTH, sigma);
  }

  /* Shift if width less than MAX_KERNEL_WIDTH */
  for (i = 0 ; i < gauss->width ; i++)
    gauss->data[i] = gauss->data[i+(MAX_KERNEL_WIDTH-gauss->width)/2];
  for (i = 0 ; i < gaussderiv->width ; i++)
    gaussderiv->data[i] = gaussderiv->data[i+(MAX_KERNEL_WIDTH-gaussderiv->width)/2];
  /* Normalize gauss and deriv */
  {
    const int hw = gaussderiv->width / 2;
    float den;
			
    den = 0.0;
    for (i = 0 ; i < gauss->width ; i++)  den += gauss->data[i];
    for (i = 0 ; i < gauss->width ; i++)  gauss->data[i] /= den;
    den = 0.0;
    for (i = -hw ; i <= hw ; i++)  den -= i*gaussderiv->data[i+hw];
    for (i = -hw ; i <= hw ; i++)  gaussderiv->data[i+hw] /= den;
  }

  sigma_last = sigma;
}
	

/*********************************************************************
 * _KLTGetKernelWidths
 *
 */

void _KLTGetKernelWidths(
  float sigma,
  int *gauss_width,
  int *gaussderiv_width)
{
  _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);
  *gauss_width = gauss_kernel.width;
  *gaussderiv_width = gaussderiv_kernel.width;
}


/*********************************************************************
 * _convolveImageHoriz
 */

static void _convolveImageHoriz(
  _KLT_FloatImage imgin,
  ConvolutionKernel kernel,
  _KLT_FloatImage imgout)
{
  float *ptrrow = imgin->data;           /* Points to row's first pixel */
  register float *ptrout = imgout->data, /* Points to next output pixel */
    *ppp;
  register float sum;
  register int radius = kernel.width / 2;
  register int ncols = imgin->ncols, nrows = imgin->nrows;
  register int i, j, k;

  /* Kernel width must be odd */
  assert(kernel.width % 2 == 1);

  /* Must read from and write to different images */
  assert(imgin != imgout);

  /* Output image must be large enough to hold result */
  assert(imgout->ncols >= imgin->ncols);
  assert(imgout->nrows >= imgin->nrows);

  /* For each row, do ... */
  for (j = 0 ; j < nrows ; j++)  {

    /* Zero leftmost columns */
    for (i = 0 ; i < radius ; i++)
      *ptrout++ = 0.0;

    /* Convolve middle columns with kernel */
    for ( ; i < ncols - radius ; i++)  {
      ppp = ptrrow + i - radius;
      sum = 0.0;
      for (k = kernel.width-1 ; k >= 0 ; k--)
        sum += *ppp++ * kernel.data[k];
      *ptrout++ = sum;
    }

    /* Zero rightmost columns */
    for ( ; i < ncols ; i++)
      *ptrout++ = 0.0;

    ptrrow += ncols;
  }
}


/*********************************************************************
 * _convolveImageVert
 */

static void _convolveImageVert(
  _KLT_FloatImage imgin,
  ConvolutionKernel kernel,
  _KLT_FloatImage imgout)
{
  float *ptrcol = imgin->data;            /* Points to row's first pixel */
  register float *ptrout = imgout->data,  /* Points to next output pixel */
    *ppp;
  register float sum;
  register int radius = kernel.width / 2;
  register int ncols = imgin->ncols, nrows = imgin->nrows;
  register int i, j, k;

  /* Kernel width must be odd */
  assert(kernel.width % 2 == 1);

  /* Must read from and write to different images */
  assert(imgin != imgout);

  /* Output image must be large enough to hold result */
  assert(imgout->ncols >= imgin->ncols);
  assert(imgout->nrows >= imgin->nrows);

  /* For each column, do ... */
  for (i = 0 ; i < ncols ; i++)  {

    /* Zero topmost rows */
    for (j = 0 ; j < radius ; j++)  {
      *ptrout = 0.0;
      ptrout += ncols;
    }

    /* Convolve middle rows with kernel */
    for ( ; j < nrows - radius ; j++)  {
      ppp = ptrcol + ncols * (j - radius);
      sum = 0.0;
      for (k = kernel.width-1 ; k >= 0 ; k--)  {
        sum += *ppp * kernel.data[k];
        ppp += ncols;
      }
      *ptrout = sum;
      ptrout += ncols;
    }

    /* Zero bottommost rows */
    for ( ; j < nrows ; j++)  {
      *ptrout = 0.0;
      ptrout += ncols;
    }

    ptrcol++;
    ptrout -= nrows * ncols - 1;
  }
}

/*SSE3 convolution implementation*/

/*********************************************************************
 * _convolveImageHorizSSE
 */

static void _convolveImageHorizSSE(
  _KLT_FloatImage imgin,
  ConvolutionKernel kernel,
  _KLT_FloatImage imgout)
{
  int i=0;
  for (i=0; i<kernel.width/2; ++i){
    float tmp=kernel.data[i];
    kernel.data[i]=kernel.data[kernel.width-1-i];
    kernel.data[kernel.width-1-i]=tmp;
  }
  
  convolveFloatSSEHor(
	  imgin->data, imgin->nrows, imgin->ncols, 
	  kernel.data, kernel.width, 
	  imgout->data);
}



static void _convolveImageVertSSE(
  _KLT_FloatImage imgin,
  ConvolutionKernel kernel,
  _KLT_FloatImage imgout)
{
  int i=0;
  for (i=0; i<kernel.width/2; ++i){
    float tmp=kernel.data[i];
    kernel.data[i]=kernel.data[kernel.width-1-i];
    kernel.data[kernel.width-1-i]=tmp;
  }
  
  convolveFloatSSEVert(
	  imgin->data, imgin->nrows, imgin->ncols, 
	  kernel.data, kernel.width, 
	  imgout->data);
}



/*********************************************************************
 * _convolveSeparate
 */

static void _convolveSeparate(
  _KLT_FloatImage imgin,
  ConvolutionKernel horiz_kernel,
  ConvolutionKernel vert_kernel,
  _KLT_FloatImage imgout)
{
 
  /* Create temporary image */
  _KLT_FloatImage tmpimg;
  tmpimg = _KLTCreateFloatImage(imgin->ncols, imgin->nrows);
  
  /*Convolute */
  _convolveImageHoriz(imgin, horiz_kernel, tmpimg);
  _convolveImageVert(tmpimg, vert_kernel, imgout);
  
}


/*Test result of SSE convolution */
int testSSEf(_KLT_FloatImage imgin,_KLT_FloatImage tmpimg,
  ConvolutionKernel horiz_kernel)
{
  double eps=1e-3;
  int i;
  int szimg=tmpimg->ncols*tmpimg->nrows;
  _KLT_FloatImage tmpimg2;
  tmpimg2 = _KLTCreateFloatImage(imgin->ncols, imgin->nrows);
    
  _convolveImageHoriz(imgin, horiz_kernel, tmpimg2);

  for (i=0; i<szimg; ++i){
    double delta=fabs(tmpimg2->data[i] - tmpimg->data[i]);
    if (delta>eps){
      _KLTFreeFloatImage(tmpimg2);
      return 0;
    }
  }
  _KLTFreeFloatImage(tmpimg2);
  return 1;
}

/* ConvolveSeparate by using SSE convolution */
static void _convolveSeparateSSE(
   _KLT_FloatImage imgin,
  ConvolutionKernel horiz_kernel,
  ConvolutionKernel vert_kernel,
  _KLT_FloatImage imgout)
{
 int testSSE=0;
 int i,j,h;
 
  /* Create temporary image */
  _KLT_FloatImage tmpimg,tmpimg2,tmpimg3;
  tmpimg = _KLTCreateFloatImage(imgin->ncols, imgin->nrows); //result of first horizontal convolution on imgin
  tmpimg2 = _KLTCreateFloatImage(imgin->nrows, imgin->ncols);//will be transponed tmpimg for second horizontal convolutioon
  tmpimg3 = _KLTCreateFloatImage(imgin->nrows, imgin->ncols);//will be transponed tmpimg2 for imgout

  /* Do convolution */
  _convolveImageHorizSSE(imgin, horiz_kernel, tmpimg);
  
  //Test result of horizontal convolution
  if (testSSE){
    if (testSSEf(imgin,tmpimg,horiz_kernel)==0){
      assert(0);
    }
  }

  //Transpose tmpimg 
  for(i=0;i<tmpimg->ncols;i++){
   h=i*tmpimg->nrows;
    for(j=0;j<tmpimg->nrows;j++){
       tmpimg2->data[h+j]=tmpimg->data[j*tmpimg->ncols+i];
    }
  }
  
  //Do horizontal convolution on transponed image == equal to vertical convolution
  _convolveImageVertSSE(tmpimg2, vert_kernel, tmpimg3);

  //Transpose it again to get final result
  for(i=0;i<tmpimg2->ncols;i++){
    h=i*tmpimg2->nrows;
    for(j=0;j<tmpimg2->nrows;j++){
      imgout->data[h+j]=tmpimg3->data[j*tmpimg2->ncols+i];
    }
  }
  
  /* Free memory */
  _KLTFreeFloatImage(tmpimg);
  _KLTFreeFloatImage(tmpimg2);
  _KLTFreeFloatImage(tmpimg3);
}





	
/*********************************************************************
 * _KLTComputeGradients
 */

void _KLTComputeGradients(
  _KLT_FloatImage img,
  float sigma,
  _KLT_FloatImage gradx,
  _KLT_FloatImage grady)
{
  /* Output images must be large enough to hold result */
  assert(gradx->ncols >= img->ncols);
  assert(gradx->nrows >= img->nrows);
  assert(grady->ncols >= img->ncols);
  assert(grady->nrows >= img->nrows);

  /* Compute kernels, if necessary */
  if (fabs(sigma - sigma_last) > 0.05)
    _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);
	
  if (useSSE){
    _convolveSeparateSSE(img, gaussderiv_kernel, gauss_kernel, gradx);
    _convolveSeparateSSE(img, gauss_kernel, gaussderiv_kernel, grady);
  }
  else{
   _convolveSeparate(img, gaussderiv_kernel, gauss_kernel, gradx);
   _convolveSeparate(img, gauss_kernel, gaussderiv_kernel, grady);
  }
}
	

/*********************************************************************
 * _KLTComputeSmoothedImage
 */

void _KLTComputeSmoothedImage(
  _KLT_FloatImage img,
  float sigma,
  _KLT_FloatImage smooth)
{	
  /* Output image must be large enough to hold result */
  assert(smooth->ncols >= img->ncols);
  assert(smooth->nrows >= img->nrows);

  /* Compute kernel, if necessary; gauss_deriv is not used */
  if (fabs(sigma - sigma_last) > 0.05)
    _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);

  if(useSSE){
    _convolveSeparateSSE(img, gauss_kernel, gauss_kernel, smooth);
  } else {
    _convolveSeparate(img, gauss_kernel, gauss_kernel, smooth);
  }
 
}



