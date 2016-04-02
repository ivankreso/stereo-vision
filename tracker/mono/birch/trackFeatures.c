/*********************************************************************
 * trackFeatures.c
 *
 *********************************************************************/

/* Standard includes */
#include <assert.h>
#include <math.h>		/* fabs() */
#include <stdlib.h>		/* malloc() */
#include <stdio.h>		/* fflush() */
#include <string.h>

/* Our includes */
#include "base.h"
#include "error.h"
#include "convolve.h"	/* for computing pyramid */
#include "klt.h"
#include "klt_util.h"	/* _KLT_FloatImage */
#include "pyramid.h"	/* _KLT_Pyramid */


extern int KLT_verbose;

typedef float *_FloatWindow;

static int myround(double f){
  return (int)(f<0?f-.5:f+.5);
}

static double mymax2(double a, double b){
  return a>b?a:b;
}
static double mymax4(double a, double b, double c, double d){
  return mymax2(mymax2(a,b),mymax2(c,d));
}

int checkResidual(
  float residual, 
  float maxresidual)
{
  //assert(lambda2>=0);
  //float f=sqrt(lambda2);
  return residual>maxresidual;
}



static void updateGaussians(
  int width, int height, 
  _FloatWindow img, 
  float maxmov,
  float alpha0,
  _FloatWindow mu, 
  _FloatWindow sigmaSqr
){
  int i,j;
  float alpha=alpha0*maxmov;
  if (alpha>1) alpha=1;
  for (j=0; j<height; ++j){
    for (i=0; i<width; ++i){
      if (alpha0==3.0){
        // printf(">\n");
        *mu   =*img;
        *sigmaSqr=*sigmaSqr;
      } else if (alpha0==2.0){
        // printf("<\n");
        *mu   =*img;
        *sigmaSqr=*sigmaSqr/2;
      } else{
        double d=(*img-*mu);
        *mu = (float) alpha*(*img)+(1-alpha)* *mu;
        *sigmaSqr= (float) (alpha*(d*d) +(1-alpha)* *sigmaSqr);
      }
      ++img;
      ++mu;
      ++sigmaSqr;
    }
  }
}


static int fcompar(const void *pf1, const void *pf2){
  double d=(*(float*)(pf1))-(*(float*)(pf2));
  int rv=0;
  if (d<0) rv=-1;
  if (d>0) rv=1;
  return rv;
}
static void getSqrMAD(
  int n, _FloatWindow img,
  float* mu, float* sigma
){
  int i;
  float* tmp=(float*)malloc(sizeof(float)*n);

  _FloatWindow p=img;
  _FloatWindow err=tmp;
  for (i=0; i<n; ++i){
    *err= (*p) * (*p);
    ++p; ++err;
  }
  qsort(tmp, n, sizeof(float), fcompar);
  *mu=tmp[(n+1)/2];

  {
    _FloatWindow q=tmp;
    for (i=0; i<n; ++i){
      *q= (float) fabs(*q-*mu);
      ++q;
    }
  }
  qsort(tmp, n, sizeof(float), fcompar);
  *sigma=(float) (tmp[(n+1)/2]/0.6754);
  free(tmp);
}


static float updateMaskGaussian(
  int width, int height, 
  _FloatWindow mu,
  _FloatWindow sigmaSqr,
  float thsigma,
  _FloatWindow mask
){
  int i,j;
  float ct=0;
  int n=width*height;
  float musqr_mu, musqr_sigma;
  getSqrMAD(n, mu, &musqr_mu, &musqr_sigma);

  for (j=0; j<height; ++j){
    for (i=0; i<width; ++i){
      if (*mask!=0 && (*sigmaSqr>thsigma*thsigma))
      {
        *mask=0;
      }
      //if (*mask==0 && *sigmaSqr<2 && fabs(*mu)<maxres/8){
      //  *mask=255;
      //}
      ct+=(*mask==0);
      ++mu;
      ++sigmaSqr;
      ++mask;
    }
  }
  return ct/width/height;
}

static float updateMaskX84(
  int width, int height, 
  _FloatWindow img,
  _FloatWindow mask
){
  float ct=0;
  _FloatWindow p=img;
  int i, n=width*height;
  float mu, sigma;
  getSqrMAD(n, img, &mu, &sigma);

  for (i=0; i<n; ++i){
    double val=(*p) * (*p);
    double minth=10000;
    double threshold=4*sigma;
    if (threshold<minth) 
      threshold=minth;
    if (fabs(val-mu)>threshold){
      *mask=0;
      ++ct;
    } else{
      *mask=255;
    }
    ++p;
    ++mask;
  }
  
  return ct/n;
}


static float combineMasks(
  int width, int height, 
  _FloatWindow mask1,
  _FloatWindow mask2
){
  int n=width*height;
  int i;
  float ct=0;
  for (i=0; i<n; ++i){
    if (*mask1==0){
      *mask2=0;
      ++ct;
    }
    ++mask1;
    ++mask2;
  }
  return ct/n;
}

static float _findMaskedResidual(
  _FloatWindow fw,
  _FloatWindow mask,
  int width,
  int height)
{
  const double ceiling2=256*256;
  double sum = 0.0;
  int w;
  int N=0;

  for ( ; height > 0 ; height--)
    for (w=0 ; w < width ; w++){
      if (*mask!=0){
        double mag2=(*fw)*(*fw);
        if (mag2>ceiling2)
          mag2=ceiling2;
        sum += mag2;
        //sum += (*fw)*(*fw);
        //sum += fabs(*fw);
        ++N;
      }
      ++mask;
      ++fw;
    }
  return (float)sqrt(sum/N);
  //return sum/N;
}



static void fillBuf(float *p, int n, float val){
  float* q=p;
  while (q<p+n) 
    *q++=val;
}



/*********************************************************************
 * _interpolate
 * 
 * Given a point (x,y) in an image, computes the bilinear interpolated 
 * gray-level value of the point in the image.  
 */

static float _interpolate(
  float x, 
  float y, 
  _KLT_FloatImage img)
{
  int xt = (int) x;  /* coordinates of top-left corner */
  int yt = (int) y;
  float ax = x - xt;
  float ay = y - yt;
  float *ptr = img->data + (img->ncols*yt) + xt;

#ifndef NDEBUG //ss
  if (xt<0 || yt<0 || xt>=img->ncols-1 || yt>=img->nrows-1) {
    fprintf(stderr, "(xt,yt)=(%d,%d)  imgsize=(%d,%d)\n"
            "(x,y)=(%f,%f)  (ax,ay)=(%f,%f)\n",
            xt, yt, img->ncols, img->nrows, x, y, ax, ay);
    fflush(stderr);
  }
#endif

  assert (xt >= 0 && yt >= 0 && xt <= img->ncols - 2 && yt <= img->nrows - 2);

  return ( (1-ax) * (1-ay) * *ptr +
           ax   * (1-ay) * *(ptr+1) +
           (1-ax) *   ay   * *(ptr+(img->ncols)) +
           ax   *   ay   * *(ptr+(img->ncols)+1) );
}


/*********************************************************************
 * _computeIntensityDifference
 *
 * Given two images and the window center in both images,
 * aligns the images wrt the window and computes the difference 
 * between the two overlaid images.
 */

static void _computeIntensityDifference(
  _KLT_FloatImage img1,   /* images */
  _KLT_FloatImage img2,
  float x1, float y1,     /* center of window in 1st img */
  float x2, float y2,     /* center of window in 2nd img */
  int width, int height,  /* size of window */
  _FloatWindow imgdiff)   /* output */
{
  register int hw = width/2, hh = height/2;
  float g1, g2;
  register int i, j;

  /* Compute values */
  for (j = -hh ; j <= hh ; j++)
    for (i = -hw ; i <= hw ; i++)  {
      g1 = _interpolate(x1+i, y1+j, img1);
      g2 = _interpolate(x2+i, y2+j, img2);
      *imgdiff++ = g1 - g2;
    }
}


/*********************************************************************
 * _computeGradientSum
 *
 * Given two gradients and the window center in both images,
 * aligns the gradients wrt the window and computes the sum of the two 
 * overlaid gradients.
 */

static void _computeGradientSum(
  _KLT_FloatImage gradx1,  /* gradient images */
  _KLT_FloatImage grady1,
  _KLT_FloatImage gradx2,
  _KLT_FloatImage grady2,
  float x1, float y1,      /* center of window in 1st img */
  float x2, float y2,      /* center of window in 2nd img */
  int width, int height,   /* size of window */
  _FloatWindow gradx,      /* output */
  _FloatWindow grady)      /*   " */
{
  register int hw = width/2, hh = height/2;
  float g1, g2;
  register int i, j;

  /* Compute values */
  for (j = -hh ; j <= hh ; j++)
    for (i = -hw ; i <= hw ; i++)  {
      g1 = _interpolate(x1+i, y1+j, gradx1);
      g2 = _interpolate(x2+i, y2+j, gradx2);
      *gradx++ = g1 + g2;
      g1 = _interpolate(x1+i, y1+j, grady1);
      g2 = _interpolate(x2+i, y2+j, grady2);
      *grady++ = g1 + g2;
    }
}


/*********************************************************************
 * _compute2by2GradientMatrix
 *
 */

static void _compute2by2GradientMatrix(
  _FloatWindow gradx,
  _FloatWindow grady,
  int width,   /* size of window */
  int height,
  float *gxx,  /* return values */
  float *gxy, 
  float *gyy) 

{
  register float gx, gy;
  register int i;

  /* Compute values */
  *gxx = 0.0;  *gxy = 0.0;  *gyy = 0.0;
  for (i = 0 ; i < width * height ; i++)  {
    gx = *gradx++;
    gy = *grady++;
    *gxx += gx*gx;
    *gxy += gx*gy;
    *gyy += gy*gy;
  }
}
	
	
/*********************************************************************
 * _compute2by1ErrorVector
 *
 */

static void _compute2by1ErrorVector(
  _FloatWindow imgdiff,
  _FloatWindow gradx,
  _FloatWindow grady,
  int width,   /* size of window */
  int height,
  float *ex,   /* return values */
  float *ey)
{
  register float diff;
  register int i;

  /* Compute values */
  *ex = 0;  *ey = 0;  
  for (i = 0 ; i < width * height ; i++)  {
    diff = *imgdiff++;
    *ex += diff * (*gradx++);
    *ey += diff * (*grady++);
  }
}


/*********************************************************************
 * _solveEquation
 *
 * Solves the 2x2 matrix equation
 *         [gxx gxy] [dx] = [ex]
 *         [gxy gyy] [dy] = [ey]
 * for dx and dy.
 *
 * Returns KLT_TRACKED on success and KLT_SMALL_DET on failure
 */

static int _solveEquation(
  float gxx, float gxy, float gyy,
  float ex, float ey,
  float small,
  float *dx, float *dy)
{
  float det = gxx*gyy - gxy*gxy;

  if (det < small){
    *dx=*dy=0;
    return KLT_SMALL_DET;
  }

  *dx = (gyy*ex - gxy*ey)/det;
  *dy = (gxx*ey - gxy*ex)/det;
  return KLT_TRACKED;
}


/*********************************************************************
 * _allocateFloatWindow
 */
	
static _FloatWindow _allocateFloatWindow(
  int width,
  int height)
{
  _FloatWindow fw;

  fw = (_FloatWindow) malloc(width*height*sizeof(float));
  if (fw == NULL)  KLTError("(_allocateFloatWindow) Out of memory.");
  return fw;
}


/*********************************************************************
 * _printFloatWindow
 * (for debugging purposes)
 */

/*
static void _printFloatWindow(
  _FloatWindow fw,
  int width,
  int height)
{
  int i, j;

  fprintf(stderr, "\n");
  for (i = 0 ; i < width ; i++)  {
    for (j = 0 ; j < height ; j++)  {
      fprintf(stderr, "%6.1f ", *fw++);
    }
    fprintf(stderr, "\n");
  }
}
*/
	

/*********************************************************************
 * _findResidual (ex _sumAbsFloatWindow)
 * (alternative = standard deviation!)
 */

static float _findResidual(
  _FloatWindow fw,
  int width,
  int height)
{
  float sum = 0.0;
  int w;
  int N=width*height;

  for ( ; height > 0 ; height--)
    for (w=0 ; w < width ; w++){
      sum += (*fw)*(*fw);
      //sum += fabs(*fw);
      ++fw;
    }
  return (float)sqrt(sum/N);
  //return sqrt(sum)/N;
  //return sum/N;
}


/*********************************************************************
 * _trackFeature
 *
 * Tracks a feature point from one image to the next.
 *
 * RETURNS
 * KLT_SMALL_DET if feature is lost,
 * KLT_MAX_ITERATIONS if tracking stopped because iterations timed out,
 * KLT_TRACKED otherwise.
 */

static int _trackFeature(
  float x1,  /* location of window in first image */
  float y1,
  float *x2, /* starting location of search in second image */
  float *y2,
  _KLT_FloatImage img1, 
  _KLT_FloatImage gradx1,
  _KLT_FloatImage grady1,
  _KLT_FloatImage img2, 
  _KLT_FloatImage gradx2,
  _KLT_FloatImage grady2,
  int width,           /* size of window */
  int height,
  int max_iterations,
  float small,         /* determinant threshold for declaring KLT_SMALL_DET */
  float th,            /* displacement threshold for stopping               */
  float max_residue,   /* residue threshold for declaring KLT_LARGE_RESIDUE */
  float *lambda2,   //ss 
  float *residual)     //ss
{
  _FloatWindow imgdiff, gradx, grady;
  float gxx, gxy, gyy, ex, ey, dx, dy;
  int iteration = 0;
  int status;
  int hw = width/2;
  int hh = height/2;
  int nc = img1->ncols;
  int nr = img1->nrows;
  float one_plus_eps = 1.000001f;   /* To prevent rounding errors */


  /* Allocate memory for windows */
  imgdiff = _allocateFloatWindow(width, height);
  gradx   = _allocateFloatWindow(width, height);
  grady   = _allocateFloatWindow(width, height);

  /* Iteratively update the window position */
  do  {

    /* If out of bounds, exit loop */
    if ( x1-hw < 0.0f ||  x1+hw > nc-one_plus_eps ||
         *x2-hw < 0.0f || *x2+hw > nc-one_plus_eps ||
         y1-hh < 0.0f ||  y1+hh > nr-one_plus_eps ||
         *y2-hh < 0.0f || *y2+hh > nr-one_plus_eps) {
      status = KLT_OOB;
      break;
    }

    /* Compute gradient and difference windows */
    _computeIntensityDifference(img1, img2, x1, y1, *x2, *y2, 
                                width, height, imgdiff);

    _computeGradientSum(gradx1, grady1, gradx2, grady2, 
      x1, y1, *x2, *y2, width, height, gradx, grady);


    /* Use these windows to construct matrices */
    _compute2by2GradientMatrix(gradx, grady, width, height, 
                               &gxx, &gxy, &gyy);
    _compute2by1ErrorVector(imgdiff, gradx, grady, width, height,
                            &ex, &ey);

    /* Using matrices, solve equation for new displacement */
    status = _solveEquation(gxx, gxy, gyy, ex, ey, small, &dx, &dy);
    *lambda2=(float)((gxx+gyy-sqrt(4*gxy*gxy+(gxx-gyy)*(gxx-gyy)))/(2*width*height));

    /// fprintf(stderr, "####  It:%d, dx=(%f,%f), l2=%f, r=%f\n",
    ///    iteration, dx,dy, *lambda2,
    ///    _findResidual(imgdiff, width, height)); //ss diag

    if (status == KLT_SMALL_DET){
      /// fprintf(stderr, "####  Got a small det!\n"); //ss diag
      status=KLT_TRACKED;      
      break;
    }

    *x2 += dx;
    *y2 += dy;
    iteration++;

  }  while ((fabs(dx)>=th || fabs(dy)>=th) && iteration < max_iterations);
  

  /* Check whether window is out of bounds */
  if (*x2-hw < 0.0f || *x2+hw > nc-one_plus_eps || 
      *y2-hh < 0.0f || *y2+hh > nr-one_plus_eps)
    status = KLT_OOB;
    /* Check whether residue is too large */
  if (status == KLT_TRACKED)  {
    _computeIntensityDifference(img1, img2, x1, y1, *x2, *y2, 
                                width, height, imgdiff);
    *residual=_findResidual(imgdiff, width, height);//ss
    if (*residual>max_residue)
      status = KLT_LARGE_RESIDUE;
  }

  /* Free memory */
  free(imgdiff);  free(gradx);  free(grady);

  /* Return appropriate value */
  if (status == KLT_SMALL_DET)  return KLT_SMALL_DET;
  else if (status == KLT_OOB)  return KLT_OOB;
  else if (status == KLT_LARGE_RESIDUE)  return KLT_LARGE_RESIDUE;
  else if (iteration >= max_iterations)  return KLT_MAX_ITERATIONS;
  else  return KLT_TRACKED;

}


/*********************************************************************/

static KLT_BOOL _outOfBounds(
  float x,
  float y,
  int ncols,
  int nrows,
  int borderx,
  int bordery)
{
  return (x < borderx || x > ncols-1-borderx ||
          y < bordery || y > nrows-1-bordery );
}




/********************************************************************** 
* CONSISTENCY CHECK OF FEATURES BY AFFINE MAPPING (BEGIN)
* 
* Created by: Thorsten Thormaehlen (University of Hannover) June 2004    
* thormae@tnt.uni-hannover.de
* 
* Permission is granted to any individual or institution to use, copy, modify,
* and distribute this part of the software, provided that this complete authorship 
* and permission notice is maintained, intact, in all copies. 
*
* This software is provided  "as is" without express or implied warranty.
*
*
* The following static functions are helpers for the affine mapping.
* They all start with "_am". 
* There are also small changes in other files for the
* affine mapping these are all marked by "for affine mapping"
* 
* Thanks to Kevin Koeser (koeser@mip.informatik.uni-kiel.de) for fixing a bug 
*/

#define SWAP_ME(X,Y) {temp=(X);(X)=(Y);(Y)=temp;}

static float **_am_matrix(long nr, long nc)
{
  float **m;
  int a;
  m = (float **) malloc((size_t)(nr*sizeof(float*)));
  m[0] = (float *) malloc((size_t)((nr*nc)*sizeof(float)));
  for(a = 1; a < nr; a++) m[a] = m[a-1]+nc;
  return m;
}

static void _am_free_matrix(float **m)
{
  free(m[0]);
  free(m);
}


static int _am_gauss_jordan_elimination(float **a, int n, float **b, int m)
{
  /* re-implemented from Numerical Recipes in C */
  int *indxc,*indxr,*ipiv;
  int i,j,k,l,ll;
  float big,dum,pivinv,temp;
  int col = 0;
  int row = 0;

  indxc=(int *)malloc((size_t) (n*sizeof(int)));
  indxr=(int *)malloc((size_t) (n*sizeof(int)));
  ipiv=(int *)malloc((size_t) (n*sizeof(int)));
  for (j=0;j<n;j++) ipiv[j]=0;
  for (i=0;i<n;i++) {
    big=0.0;
    for (j=0;j<n;j++)
      if (ipiv[j] != 1)
        for (k=0;k<n;k++) {
          if (ipiv[k] == 0) {
            if (fabs(a[j][k]) >= big) {
              big=(float)fabs(a[j][k]);
              row=j;
              col=k;
            }
          } else if (ipiv[k] > 1) return KLT_SMALL_DET;
        }
    ++(ipiv[col]);
    if (row != col) {
      for (l=0;l<n;l++) SWAP_ME(a[row][l],a[col][l])
      for (l=0;l<m;l++) SWAP_ME(b[row][l],b[col][l])
    }
    indxr[i]=row;
    indxc[i]=col;
    if (a[col][col] == 0.0) return KLT_SMALL_DET;
    pivinv=1.0f/a[col][col];
    a[col][col]=1.0;
    for (l=0;l<n;l++) a[col][l] *= pivinv;
    for (l=0;l<m;l++) b[col][l] *= pivinv;
    for (ll=0;ll<n;ll++)
      if (ll != col) {
        dum=a[ll][col];
        a[ll][col]=0.0;
        for (l=0;l<n;l++) a[ll][l] -= a[col][l]*dum;
        for (l=0;l<m;l++) b[ll][l] -= b[col][l]*dum;
      }
  }
  for (l=n-1;l>=0;l--) {
    if (indxr[l] != indxc[l])
      for (k=0;k<n;k++)
        SWAP_ME(a[k][indxr[l]],a[k][indxc[l]]);
  }
  free(ipiv);
  free(indxr);
  free(indxc);

  return KLT_TRACKED;
}

/*********************************************************************
 * _am_getGradientWinAffine
 *
 * aligns the gradients with the affine transformed window 
 */

static void _am_getGradientWinAffine(
  _KLT_FloatImage in_gradx,
  _KLT_FloatImage in_grady,
  float x, float y,      /* center of window*/
  float Axx, float Ayx , float Axy, float Ayy,    /* affine mapping */
  int width, int height,   /* size of window */
  _FloatWindow out_gradx,      /* output */
  _FloatWindow out_grady)      /* output */
{
  register int hw = width/2, hh = height/2;
  register int i, j;
  float mi, mj;
 
  /* Compute values */
  for (j = -hh ; j <= hh ; j++)
    for (i = -hw ; i <= hw ; i++)  {
      mi = Axx * i + Axy * j;
      mj = Ayx * i + Ayy * j;
      *out_gradx++ = _interpolate(x+mi, y+mj, in_gradx);
      *out_grady++ = _interpolate(x+mi, y+mj, in_grady);
    }
  
}

/*********************************************************************
 * _computeAffineMappedImage
 * used only for DEBUG output
 *     
*/

static void _am_computeAffineMappedImage(
  _KLT_FloatImage img,   /* images */
  float x, float y,      /* center of window  */
  float Axx, float Ayx , float Axy, float Ayy,    /* affine mapping */   
  float lambda, float delta,
  int width, int height,  /* size of window */
  _FloatWindow imgdiff)   /* output */
{
  register int hw = width/2, hh = height/2;
  register int i, j;
  float mi, mj;

  /* Compute values */
  for (j = -hh ; j <= hh ; j++)
    for (i = -hw ; i <= hw ; i++){
      mi = Axx * i + Axy * j;
      mj = Ayx * i + Ayy * j;
      *imgdiff++ = delta+lambda*_interpolate(x+mi, y+mj, img);
    }
}


/*********************************************************************
 * _getSubFloatImage
 */

static void _am_getSubFloatImage(
  _KLT_FloatImage img,   /* image */
  float x, float y,     /* center of window */
  _KLT_FloatImage window)   /* output */
{
  register int hw = window->ncols/2, hh = window->nrows/2;
  int x0 = (int) x;
  int y0 = (int) y;
  float * windata = window->data; 
  int offset;
  register int i, j;

  assert(x0 - hw >= 0);
  assert(y0 - hh >= 0);
  assert(x0 + hw <= img->ncols);
  assert(y0 + hh <= img->nrows); 

  /* copy values */
  for (j = -hh ; j <= hh ; j++)
    for (i = -hw ; i <= hw ; i++)  {
      offset = (j+y0)*img->ncols + (i+x0);
      *windata++ = *(img->data+offset);
    }
}

/*********************************************************************
 * _am_computeIntensityDifferenceAffine
 *
 * Given two images and the window center in both images,
 * aligns the images with the window and computes the difference 
 * between the two overlaid images using the affine mapping.
 *       A =  [ Axx Axy]
 *            [ Ayx Ayy]        
*/

static void _am_computeIntensityDifferenceAffine(
  _KLT_FloatImage img1,   /* images */
  _KLT_FloatImage img2,
  float x1, float y1,     /* center of window in 1st img */
  float x2, float y2,      /* center of window in 2nd img */
  float Axx, float Ayx , float Axy, float Ayy,    /* affine mapping */   
  float lambda, float delta,
  int width, int height,  /* size of window */
  _FloatWindow imgdiff)   /* output */
{
  register int hw = width/2, hh = height/2;
  float g1, g2;
  register int i, j;
  float mi, mj;

  /* Compute values */
  for (j = -hh ; j <= hh ; j++)
    for (i = -hw ; i <= hw ; i++)  {
      g1 = _interpolate(x1+i, y1+j, img1);
      mi = Axx * i + Axy * j;
      mj = Ayx * i + Ayy * j;
      g2 = lambda*_interpolate(x2+mi, y2+mj, img2)+delta;
      *imgdiff++ = g1 - g2;
    }
}


/*********************************************************************
 * ss affine+contrast compensation
 *
 */


static void _am_compute8by1ErrorVector(
  _FloatWindow imgdiff,
  _FloatWindow img2,
  float lambda,
  _FloatWindow gradx,
  _FloatWindow grady,
  int width,   /* size of window */
  int height,
  float **e)  /* return values */
{
  register int hw = width/2, hh = height/2;
  register int i, j;
  register float diff, gx,gy, i2;

  /* Set values to zero */  
  for(i = 0; i < 8; i++) e[i][0] = 0.0; 
  
  /* Compute values */
  for (j = -hh ; j <= hh ; j++) {
    for (i = -hw ; i <= hw ; i++)  {
      diff = *imgdiff++;
      gx = *gradx++;
      gy = *grady++;
      i2=*img2++;
      e[0][0] += diff*lambda*gx * i;
      e[1][0] += diff*lambda*gy * i;
      e[2][0] += diff*lambda*gx * j; 
      e[3][0] += diff*lambda*gy * j; 
      e[4][0] += diff*lambda*gx;
      e[5][0] += diff*lambda*gy; 
      e[6][0] += diff*i2;
      e[7][0] += diff*1;
    }
  }
  
  for(i = 0; i < 8; i++) e[i][0] *= 0.5;
  
}
static void _am_compute8by8GradientMatrix(
  _FloatWindow img2,
  float lambda,
  _FloatWindow gradx,
  _FloatWindow grady,
  int width,   /* size of window */
  int height,
  float **T)  /* return values */
{
  register int hw = width/2, hh = height/2;
  register int i, j;
  float gx, gy, gxx, gxy, gyy,  x, y, xx, xy, yy, i2;
 
  
  /* Set values to zero */ 
  for (j = 0 ; j < 8 ; j++)  {
    for (i = j ; i < 8 ; i++)  {
      T[j][i] = 0.0;
    }
  }
  
  for (j = -hh ; j <= hh ; j++) {
    for (i = -hw ; i <= hw ; i++)  {
      i2=*img2++;
      gx = *gradx++;
      gy = *grady++;
      gxx = gx * gx;
      gxy = gx * gy;
      gyy = gy * gy;
      x = (float) i; 
      y = (float) j; 
      xx = x * x;
      xy = x * y;
      yy = y * y;
      
      T[0][0] += lambda*lambda* xx * gxx; 
      T[0][1] += lambda*lambda* xx * gxy;
      T[0][2] += lambda*lambda* xy * gxx;
      T[0][3] += lambda*lambda* xy * gxy;
      T[0][4] += lambda*lambda* x  * gxx;
      T[0][5] += lambda*lambda* x  * gxy;
      T[0][6] += lambda* x  * gx * i2;
      T[0][7] += lambda* x  * gx;

      T[1][1] += lambda*lambda* xx * gyy;
      T[1][2] += lambda*lambda* xy * gxy;
      T[1][3] += lambda*lambda* xy * gyy;
      T[1][4] += lambda*lambda* x  * gxy;
      T[1][5] += lambda*lambda* x  * gyy;
      T[1][6] += lambda* x  * gy * i2;
      T[1][7] += lambda* x  * gy;
 
      T[2][2] += lambda*lambda* yy * gxx;
      T[2][3] += lambda*lambda* yy * gxy;
      T[2][4] += lambda*lambda* y  * gxx;
      T[2][5] += lambda*lambda* y  * gxy;
      T[2][6] += lambda* y  * gx * i2;
      T[2][7] += lambda* y  * gx;
 
      T[3][3] += lambda*lambda* yy * gyy;
      T[3][4] += lambda*lambda* y  * gxy;
      T[3][5] += lambda*lambda* y  * gyy; 
      T[3][6] += lambda* y  * gy * i2;
      T[3][7] += lambda* y  * gy;

      T[4][4] += lambda*lambda* gxx; 
      T[4][5] += lambda*lambda* gxy;
      T[4][6] += lambda* gx * i2;
      T[4][7] += lambda* gx;
      
      T[5][5] += lambda*lambda* gyy; 
      T[5][6] += lambda* gy * i2;
      T[5][7] += lambda* gy;

      T[6][6] += i2 * i2;
      T[6][7] += i2;

      T[7][7] += 1;
    }
  }
  
  for (j = 0 ; j < 7 ; j++)  {
    for (i = j+1 ; i < 8 ; i++)  {
      T[i][j] = T[j][i];
    }
  }

}




/*********************************************************************
 * ss scale compensation
 *
 */

static void _am_compute3by1ErrorVector(//ss
  _FloatWindow mask,
  _FloatWindow gauss,
  _FloatWindow imgdiff,
  _FloatWindow gradx,
  _FloatWindow grady,
  int width,   /* size of window */
  int height,
  float **e)  /* return values */
{
  register int hw = width/2, hh = height/2;
  register int i, j;
  register float diff, gx, gy, m, w;

  /* Set values to zero */  
  for(i = 0; i < 3; i++) e[i][0] = 0.0; 
  
  /* Compute values */
  for (j = -hh ; j <= hh ; j++) {
    for (i = -hw ; i <= hw ; i++)  {
      gx = *gradx++;
      gy = *grady++;
      diff = *imgdiff++;
      m=*mask++;
      w=*gauss++;
      if (m!=0){
        e[0][0] += diff*gx*i + diff*gy*j *w;
        e[1][0] += diff*gx *w;
        e[2][0] += diff*gy *w;
      }
    }
  }
  for(i = 0; i < 3; i++) e[i][0] *= 0.5;
}
static void _am_compute3by3GradientMatrix(
  _FloatWindow mask,
  _FloatWindow gauss,
  _FloatWindow gradx,
  _FloatWindow grady,
  int width,   /* size of window */
  int height,
  float **T)  /* return values */
{
  register int hw = width/2, hh = height/2;
  register int i, j;
  float gx, gy, m, w;
 
  
  /* Set values to zero */ 
  for (j = 0 ; j < 3 ; j++)  {
    for (i = 0 ; i < 3 ; i++)  {
      T[j][i] = 0.0;
    }
  }
  
  for (j = -hh ; j <= hh ; j++) {
    for (i = -hw ; i <= hw ; i++)  {
      gx = *gradx++;
      gy = *grady++;
      m=*mask++;
      w=*gauss++;
      if (m!=0){
        T[0][0] += (i*gx+j*gy) * (i*gx+j*gy) * w;
        T[0][1] += (i*gx+j*gy)*gx * w;
        T[0][2] += (i*gx+j*gy)*gy * w;

        T[1][1] += gx*gx * w;
        T[1][2] += gx*gy * w;

        T[2][2] += gy*gy * w;
      }
    }
  }
  
  for (j = 0 ; j < 3 ; j++)  {
    for (i = j+1 ; i < 3 ; i++)  {
      T[i][j] = T[j][i];
    }
  }

}




/*********************************************************************
 * ss contrast+scale+background compensation
 */

static void _am_compute2x2Masked(
  _FloatWindow mask,
  _FloatWindow gradx,
  _FloatWindow grady,
  int width,   /* size of window */
  int height,
  float *gxx,
  float *gxy,
  float *gyy)  /* return values */
{
  register int hw = width/2, hh = height/2;
  register int i, j;
 
  *gxx=0;*gyy=0;*gxy=0;
  for (j = -hh ; j <= hh ; j++) {
    for (i = -hw ; i <= hw ; i++)  {
      float gx = *gradx++;
      float gy = *grady++;
      float m=*mask++;
      if (m!=0){
        *gxx += gx*gx;
        *gyy += gy*gy;
        *gxy += gx*gy;
      }
    }
  }
}

static void _am_compute5by1ErrorVector(//ss
  _FloatWindow mask,
  _FloatWindow gauss,
  _FloatWindow imgdiff,
  _FloatWindow img2,
  float lambda,
  _FloatWindow gradx,
  _FloatWindow grady,
  int width,   /* size of window */
  int height,
  float **e)  /* return values */
{
  register int hw = width/2, hh = height/2;
  register int i, j;
  register float diff, gx,gy, i2, m, w;

  /* Set values to zero */  
  for(i = 0; i < 5; i++) e[i][0] = 0.0; 
  
  /* Compute values */
  for (j = -hh ; j <= hh ; j++) {
    for (i = -hw ; i <= hw ; i++)  {
      diff = *imgdiff++;
      gx = *gradx++;
      gy = *grady++;
      i2=*img2++;
      m=*mask++;
      w=*gauss++;
      if (m!=0){
        e[0][0] += diff*lambda*(gx*i + gy*j) *w;
        e[1][0] += diff*lambda*gx *w;
        e[2][0] += diff*lambda*gy *w; 
        e[3][0] += diff*i2 *w;
        e[4][0] += diff*1 *w;
      }
    }
  }
  for(i = 0; i < 5; i++) e[i][0] *= 0.5;
}
static void _am_compute5by5GradientMatrix(
  _FloatWindow mask,
  _FloatWindow gauss,
  _FloatWindow img2,
  float lambda,
  _FloatWindow gradx,
  _FloatWindow grady,
  int width,   /* size of window */
  int height,
  float **T)  /* return values */
{
  register int hw = width/2, hh = height/2;
  register int i, j;
  float gx, gy, x, y, i2, m, w;
 
  
  /* Set values to zero */ 
  for (j = 0 ; j < 5 ; j++)  {
    for (i = 0 ; i < 5 ; i++)  {
      T[j][i] = 0.0;
    }
  }
  
  for (j = -hh ; j <= hh ; j++) {
    for (i = -hw ; i <= hw ; i++)  {
      i2=*img2++;
      gx = *gradx++;
      gy = *grady++;
      m=*mask++;
      w=*gauss++;
      if (m!=0){
        x = (float) i; 
        y = (float) j; 
        T[0][0] += lambda*lambda*(x*gx+y*gy) * (x*gx+y*gy) *w;
        T[0][1] += lambda*lambda*(x*gx+y*gy)*gx *w;
        T[0][2] += lambda*lambda*(x*gx+y*gy)*gy *w;
        T[0][3] += lambda* (x*gx+y*gy) * i2 *w;
        T[0][4] += lambda* (x*gx+y*gy) *w;

        T[1][1] += lambda*lambda*gx*gx *w;
        T[1][2] += lambda*lambda*gx*gy *w;
        T[1][3] += lambda* gx * i2 *w;
        T[1][4] += lambda* gx *w;

        T[2][2] += lambda*lambda*gy*gy *w;
        T[2][3] += lambda* gy * i2 *w;
        T[2][4] += lambda* gy *w;

        T[3][3] += i2 * i2 *w;
        T[3][4] += i2 *w;

        T[4][4] += 1 *w;
      }
    }
  }
  
  for (j = 0 ; j < 5 ; j++)  {
    for (i = j+1 ; i < 5 ; i++)  {
      T[i][j] = T[j][i];
    }
  }

}


/*********************************************************************
 * _am_compute6by6GradientMatrix
 *
 */

static void _am_compute6by6GradientMatrix(
  _FloatWindow gradx,
  _FloatWindow grady,
  int width,   /* size of window */
  int height,
  float **T)  /* return values */
{
  register int hw = width/2, hh = height/2;
  register int i, j;
  float gx, gy, gxx, gxy, gyy,  x, y, xx, xy, yy;
 
  
  /* Set values to zero */ 
  for (j = 0 ; j < 6 ; j++)  {
    for (i = j ; i < 6 ; i++)  {
      T[j][i] = 0.0;
    }
  }
  
  for (j = -hh ; j <= hh ; j++) {
    for (i = -hw ; i <= hw ; i++)  {
      gx = *gradx++;
      gy = *grady++;
      gxx = gx * gx;
      gxy = gx * gy;
      gyy = gy * gy;
      x = (float) i; 
      y = (float) j; 
      xx = x * x;
      xy = x * y;
      yy = y * y;
      
      T[0][0] += xx * gxx; 
      T[0][1] += xx * gxy;
      T[0][2] += xy * gxx;
      T[0][3] += xy * gxy;
      T[0][4] += x  * gxx;
      T[0][5] += x  * gxy;

      T[1][1] += xx * gyy;
      T[1][2] += xy * gxy;
      T[1][3] += xy * gyy;
      T[1][4] += x  * gxy;
      T[1][5] += x  * gyy;
 
      T[2][2] += yy * gxx;
      T[2][3] += yy * gxy;
      T[2][4] += y  * gxx;
      T[2][5] += y  * gxy;
 
      T[3][3] += yy * gyy;
      T[3][4] += y  * gxy;
      T[3][5] += y  * gyy; 

      T[4][4] += gxx; 
      T[4][5] += gxy;
      
      T[5][5] += gyy; 
    }
  }
  
  for (j = 0 ; j < 5 ; j++)  {
    for (i = j+1 ; i < 6 ; i++)  {
      T[i][j] = T[j][i];
    }
  }

}



/*********************************************************************
 * _am_compute6by1ErrorVector
 *
 */

static void _am_compute6by1ErrorVector(
  _FloatWindow imgdiff,
  _FloatWindow gradx,
  _FloatWindow grady,
  int width,   /* size of window */
  int height,
  float **e)  /* return values */
{
  register int hw = width/2, hh = height/2;
  register int i, j;
  register float diff,  diffgradx,  diffgrady;

  /* Set values to zero */  
  for(i = 0; i < 6; i++) e[i][0] = 0.0; 
  
  /* Compute values */
  for (j = -hh ; j <= hh ; j++) {
    for (i = -hw ; i <= hw ; i++)  {
      diff = *imgdiff++;
      diffgradx = diff * (*gradx++);
      diffgrady = diff * (*grady++);
      e[0][0] += diffgradx * i;
      e[1][0] += diffgrady * i;
      e[2][0] += diffgradx * j; 
      e[3][0] += diffgrady * j; 
      e[4][0] += diffgradx;
      e[5][0] += diffgrady; 
    }
  }
  
  for(i = 0; i < 6; i++) e[i][0] *= 0.5;
  
}


/*********************************************************************
 * _am_compute4by4GradientMatrix
 *
 */

static void _am_compute4by4GradientMatrix(
  _FloatWindow gradx,
  _FloatWindow grady,
  int width,   /* size of window */
  int height,
  float **T)  /* return values */
{
  register int hw = width/2, hh = height/2;
  register int i, j;
  float gx, gy, x, y;
 
  
  /* Set values to zero */ 
  for (j = 0 ; j < 4 ; j++)  {
    for (i = 0 ; i < 4 ; i++)  {
      T[j][i] = 0.0;
    }
  }
  
  for (j = -hh ; j <= hh ; j++) {
    for (i = -hw ; i <= hw ; i++)  {
      gx = *gradx++;
      gy = *grady++;
      x = (float) i; 
      y = (float) j; 
      T[0][0] += (x*gx+y*gy) * (x*gx+y*gy);
      T[0][1] += (x*gx+y*gy)*(x*gy-y*gx);
      T[0][2] += (x*gx+y*gy)*gx;
      T[0][3] += (x*gx+y*gy)*gy;
   
      T[1][1] += (x*gy-y*gx) * (x*gy-y*gx);
      T[1][2] += (x*gy-y*gx)*gx;
      T[1][3] += (x*gy-y*gx)*gy;
     
      T[2][2] += gx*gx;
      T[2][3] += gx*gy;
      
      T[3][3] += gy*gy;
    }
  }
  
  for (j = 0 ; j < 3 ; j++)  {
    for (i = j+1 ; i < 4 ; i++)  {
      T[i][j] = T[j][i];
    }
  }

}

/*********************************************************************
 * _am_compute4by1ErrorVector
 *
 */

static void _am_compute4by1ErrorVector(
  _FloatWindow imgdiff,
  _FloatWindow gradx,
  _FloatWindow grady,
  int width,   /* size of window */
  int height,
  float **e)  /* return values */
{
  register int hw = width/2, hh = height/2;
  register int i, j;
  register float diff,  diffgradx,  diffgrady;

  /* Set values to zero */  
  for(i = 0; i < 4; i++) e[i][0] = 0.0; 
  
  /* Compute values */
  for (j = -hh ; j <= hh ; j++) {
    for (i = -hw ; i <= hw ; i++)  {
      diff = *imgdiff++;
      diffgradx = diff * (*gradx++);
      diffgrady = diff * (*grady++);
      e[0][0] += diffgradx * i + diffgrady * j;
      e[1][0] += diffgrady * i - diffgradx * j;
      e[2][0] += diffgradx;
      e[3][0] += diffgrady;
    }
  }
  
  for(i = 0; i < 4; i++) e[i][0] *= 0.5;
  
}





/*********************************************************************
 * _am_trackFeatureAffine
 *
 * Tracks a feature point from the image of first occurrence to the actual image.
 *
 * RETURNS
 * KLT_SMALL_DET or KLT_LARGE_RESIDUE or KLT_OOB if feature is lost,
 * KLT_TRACKED otherwise.
 */

/* if you enalbe the DEBUG_AFFINE_MAPPING make sure you have created a directory "./debug" */
/* #define DEBUG_AFFINE_MAPPING */

#ifdef DEBUG_AFFINE_MAPPING
static int counter = 0;
static int glob_index = 0;
#endif

static int _am_trackFeatureAffine(
      float x1,  /* location of window in first image */
      float y1,
      float *x2, /* starting location of search in second image */
      float *y2,
      _KLT_FloatImage img1, 
      _KLT_FloatImage gradx1,
      _KLT_FloatImage grady1,
      _KLT_FloatImage img2, 
      _KLT_FloatImage gradx2,
      _KLT_FloatImage grady2,
      int width,           /* size of window */
      int height,
      int max_iterations,
      float small,         /* determinant threshold for declaring KLT_SMALL_DET */
      float th,            /* displacement threshold for stopping  */
      float th_aff,
      float max_residue,   /* residue threshold for declaring KLT_LARGE_RESIDUE */
      int affine_map,      /* whether to evaluates the consistency of features with affine mapping */
      float mdd,           /* difference between the displacements */
      float *Axx, float *Ayx, 
      float *Axy, float *Ayy,     /* used for affine mapping */
      float *lambda, float *delta, //ss contrast compensation
      _KLT_FloatImage img_warp,    //ss 
      _KLT_FloatImage img_mask,    //ss 
      _KLT_FloatImage img_mu,      //ss 
      _KLT_FloatImage img_sigmaSqr,//ss 
      float *lambda2,              //ss
      float *residual_start,       //ss 
      float *residual_stop,        //ss 
      float bgmodel_alpha,         //ss 
      float bgmodel_thsigma,       //ss 
      float dxtrans,               //ss  
      float dytrans,               //ss 
      int borderx,
      int bordery,
      float* img_gauss,//ss 
      int index)       //ss 
{
  _FloatWindow imgdiff, gradx, grady, tmpmask1, tmpmask2;
  float gxx, gxy, gyy, ex, ey, dx, dy;
  int iteration = 0;
  int status = 0;
  int hw = width/2;
  int hh = height/2;
  int nc1 = img1->ncols;
  int nr1 = img1->nrows;
  int nc2 = img2->ncols;
  int nr2 = img2->nrows;
  float **a;
  float **T; 
  float one_plus_eps = 1.000001f;   /* To prevent rounding errors */
  float old_x2 = *x2;
  float old_y2 = *y2;
  float residual=0;
  KLT_BOOL convergence = FALSE;

  float old_ul_x =  *Axx * (-hw) + *Axy *   hh  + *x2;  /* upper left corner */
  float old_ul_y =  *Ayx * (-hw) + *Ayy *   hh  + *y2; 
  float old_ll_x =  *Axx * (-hw) + *Axy * (-hh) + *x2;  /* lower left corner */
  float old_ll_y =  *Ayx * (-hw) + *Ayy * (-hh) + *y2;
  float old_ur_x =  *Axx *   hw  + *Axy *   hh  + *x2;  /* upper right corner */
  float old_ur_y =  *Ayx *   hw  + *Ayy *   hh  + *y2;
  float old_lr_x =  *Axx *   hw  + *Axy * (-hh) + *x2;  /* lower right corner */
  float old_lr_y =  *Ayx *   hw  + *Ayy * (-hh) + *y2;
  //double old_mag  =  *Ayy + *Axx;

#ifdef DEBUG_AFFINE_MAPPING
  char fname[80];
  _KLT_FloatImage aff_diff_win = _KLTCreateFloatImage(width,height);
  printf("starting location x2=%f y2=%f\n", *x2, *y2);
#endif
  
  /* Allocate memory for windows */
  imgdiff = _allocateFloatWindow(width, height);
  gradx   = _allocateFloatWindow(width, height);
  grady   = _allocateFloatWindow(width, height);
  tmpmask1 = _allocateFloatWindow(width, height);
  tmpmask2 = _allocateFloatWindow(width, height);
  T = _am_matrix(8,8);
  a = _am_matrix(8,1);

  /* Iteratively update the window position */
  //if (index == 98){
  //  savegrey(img_mask);
  //}
  do  {
    if(!affine_map) {
      /* pure translation tracker */
      
      /* If out of bounds, exit loop */
      if ( x1-hw < 0.0f ||  x1+hw > nc1-one_plus_eps ||
          *x2-hw < 0.0f || *x2+hw > nc2-one_plus_eps ||
          y1-hh < 0.0f ||  y1+hh > nr1-one_plus_eps ||
          *y2-hh < 0.0f || *y2+hh > nr2-one_plus_eps) {
        status = KLT_OOB;
        break;
      }
      
      /* Compute gradient and difference windows */
      _computeIntensityDifference(img1, img2, x1, y1, *x2, *y2, 
        width, height, imgdiff);
      
#ifdef DEBUG_AFFINE_MAPPING	
      aff_diff_win->data = imgdiff;
      sprintf(fname, "./debug/kltimg_trans_diff_win%03d.%03d.pgm", glob_index, counter);
      printf("%s\n", fname);
      _KLTWriteAbsFloatImageToPGM(aff_diff_win, fname,256.0);
      printf("iter = %d translation tracker res: %f\n", iteration, _findResidual(imgdiff, width, height));
#endif
  
      _computeGradientSum(gradx1, grady1, gradx2, grady2, 
        x1, y1, *x2, *y2, width, height, gradx, grady);

      /* Use these windows to construct matrices */
      _compute2by2GradientMatrix(gradx, grady, width, height, 
        &gxx, &gxy, &gyy);
      _compute2by1ErrorVector(imgdiff, gradx, grady, width, height,
        &ex, &ey);

      /* Using matrices, solve equation for new displacement */
      status = _solveEquation(gxx, gxy, gyy, ex, ey, small, &dx, &dy);

      convergence = (fabs(dx) < th && fabs(dy) < th);
      
      *x2 += dx;
      *y2 += dy;
      
    }else{
      /* affine tracker */
      
      float ul_x =  *Axx * (-hw) + *Axy *   hh  + *x2;  /* upper left corner */
      float ul_y =  *Ayx * (-hw) + *Ayy *   hh  + *y2; 
      float ll_x =  *Axx * (-hw) + *Axy * (-hh) + *x2;  /* lower left corner */
      float ll_y =  *Ayx * (-hw) + *Ayy * (-hh) + *y2;
      float ur_x =  *Axx *   hw  + *Axy *   hh  + *x2;  /* upper right corner */
      float ur_y =  *Ayx *   hw  + *Ayy *   hh  + *y2;
      float lr_x =  *Axx *   hw  + *Axy * (-hh) + *x2;  /* lower right corner */
      float lr_y =  *Ayx *   hw  + *Ayy * (-hh) + *y2;

      /* If out of bounds, exit loop */
      if ( x1-hw < 0.0f ||  x1+hw > nc1-one_plus_eps ||
           y1-hh < 0.0f ||  y1+hh > nr1-one_plus_eps ||
           ul_x  < 0.0f ||  ul_x  > nc2-one_plus_eps ||
           ll_x  < 0.0f ||  ll_x  > nc2-one_plus_eps ||
           ur_x  < 0.0f ||  ur_x  > nc2-one_plus_eps ||
           lr_x  < 0.0f ||  lr_x  > nc2-one_plus_eps ||
           ul_y  < 0.0f ||  ul_y  > nr2-one_plus_eps ||
           ll_y  < 0.0f ||  ll_y  > nr2-one_plus_eps ||
           ur_y  < 0.0f ||  ur_y  > nr2-one_plus_eps ||
           lr_y  < 0.0f ||  lr_y  > nr2-one_plus_eps) {
        status = KLT_OOB;
        break;
      }

#ifdef DEBUG_AFFINE_MAPPING
      counter++;
      _am_computeAffineMappedImage(img1, x1, y1,  1.0, 0.0 , 0.0, 1.0, 1,0, width, height, imgdiff);
      aff_diff_win->data = imgdiff;
      sprintf(fname, "./debug/kltimg_aff_diff_win%03d.%03d_1.pgm", glob_index, counter);
      printf("%s\n", fname);
      _KLTWriteAbsFloatImageToPGM(aff_diff_win, fname,256.0);
      
      _am_computeAffineMappedImage(img2, *x2, *y2,  *Axx, *Ayx , *Axy, *Ayy, 1,0,width, height, imgdiff);
      aff_diff_win->data = imgdiff;
      sprintf(fname, "./debug/kltimg_aff_diff_win%03d.%03d_2.pgm", glob_index, counter);
      printf("%s\n", fname);
      _KLTWriteAbsFloatImageToPGM(aff_diff_win, fname,256.0);
#endif
      
      _am_computeIntensityDifferenceAffine(img1,img2, x1,y1, *x2,*y2,  
        *Axx,*Ayx,*Axy,*Ayy, *lambda,*delta,
        width, height, imgdiff);
      residual=_findResidual(imgdiff, width, height);//ss
      if (0==iteration) *residual_start=residual;
#ifdef DEBUG_AFFINE_MAPPING    
      aff_diff_win->data = imgdiff;
      sprintf(fname, "./debug/kltimg_aff_diff_win%03d.%03d_3.pgm", glob_index,counter);
      printf("%s\n", fname);
      _KLTWriteAbsFloatImageToPGM(aff_diff_win, fname,256.0);
      
      printf("iter = %d affine tracker res: %f\n", iteration, _findResidual(imgdiff, width, height));
#endif      
      _am_getGradientWinAffine(gradx2, grady2, *x2, *y2, *Axx, *Ayx , *Axy, *Ayy,
         width, height, gradx, grady);

      switch(affine_map){
      case 1:
        _am_compute4by1ErrorVector(imgdiff, gradx, grady, width, height, a);
        _am_compute4by4GradientMatrix(gradx, grady, width, height, T);

        status = _am_gauss_jordan_elimination(T,4,a,1);

        *Axx += a[0][0];
        *Ayx += a[1][0];
        *Ayy = *Axx;
        *Axy = -(*Ayx);

        dx = a[2][0];
        dy = a[3][0];

        break;
      case 2:
        _am_compute6by1ErrorVector(imgdiff, gradx, grady, width, height, a);
        _am_compute6by6GradientMatrix(gradx, grady, width, height, T);

        status = _am_gauss_jordan_elimination(T,6,a,1);

        *Axx += a[0][0];
        *Ayx += a[1][0];
        *Axy += a[2][0];
        *Ayy += a[3][0];

        dx = a[4][0];
        dy = a[5][0];
        break;
      case 3://ss (backward composition)
        {
          // transfer the gradient of the original image
          // into gradx, grady
          int hw = width/2;
          int hh = height/2;
          int i, j;
          float *gx=gradx;
          float *gy=grady;
          for (j = -hh ; j <= hh ; j++){
            for (i = -hw ; i <= hw ; i++)  {
              *gx++=_interpolate(x1+i, y1+j, gradx1);
              *gy++=_interpolate(x1+i, y1+j, grady1);
            }
          }
        }

        _am_compute6by1ErrorVector(imgdiff, gradx, grady, width, height, a);
        {
          int i;
          for (i=0; i<6; ++i)
            a[i][0]=-a[i][0];
        }
        _am_compute6by6GradientMatrix(gradx, grady, width, height, T);
        status = _am_gauss_jordan_elimination(T,6,a,1);
        
        { //C=A*B^(-1)
          double a11=(*Axx);
          double a12=*Axy;
          double a21=*Ayx;
          double a22=(*Ayy);
          double a13=0;
          double a23=0;
          double b11=1+a[0][0];
          double b12=a[2][0];
          double b21=a[1][0];
          double b22=1+a[3][0];
          double b13=a[4][0];
          double b23=a[5][0];

          double denom=b12*b21 - b11*b22;

/*
          double c11=( a21*b12 - a11*b22)/denom;
          double c12=( a22*b12 - a12*b22)/denom;
          double c21=(-a21*b11 + a11*b21)/denom;
          double c22=(-a22*b11 + a12*b21)/denom;
          double c13=(-a23*b12 - a13*b22 + b13*b22 - b12*b23)/denom;
          double c23=(-a23*b11 + a13*b21 - b13*b21 + b11*b23)/denom;
*/

          double c11=( a12*b21 - a11*b22)/denom;
          double c12=(-a12*b11 + a11*b12)/denom;
          double c21=( a22*b21 - a21*b22)/denom;
          double c22=(-a22*b11 + a21*b12)/denom;
          double c13=(a13*b12*b21 - a12*b13*b21 - a13*b11*b22 + 
                      a11*b13*b22 + a12*b11*b23 - a11*b12*b23)/denom;
          double c23=(a23*b12*b21 - a22*b13*b21 - a23*b11*b22 + 
                      a21*b13*b22 + a22*b11*b23 - a21*b12*b23)/denom;
/*
          fprintf(stderr, "A=[%f %f %f]\n", a11,a12,a13);
          fprintf(stderr, "  [%f %f %f]\n", a21,a22,a23);
          fprintf(stderr, "B=[%f %f %f]\n", b11,b12,b13);
          fprintf(stderr, "  [%f %f %f]\n", b21,b22,b23);
          fprintf(stderr, "C=[%f %f %f]\n", c11,c12,c13);
          fprintf(stderr, "  [%f %f %f]\n", c21,c22,c23);
*/
          *Axx=(float) c11;
          *Axy=(float) c12;
          *Ayx=(float) c21;
          *Ayy=(float) c22;
          dx=(float) c13;
          dy=(float) c23;
        }
        break;
      case 4://ss (forward composition)
        {
          // multiply the gradient of the warped image
          // by the jacobian of the warp over x
          int hw = width/2;
          int hh = height/2;
          int i, j;
          float *gx=gradx;
          float *gy=grady;
          for (j = -hh ; j <= hh ; j++){
            for (i = -hw ; i <= hw ; i++)  {
              float ggx=*Axx* *gx+*Axy* *gy;
              float ggy=*Ayx* *gx+*Ayy* *gy;
              *gx++=ggx;
              *gy++=ggy;
            }
          }
        }

        _am_compute6by1ErrorVector(imgdiff, gradx, grady, width, height, a);
        _am_compute6by6GradientMatrix(gradx, grady, width, height, T);
        status = _am_gauss_jordan_elimination(T,6,a,1);
        
        { //C=A*B
          double a11=(*Axx);
          double a12=*Axy;
          double a21=*Ayx;
          double a22=(*Ayy);
          double a13=0;
          double a23=0;
          double b11=1+a[0][0];
          double b12=a[2][0];
          double b21=a[1][0];
          double b22=1+a[3][0];
          double b13=a[4][0];
          double b23=a[5][0];

          double c11=(a11*b11 + a12*b21);
          double c12=(a11*b12 + a12*b22);
          double c21=(a21*b11 + a22*b21);
          double c22=(a21*b12 + a22*b22);
          double c13=(a13 + a11*b13 + a12*b23);
          double c23=(a23 + a21*b13 + a22*b23);
/*
          fprintf(stderr, "A=[%f %f %f]\n", a11,a12,a13);
          fprintf(stderr, "  [%f %f %f]\n", a21,a22,a23);
          fprintf(stderr, "B=[%f %f %f]\n", b11,b12,b13);
          fprintf(stderr, "  [%f %f %f]\n", b21,b22,b23);
          fprintf(stderr, "C=[%f %f %f]\n", c11,c12,c13);
          fprintf(stderr, "  [%f %f %f]\n", c21,c22,c23);
*/
          *Axx=(float) c11;
          *Axy=(float) c12;
          *Ayx=(float) c21;
          *Ayy=(float) c22;
          dx=(float) c13;
          dy=(float) c23;
        }
        break;
      case 5:// ss
        _am_compute3by1ErrorVector(
          img_mask->data, 
          img_gauss,
          imgdiff, gradx, grady, 
          width, height, a);
        _am_compute3by3GradientMatrix(
          img_mask->data, 
          img_gauss,
          gradx, grady, width, height, T);

        status = _am_gauss_jordan_elimination(T,3,a,1);

        *Axx += a[0][0];
        *Ayy = *Axx;
        assert(0==*Ayx && 0==*Axy);

        dx = a[1][0];
        dy = a[2][0];

        break;
      case 6:// ss
        _am_computeAffineMappedImage(img2, *x2,*y2, 
          *Axx,*Ayx,*Axy,*Ayy, 1,0, width,height, img_warp->data);
        _am_compute8by1ErrorVector(imgdiff, img_warp->data,*lambda, 
          gradx,grady, width,height, a);
        _am_compute8by8GradientMatrix(img_warp->data,*lambda, 
          gradx,grady, width,height, T);

        status = _am_gauss_jordan_elimination(T,8,a,1);

        *Axx += a[0][0];
        *Ayx += a[1][0];
        *Axy += a[2][0];
        *Ayy += a[3][0];
        dx = a[4][0];
        dy = a[5][0];

        *lambda += a[6][0];
        *delta  += a[7][0];
        break;
      case 7:// ss
        _am_computeAffineMappedImage(img2, *x2,*y2, 
          *Axx,*Ayx,*Axy,*Ayy, 1,0, width,height, img_warp->data);
        _am_compute5by1ErrorVector(
          img_mask->data,
          img_gauss,
          imgdiff, img_warp->data, *lambda,
          gradx, grady, width, height, a);
        _am_compute5by5GradientMatrix(
          img_mask->data, 
          img_gauss,
          img_warp->data,*lambda,
          gradx, grady, width, height, T);

        status = _am_gauss_jordan_elimination(T,5,a,1);

        *Axx += a[0][0];
        *Ayy = *Axx;
        assert(0==*Ayx && 0==*Axy);

        dx = a[1][0];
        dy = a[2][0];

        *lambda += a[3][0];
        *delta  += a[4][0];
        break;
      default:/*notreached*/
        assert(0);
        dx=dy=0;        
      }
      
      *x2 += dx;
      *y2 += dy;
      
      /* old upper left corner - new upper left corner */
      ul_x -=  *Axx * (-hw) + *Axy *   hh  + *x2;  
      ul_y -=  *Ayx * (-hw) + *Ayy *   hh  + *y2; 
      /* old lower left corner - new lower left corner */
      ll_x -=  *Axx * (-hw) + *Axy * (-hh) + *x2;  
      ll_y -=  *Ayx * (-hw) + *Ayy * (-hh) + *y2;
      /* old upper right corner - new upper right corner */
      ur_x -=  *Axx *   hw  + *Axy *   hh  + *x2;  
      ur_y -=  *Ayx *   hw  + *Ayy *   hh  + *y2;
      /* old lower right corner - new lower right corner */
      lr_x -=  *Axx *   hw  + *Axy * (-hh) + *x2;  
      lr_y -=  *Ayx *   hw  + *Ayy * (-hh) + *y2;

#ifdef DEBUG_AFFINE_MAPPING 
      printf ("iter = %d, ul_x=%f ul_y=%f ll_x=%f ll_y=%f ur_x=%f ur_y=%f lr_x=%f lr_y=%f \n",
        iteration, ul_x, ul_y, ll_x, ll_y, ur_x, ur_y, lr_x, lr_y);
#endif  

      convergence = ( fabs(dx) < th && fabs(dy) < th  &&
                      fabs(ul_x) < th_aff && fabs(ul_y) < th_aff &&
                      fabs(ll_x) < th_aff && fabs(ll_y) < th_aff &&
                      fabs(ur_x) < th_aff && fabs(ur_y) < th_aff &&
                      fabs(lr_x) < th_aff && fabs(lr_y) < th_aff);
    }
    
    if (status == KLT_SMALL_DET)  break;

    iteration++;
#ifdef DEBUG_AFFINE_MAPPING 
    printf ("iter = %d, x1=%f, y1=%f, x2=%f, y2=%f,  Axx=%f, Ayx=%f , Axy=%f, Ayy=%f \n",iteration, x1, y1, *x2, *y2,  *Axx, *Ayx , *Axy, *Ayy);
#endif   
    /// fprintf(stderr, "####  AffIt:%d, x1=(%f,%f), x2=(%f,%f), R=%f, Axx=%f,Ayx=%f,Axy=%f,Ayy=%f, l=%f,d=%f\n",
    ///   iteration, x1,y1, *x2,*y2, residual, *Axx,*Ayx ,*Axy,*Ayy, *lambda,*delta); //ss diag
  }  while (!convergence && iteration<max_iterations); 
    /*}  while ( (fabs(dx)>=th || fabs(dy)>=th || (affine_map && iteration < 8) ) && iteration < max_iterations); */
  _am_free_matrix(T);
  _am_free_matrix(a);
  {
    double ul_x =  *Axx * (-hw) + *Axy *   hh  + *x2;  /* upper left corner */
    double ul_y =  *Ayx * (-hw) + *Ayy *   hh  + *y2; 
    double ll_x =  *Axx * (-hw) + *Axy * (-hh) + *x2;  /* lower left corner */
    double ll_y =  *Ayx * (-hw) + *Ayy * (-hh) + *y2;
    double ur_x =  *Axx *   hw  + *Axy *   hh  + *x2;  /* upper right corner */
    double ur_y =  *Ayx *   hw  + *Ayy *   hh  + *y2;
    double lr_x =  *Axx *   hw  + *Axy * (-hh) + *x2;  /* lower right corner */
    double lr_y =  *Ayx *   hw  + *Ayy * (-hh) + *y2;
    /* Check whether window is out of bounds */
    if (*Axx<0 || *Ayy<0 ||
        ul_x < borderx || 
        ll_x < borderx || 
        ur_x > nc2-borderx-one_plus_eps || 
        lr_x > nc2-borderx-one_plus_eps || 
        ul_y > nr2-bordery-one_plus_eps || 
        ll_y < bordery || 
        ur_y > nr2-bordery-one_plus_eps || 
        lr_y < bordery)
    status = KLT_OOB-1000; //ss
    if (index==-1)
      printf ("%d: %f,%f\n", index,ul_y,ll_y);

    /* calculate the residuals */
    *residual_start=-1;
    *residual_stop =-1;
    if (status == KLT_TRACKED) {
      _am_computeAffineMappedImage(img2, *x2,*y2, *Axx,*Ayx,*Axy,*Ayy, 
        *lambda,*delta, width,height, img_warp->data);//ss

      if(!affine_map){
        _computeIntensityDifference(img1, img2, x1, y1, *x2, *y2, 
          width, height, imgdiff);
      }else{
        _am_computeIntensityDifferenceAffine(img1,img2, x1,y1, *x2,*y2,  
          *Axx,*Ayx,*Axy,*Ayy, *lambda,*delta,
          width, height, imgdiff);
      }

      if (bgmodel_alpha>0){
        double maxrcmovx=mymax4(
          fabs(ul_x-old_ul_x),
          fabs(ll_x-old_ll_x),
          fabs(ur_x-old_ur_x),
          fabs(lr_x-old_lr_x));
        double maxrcmovy=mymax4(
          fabs(ul_y-old_ul_y),
          fabs(ll_y-old_ll_y),
          fabs(ur_y-old_ur_y),
          fabs(lr_y-old_lr_y));
        double maxmov=mymax2(dxtrans+maxrcmovx,dytrans+maxrcmovy);

        updateGaussians(width,height, imgdiff, 
            (float)maxmov,bgmodel_alpha, 
            img_mu->data,img_sigmaSqr->data);
        {
          double garbage1=updateMaskGaussian(width,height, 
              img_mu->data,img_sigmaSqr->data, 
              bgmodel_thsigma, img_mask->data);
          if (garbage1>0.4)
            status = KLT_NOT_FOUND-1100;
        }{
          double garbage2=updateMaskX84(width,height, 
              img_mu->data, tmpmask1);
          garbage2+=
          combineMasks(width,height, 
              img_mask->data, tmpmask1);
          if (garbage2>0.65){
            status = KLT_NOT_FOUND-1000;
          }
        }
      } else{
        memcpy(tmpmask1, img_mask->data, width*height*sizeof(float));
      }

      //memcpy(img_mask->data, tmpmask, width*height*sizeof(float));
      //memcpy(img_mask->data, imgdiff, width*height*sizeof(float));
      //memcpy(img_warp->data, img_mu->data, width*height*sizeof(float));
      //memcpy(img_warp->data, img_gauss, width*height*sizeof(float));

      *residual_start=_findResidual(imgdiff, width, height);
      *residual_stop =_findMaskedResidual(imgdiff, tmpmask1, width, height);
    }
  }

  /* Check whether the contrast compensation went awry */
  if (status == KLT_TRACKED && *lambda<0) {
    status = KLT_OOB-1050; //ss
  }

  /* Check whether feature point has moved too much during iteration*/
  if (status == KLT_TRACKED && 
     (fabs(*x2-old_x2)>mdd || fabs(*y2-old_y2)>mdd))
  {
    status = KLT_OOB-1100; //ss
  }
      
  if (status == KLT_TRACKED) {
    /* Check whether the residue is too large */
    float gxx,gyy,gxy;
    _am_compute2x2Masked(img_mask->data, gradx, grady, width, height, &gxx, &gxy, &gyy);
    *lambda2=(float)((*lambda)*(*lambda)*(gxx+gyy-sqrt(4*gxy*gxy+(gxx-gyy)*(gxx-gyy)))/(2*width*height));
    if (*lambda2<2)
      status = KLT_SMALL_DET-1200;

    if (checkResidual(*residual_stop, max_residue))
      status = KLT_LARGE_RESIDUE;

    if (checkResidual(*residual_stop, max_residue/2)){
      //double magfactor=old_mag/(*Ayy+*Axx);
      //if (magfactor>1.5 || magfactor<0.66)
      //  status = KLT_OOB-1300; //ss

      if (*lambda2<4)
        status = KLT_SMALL_DET-1300;
    }
  }
  

#ifdef DEBUG_AFFINE_MAPPING
    printf("iter = %d final_res = %f\n", iteration, _findResidual(imgdiff, width, height));
#endif 

  /* Free memory */
  free(imgdiff);  free(gradx);  free(grady); free(tmpmask1);free(tmpmask2);

#ifdef DEBUG_AFFINE_MAPPING
  printf("iter = %d status=%d\n", iteration, status);
#endif 
  
  /* Return appropriate value */
  return status;
}



int _am_trackFeatureAffineWrap(
  KLT_TrackingContext tc, 
  KLT_Feature feat,
  _KLT_Pyramid pyr, 
  _KLT_Pyramid pyr_gradx, 
  _KLT_Pyramid pyr_grady, 
  float xlocout, float ylocout,
  float xlocold, float ylocold)
{
  static int szmagfactor=sizeof(feat->magfactor)/sizeof(feat->magfactor[0]);

  double scale;
  float axx, axy, ayx, ayy;
  double bgmodel_alpha=tc->affine_bgmodel_alpha;
  int val;

  // hysteresis test
  double mag=(feat->aff_Axx+feat->aff_Ayy)/2.0;
  int pyrind=feat->pyrind;
  double pyrthis=pow(tc->subsampling, pyrind);
  double pyrnext=tc->subsampling*pyrthis;
  if (mag/pyrnext>0.9){
    if (pyrind+1<tc->nPyramidLevels)
      ++pyrind;
  }
  if (mag/pyrthis<0.8){
    if (pyrind-1>=0)
      --pyrind;
  }

  // apply the scale
  {
    scale =pow(tc->subsampling, pyrind);
    axx=(float)(feat->aff_Axx/scale);
    axy=(float)(feat->aff_Axy/scale);
    ayx=(float)(feat->aff_Ayx/scale);
    ayy=(float)(feat->aff_Ayy/scale);
    xlocout/=(float)scale;
    ylocout/=(float)scale;
  }

  // init the gaussians on scale change
  if (bgmodel_alpha>0){
    if (feat->pyrind<pyrind){
      bgmodel_alpha=3.0;
    }
    if (feat->pyrind>pyrind){
      bgmodel_alpha=2.0;
    }
  }
  feat->pyrind=pyrind;

  /* affine tracking */
  val = _am_trackFeatureAffine(
            feat->aff_x, 
            feat->aff_y,
            &xlocout, &ylocout,
            feat->aff_img, 
            feat->aff_img_gradx, 
            feat->aff_img_grady,
            pyr->img[pyrind], 
            pyr_gradx->img[pyrind], 
            pyr_grady->img[pyrind],
            tc->affine_window_width, tc->affine_window_height,
            tc->affine_max_iterations,
            tc->min_determinant,
            tc->min_displacement,
            tc->affine_min_displacement,
            tc->affine_max_residue, 
            tc->affineConsistencyCheck,
            tc->affine_max_displacement_differ,
            &axx,
            &ayx,
            &axy,
            &ayy,
            &feat->lambda,
            &feat->delta,
            feat->aff_img_last, 
            feat->aff_img_mask, 
            feat->aff_img_mu, 
            feat->aff_img_sigmaSqr, 
            &feat->lambda2,
            &feat->residual_trans,
            &feat->residual_aff,
            (float)bgmodel_alpha,
            (float)tc->affine_bgmodel_thsigma,
            (float)fabs(xlocold-xlocout), (float)fabs(ylocold-ylocout),
            5, 5,
            (float*)tc->img_gauss,
            feat->id);
  feat->indexMagfactor=(feat->indexMagfactor+1)%szmagfactor;

  // revert the scale
  feat->aff_Axx=(float)(axx*scale);
  feat->aff_Axy=(float)(axy*scale);
  feat->aff_Ayx=(float)(ayx*scale);
  feat->aff_Ayy=(float)(ayy*scale);
  xlocout*=(float)(scale);
  ylocout*=(float)(scale);

  //fprintf(stderr, "####  Affine correction: %f,%f\n",//ss diag
  //  xlocout-feat->x, ylocout-feat->y);//ss
  feat->x = xlocout;//ss 
  feat->y = ylocout;//ss 
  feat->val = val;

  if (0){
    if (val!=KLT_TRACKED){//ss diag
      fprintf(stderr, "####%4d: affine tracking failure %6d at (%f,%f)\n", 
        feat->id, val, xlocout,ylocout);
    }
  }
  if (0){
    if (val==KLT_TRACKED){//ss diag
      fprintf(stderr, "####%4d: affine tracking at (%f,%f)\n", 
        feat->id, xlocout,ylocout);
    }
  }
  return val;
}

/*
 * CONSISTENCY CHECK OF FEATURES BY AFFINE MAPPING (END)
 **********************************************************************/



/*********************************************************************
 * KLTTrackFeatures
 *
 * Tracks feature points from one image to the next.
 */

void KLTTrackFeatures(
  KLT_TrackingContext tc,
  KLT_PixelType *img1,
  KLT_PixelType *img2,
  int ncols,
  int nrows,
  KLT_FeatureList featurelist)
{
  _KLT_FloatImage tmpimg, floatimg1=0, floatimg2;
  _KLT_Pyramid pyramid1, pyramid1_gradx, pyramid1_grady,
    pyramid2, pyramid2_gradx, pyramid2_grady;
  int subsampling = tc->subsampling; //ss
  float xloc,yloc, xlocout,ylocout, xlocold,ylocold;
  int val=KLT_NOT_FOUND;
  int indx, r;
  KLT_BOOL floatimg1_created = FALSE;
  int i;
  static int szmagfactor=sizeof(featurelist->feature[indx]->magfactor)/sizeof(featurelist->feature[indx]->magfactor[0]);

  if (KLT_verbose >= 1)  {
    fprintf(stderr,  "(KLT) Tracking %d features in a %d by %d image...\n",
            KLTCountRemainingFeatures(featurelist), ncols, nrows);
    fflush(stderr);
  }

  /* Check window size (and correct if necessary) */
  if (tc->window_width % 2 != 1) {
    tc->window_width = tc->window_width+1;
    KLTWarning("Tracking context's window width must be odd.  "
               "Changing to %d.\n", tc->window_width);
  }
  if (tc->window_height % 2 != 1) {
    tc->window_height = tc->window_height+1;
    KLTWarning("Tracking context's window height must be odd.  "
               "Changing to %d.\n", tc->window_height);
  }
  if (tc->window_width < 3) {
    tc->window_width = 3;
    KLTWarning("Tracking context's window width must be at least three.  \n"
               "Changing to %d.\n", tc->window_width);
  }
  if (tc->window_height < 3) {
    tc->window_height = 3;
    KLTWarning("Tracking context's window height must be at least three.  \n"
               "Changing to %d.\n", tc->window_height);
  }

  /* Create temporary image */
  tmpimg = _KLTCreateFloatImage(ncols, nrows);

  /* Process first image by converting to float, smoothing, computing */
  /* pyramid, and computing gradient pyramids */
  if (tc->sequentialMode && tc->pyramid_last != NULL)  {
    pyramid1 = (_KLT_Pyramid) tc->pyramid_last;
    pyramid1_gradx = (_KLT_Pyramid) tc->pyramid_last_gradx;
    pyramid1_grady = (_KLT_Pyramid) tc->pyramid_last_grady;
    if (pyramid1->ncols[0] != ncols || pyramid1->nrows[0] != nrows)
      KLTError("(KLTTrackFeatures) Size of incoming image (%d by %d) "
               "is different from size of previous image (%d by %d)\n",
               ncols, nrows, pyramid1->ncols[0], pyramid1->nrows[0]);
    assert(pyramid1_gradx != NULL);
    assert(pyramid1_grady != NULL);
  } else  {
    floatimg1_created = TRUE;
    floatimg1 = _KLTCreateFloatImage(ncols, nrows);
    _KLTToFloatImage(img1, ncols, nrows, tmpimg);
    _KLTComputeSmoothedImage(tmpimg, _KLTComputeSmoothSigma(tc), floatimg1);
    pyramid1 = _KLTCreatePyramid(ncols, nrows, subsampling, tc->nPyramidLevels);
    _KLTComputePyramid(floatimg1, pyramid1, tc->pyramid_sigma_fact);
    pyramid1_gradx = _KLTCreatePyramid(ncols, nrows, subsampling, tc->nPyramidLevels);
    pyramid1_grady = _KLTCreatePyramid(ncols, nrows, subsampling, tc->nPyramidLevels);
    for (i = 0 ; i < tc->nPyramidLevels ; i++)
      _KLTComputeGradients(pyramid1->img[i], tc->grad_sigma, 
                           pyramid1_gradx->img[i],
                           pyramid1_grady->img[i]);
  }

  /* Do the same thing with second image */
  floatimg2 = _KLTCreateFloatImage(ncols, nrows);
  _KLTToFloatImage(img2, ncols, nrows, tmpimg);
  _KLTComputeSmoothedImage(tmpimg, _KLTComputeSmoothSigma(tc), floatimg2);
  pyramid2 = _KLTCreatePyramid(ncols, nrows, subsampling, tc->nPyramidLevels);
  _KLTComputePyramid(floatimg2, pyramid2, tc->pyramid_sigma_fact);
  pyramid2_gradx = _KLTCreatePyramid(ncols, nrows, subsampling, tc->nPyramidLevels);
  pyramid2_grady = _KLTCreatePyramid(ncols, nrows, subsampling, tc->nPyramidLevels);
  for (i = 0 ; i < tc->nPyramidLevels ; i++)
    _KLTComputeGradients(pyramid2->img[i], tc->grad_sigma, 
                         pyramid2_gradx->img[i],
                         pyramid2_grady->img[i]);

  /* Write internal images */
  if (tc->writeInternalImages)  {/*//ss
    char fname[80];
    for (i = 0 ; i < tc->nPyramidLevels ; i++)  {
      sprintf(fname, "kltimg_tf_i%d.pgm", i);
      _KLTWriteFloatImageToPGM(pyramid1->img[i], fname);
      sprintf(fname, "kltimg_tf_i%d_gx.pgm", i);
      _KLTWriteFloatImageToPGM(pyramid1_gradx->img[i], fname);
      sprintf(fname, "kltimg_tf_i%d_gy.pgm", i);
      _KLTWriteFloatImageToPGM(pyramid1_grady->img[i], fname);
      sprintf(fname, "kltimg_tf_j%d.pgm", i);
      _KLTWriteFloatImageToPGM(pyramid2->img[i], fname);
      sprintf(fname, "kltimg_tf_j%d_gx.pgm", i);
      _KLTWriteFloatImageToPGM(pyramid2_gradx->img[i], fname);
      sprintf(fname, "kltimg_tf_j%d_gy.pgm", i);
      _KLTWriteFloatImageToPGM(pyramid2_grady->img[i], fname);
    }*///ss
  }

  /* For each feature, do ... */
  for (indx = 0 ; indx < featurelist->nFeatures ; indx++)  {

    /* Only track features that are not lost */
    if (featurelist->feature[indx]->val >= 0)  {

      double pyrScale=1;
      double scale=(featurelist->feature[indx]->aff_Axx+
                    featurelist->feature[indx]->aff_Axx)/2.0;

      xloc= xlocold= featurelist->feature[indx]->x;
      yloc= ylocold= featurelist->feature[indx]->y;
      xlocout= featurelist->feature[indx]->xpred;
      if (xlocout==-1) xlocout=xloc;
      ylocout= featurelist->feature[indx]->ypred;
      if (ylocout==-1) ylocout=yloc;

      
      if (0){
        fprintf(stderr, "#### Tracking (%f,%f) -> (%f,%f)\n", //ss diag
           xloc, yloc, xlocout, ylocout);
      }

      /* Transform location to coarsest resolution */
      for (r = tc->nPyramidLevels - 1 ; r >= 0 ; r--)  {
        xloc /= subsampling;     yloc /= subsampling;
        xlocout /= subsampling;  ylocout /= subsampling;
        pyrScale *= subsampling;
      }

      /* Beginning with coarsest resolution, do ... */
      for (r = tc->nPyramidLevels - 1 ; r >= 0 ; r--)  {
        int idealSize=tc->window_width;
        if (tc->affineConsistencyCheck >= 0){
          idealSize=2*myround(tc->affine_window_width*scale/(double)pyrScale/2.0-0.5)+1;
          if (idealSize>tc->window_width){
            idealSize=tc->window_width;
          }
        }

        /* Track feature at current resolution */
        xloc *= subsampling;  yloc *= subsampling;
        xlocout *= subsampling;  ylocout *= subsampling;
        pyrScale /= subsampling;

        if (idealSize<3) continue;

        /// fprintf(stderr, "####  Before tracking: %f,%f@%d\n",xlocout,ylocout, idealSize); //ss diag 
        val = _trackFeature(xloc, yloc, 
                            &xlocout, &ylocout,
                            pyramid1->img[r], 
                            pyramid1_gradx->img[r], pyramid1_grady->img[r], 
                            pyramid2->img[r], 
                            pyramid2_gradx->img[r], pyramid2_grady->img[r],
                            idealSize, idealSize,
                            tc->max_iterations,
                            tc->min_determinant,
                            tc->min_displacement,
                            tc->max_residue,
                            &featurelist->feature[indx]->lambda2,
                            &featurelist->feature[indx]->residual_trans);
        /// fprintf(stderr, "####  After tracking: %f,%f\n",xlocout,ylocout); //ss diag 

        if (val==KLT_SMALL_DET || val==KLT_OOB)
          break;
      }
      
      /// fprintf(stderr, "####  Total: dx=%f, dy=%f\n",xlocout-xloc,ylocout-yloc); //ss diag 

      /* Record feature */
      if (val == KLT_OOB)  {
        featurelist->feature[indx]->x   = -1.0;
        featurelist->feature[indx]->y   = -1.0;
        val= featurelist->feature[indx]->val = KLT_OOB-10000;
      //} else if (_outOfBounds(xlocout, ylocout, ncols, nrows, tc->borderx, tc->bordery))  {
      //  featurelist->feature[indx]->x   = -1.0;
      //  featurelist->feature[indx]->y   = -1.0;
      //  val= featurelist->feature[indx]->val= KLT_OOB-11000;
      } else if (val == KLT_SMALL_DET)  {
        featurelist->feature[indx]->x   = -1.0;
        featurelist->feature[indx]->y   = -1.0;
        val= featurelist->feature[indx]->val = KLT_SMALL_DET-10000;
      } else if (val == KLT_LARGE_RESIDUE)  {
        featurelist->feature[indx]->x   = -1.0;
        featurelist->feature[indx]->y   = -1.0;
        val= featurelist->feature[indx]->val = KLT_LARGE_RESIDUE-10000;
      } else if (val == KLT_MAX_ITERATIONS)  {
        featurelist->feature[indx]->x   = -1.0;
        featurelist->feature[indx]->y   = -1.0;
        val= featurelist->feature[indx]->val = KLT_MAX_ITERATIONS-10000;
      } else  {
        featurelist->feature[indx]->x = xlocout;
        featurelist->feature[indx]->y = ylocout;
        featurelist->feature[indx]->val = KLT_TRACKED;
        if (tc->affineConsistencyCheck >= 0 && val == KLT_TRACKED)  { /*for affine mapping*/
          int border = 2; /* add border for interpolation */
  
#ifdef DEBUG_AFFINE_MAPPING	  
      	  glob_index = indx;
#endif
  
          if(!featurelist->feature[indx]->aff_img){
            int i;
            featurelist->feature[indx]->indexMagfactor=0;
            for (i=0; i<szmagfactor; ++i){
              featurelist->feature[indx]->magfactor[i]=1;
            }
            /* save image and gradient for each feature at finest resolution after first successful track */
            featurelist->feature[indx]->aff_img = _KLTCreateFloatImage((tc->affine_window_width+border), (tc->affine_window_height+border));
            featurelist->feature[indx]->aff_img_last = _KLTCreateFloatImage((tc->affine_window_width), (tc->affine_window_height));
            featurelist->feature[indx]->aff_img_mask = _KLTCreateFloatImage((tc->affine_window_width), (tc->affine_window_height));
            featurelist->feature[indx]->aff_img_gradx = _KLTCreateFloatImage((tc->affine_window_width+border), (tc->affine_window_height+border));
            featurelist->feature[indx]->aff_img_grady = _KLTCreateFloatImage((tc->affine_window_width+border), (tc->affine_window_height+border));
            featurelist->feature[indx]->aff_img_mu = _KLTCreateFloatImage((tc->affine_window_width), (tc->affine_window_height));
            featurelist->feature[indx]->aff_img_sigmaSqr = _KLTCreateFloatImage((tc->affine_window_width), (tc->affine_window_height));
            featurelist->feature[indx]->pyrind=0;
            _am_getSubFloatImage(pyramid1->img[0],xloc,yloc,featurelist->feature[indx]->aff_img);
            _am_getSubFloatImage(pyramid1_gradx->img[0],xloc,yloc,featurelist->feature[indx]->aff_img_gradx);
            _am_getSubFloatImage(pyramid1_grady->img[0],xloc,yloc,featurelist->feature[indx]->aff_img_grady);
            //_am_getSubFloatImage(pyramid1->img[0],xloc,yloc,featurelist->feature[indx]->aff_img_mu);
            fillBuf(featurelist->feature[indx]->aff_img_mu->data,    tc->affine_window_width*tc->affine_window_height, 0);
            fillBuf(featurelist->feature[indx]->aff_img_sigmaSqr->data, tc->affine_window_width*tc->affine_window_height, 5*5);
            fillBuf(featurelist->feature[indx]->aff_img_mask->data,  tc->affine_window_width*tc->affine_window_height, 255);
            fillBuf(featurelist->feature[indx]->aff_img_last->data,  tc->affine_window_width*tc->affine_window_height, 255);
            featurelist->feature[indx]->aff_x = xloc - (int) xloc + (tc->affine_window_width+border)/2;
            featurelist->feature[indx]->aff_y = yloc - (int) yloc + (tc->affine_window_height+border)/2;;
          }else{
            /* affine tracking */
            val = _am_trackFeatureAffineWrap(tc, 
              featurelist->feature[indx],
              pyramid2, pyramid2_gradx, pyramid2_grady,
              xlocout, ylocout, xlocold, ylocold);
          }
        }
      }
      if (0){//ss diag
        if ( val!=KLT_TRACKED){
          fprintf(stderr, "####%4d: translational tracking failure %6d at (%f,%f)\n", 
            indx, val, xlocout,ylocout);
        }
      }
      if (0){//ss diag
        if ( val==KLT_TRACKED){
          fprintf(stderr, "####  Feature: #%d: (%f,%f)\n", indx,
            featurelist->feature[indx]->x,
            featurelist->feature[indx]->y);
        }
      }

      //ss
      featurelist->feature[indx]->xpred=-1;
      featurelist->feature[indx]->ypred=-1;
    }
  }

  if (tc->sequentialMode)  {
    tc->pyramid_last = pyramid2;
    tc->pyramid_last_gradx = pyramid2_gradx;
    tc->pyramid_last_grady = pyramid2_grady;
  } else  {
    _KLTFreePyramid(pyramid2);
    _KLTFreePyramid(pyramid2_gradx);
    _KLTFreePyramid(pyramid2_grady);
  }

  /* Free memory */
  _KLTFreeFloatImage(tmpimg);
  if (floatimg1_created)  _KLTFreeFloatImage(floatimg1);
  _KLTFreeFloatImage(floatimg2);
  _KLTFreePyramid(pyramid1);
  _KLTFreePyramid(pyramid1_gradx);
  _KLTFreePyramid(pyramid1_grady);

  if (KLT_verbose >= 1)  {
    fprintf(stderr,  "\n\t%d features successfully tracked.\n",
            KLTCountRemainingFeatures(featurelist));
    if (tc->writeInternalImages)
      fprintf(stderr,  "\tWrote images to 'kltimg_tf*.pgm'.\n");
    fflush(stderr);
  }

}


int KLTResumeFeature(
  KLT_TrackingContext tc,
  KLT_Feature feature,
  double x, double y, double mag)
{
  int rv=0;
  if (feature->aff_img == 0){
    rv=-10;
  } else{
    feature->aff_Axx = (float)mag;
    feature->aff_Ayx = 0.0;
    feature->aff_Axy = 0.0;
    feature->aff_Ayy = (float)mag;

    feature->lambda  = 1.0;
    feature->delta   = 0.0;

    fillBuf(feature->aff_img_mu->data,       tc->affine_window_width*tc->affine_window_height, 0);
    fillBuf(feature->aff_img_sigmaSqr->data, tc->affine_window_width*tc->affine_window_height, 5*5);
    fillBuf(feature->aff_img_mask->data,     tc->affine_window_width*tc->affine_window_height, 255);

    {
      double mddold=tc->affine_max_displacement_differ;
      tc->affine_max_displacement_differ=20;
      rv= _am_trackFeatureAffineWrap(tc, feature, 
        (_KLT_Pyramid) tc->pyramid_last, 
        (_KLT_Pyramid) tc->pyramid_last_gradx, 
        (_KLT_Pyramid) tc->pyramid_last_grady, 
        (float) x, (float) y, (float) x, (float) y);
      tc->affine_max_displacement_differ=(float)mddold;
    }

    if (0){
      if (rv!=KLT_TRACKED){//ss diag
        fprintf(stderr, "####%4d: resuming failure %6d at (%f,%f)\n", 
          feature->id, rv, x,y);
      }
    }
    if (0){
      if (rv==KLT_TRACKED){//ss diag
        fprintf(stderr, "####%4d: resuming at (%f,%f)\n", 
          feature->id, x,y);
      }
    }
  }
  return rv;
}


void KLTSetFirstImage(
  KLT_TrackingContext tc,
  KLT_PixelType *img,
  int ncols,
  int nrows)
{
  _KLT_FloatImage tmpimg = _KLTCreateFloatImage(ncols, nrows);
  _KLT_FloatImage floatimg = _KLTCreateFloatImage(ncols, nrows);
  _KLTToFloatImage(img, ncols, nrows, tmpimg);
  _KLTComputeSmoothedImage(tmpimg, _KLTComputeSmoothSigma(tc), floatimg);

  if (tc->pyramid_last)
    _KLTFreePyramid((_KLT_Pyramid)tc->pyramid_last);
  tc->pyramid_last = _KLTCreatePyramid(ncols, nrows, tc->subsampling, tc->nPyramidLevels);
  _KLTComputePyramid(floatimg, (_KLT_Pyramid)tc->pyramid_last, tc->pyramid_sigma_fact);

  if (tc->pyramid_last_gradx)
    _KLTFreePyramid((_KLT_Pyramid)tc->pyramid_last_gradx);
  tc->pyramid_last_gradx = _KLTCreatePyramid(ncols, nrows, tc->subsampling, tc->nPyramidLevels);
  if (tc->pyramid_last_grady)
    _KLTFreePyramid((_KLT_Pyramid)tc->pyramid_last_grady);
  tc->pyramid_last_grady = _KLTCreatePyramid(ncols, nrows, tc->subsampling, tc->nPyramidLevels);
  {
    int i;
    for (i=0; i<tc->nPyramidLevels; ++i)
      _KLTComputeGradients(
         ((_KLT_Pyramid) tc->pyramid_last)->img[i], tc->grad_sigma, 
         ((_KLT_Pyramid) tc->pyramid_last_gradx)->img[i],
         ((_KLT_Pyramid) tc->pyramid_last_grady)->img[i]);
  }
  _KLTFreeFloatImage(tmpimg);
  _KLTFreeFloatImage(floatimg);
}
