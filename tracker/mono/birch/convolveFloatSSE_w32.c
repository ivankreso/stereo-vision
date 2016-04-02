#include "convolveFloatSSE.h"

#include <assert.h>


void convolveFloatSSEHor(
  float* imgin, int nrows, int ncols,
  float* kernel, int szker,
  float* imgout)
{
#if !defined(_WIN64)
  float *pPixIn = imgin;
  float *pPixOut = imgout;
  int radius = szker / 2;

  const int szSSEvec=4;
  const int szSSEreg=szSSEvec*sizeof(float);
  const int nPixels=(ncols-2*radius);
  const int nPackets=nPixels/szSSEvec;
  int i,j,k;

  assert(szker % 2 == 1);
  assert(imgin != imgout);
  
  /* For each row, do ... */
  for (i = 0 ; i < nrows ; i++)  {

    /* Zero leftmost columns */
    for (j=0; j < radius ; j++){
      *pPixOut++=0.0;
    }
  
    /* Convolute SIMD: */
    __asm {
      align 16
      mov ebx, kernel
      mov esi, pPixIn
      mov edi, pPixOut
	 
      /* begin loop 
	    for (jPacket=0; jPacket<nPackets; jPacket++)*/
      xor ecx,ecx 
	  jmp mytest_jPacket
myloop_jPacket:
        pxor xmm3, xmm3 // 0 ->xmm3
        
        /* begin loop 
           for (k=0; k<szker; ++k) { */
        xor eax,eax
        jmp mytest_k
myloop_k:
        movss xmm1, xmmword ptr [ebx+eax*4] //szSSEvec
        shufps xmm1, xmm1,0
        movups xmm2, xmmword ptr [esi+eax*4] //szSSEvec
        mulps xmm2,xmm1
        addps xmm3,xmm2
        
        add eax, 1

        
mytest_k:        
        cmp eax,szker
        jl myloop_k
        /* end loop 'for k' */

        movups xmmword ptr [edi],xmm3

        add	esi, szSSEreg//szSSEreg???
        add	edi, szSSEreg
		
        add ecx, 1
        
mytest_jPacket:
      cmp ecx,nPackets
      jl myloop_jPacket
      /* end loop 'for jPacket' */
    }
    pPixIn +=nPackets*szSSEvec;
    pPixOut+=nPackets*szSSEvec;
    
    /* Convolute the rest one by one */
	{
      int countRest=nPixels - nPackets*szSSEvec;
      for (j=0; j<countRest; j++){
        float sum=0;
        for (k=0; k<szker; ++k) {
          sum+=pPixIn[k]*kernel[k];
        }
        *pPixOut++=sum;
        
        pPixIn++;
      }
	}

    /* Zero rightmost columns */
    for (j=0; j < radius; ++j){
      *pPixOut++ = 0.0;
     
     
    }

    pPixIn += 2* radius;
  }

#endif  
}



void convolveFloatSSEVert(
  float* imgin, int nrows, int ncols,
  float* kernel, int szker,
  float* imgout)
{
#if !defined(_WIN64)
  float *pPixIn = imgin;
  float *pPixOut = imgout;
  int radius = szker / 2;

  const int szSSEvec=4;
  const int szSSEreg=szSSEvec*sizeof(float);
  const int nPixels=(ncols-2*radius);
  const int nPackets=nPixels/szSSEvec;
  int i,j,k;

  assert(szker % 2 == 1);
  assert(imgin != imgout);
  
  /* For each row, do ... */
  for (i = 0 ; i < nrows ; i++)  {

    /* Zero leftmost columns */
    for (j=0; j < radius ; j++){
      *pPixOut++=0.0;
    }
  
    /* Convolute SIMD: */
    __asm {
      align 16
      mov ebx, kernel
      mov esi, pPixIn
      mov edi, pPixOut
	 
      /* begin loop 
	    for (jPacket=0; jPacket<nPackets; jPacket++)*/
      xor ecx,ecx 
	  jmp mytest_jPacket
myloop_jPacket:
        pxor xmm3, xmm3 // 0 ->xmm3
        
        /* begin loop 
           for (k=0; k<szker; ++k) { */
        xor eax,eax
        jmp mytest_k
myloop_k:
        movss xmm1, xmmword ptr [ebx+eax*4] //szSSEvec
        shufps xmm1, xmm1,0
        movups xmm2, xmmword ptr [esi+eax*4] //szSSEvec
        mulps xmm2,xmm1
        addps xmm3,xmm2
        
        add eax, 1

        
mytest_k:        
        cmp eax,szker
        jl myloop_k
        /* end loop 'for k' */

        movups xmmword ptr [edi],xmm3

        add	esi, szSSEreg//szSSEreg???
        add	edi, szSSEreg
		
        add ecx, 1
        
mytest_jPacket:
      cmp ecx,nPackets
      jl myloop_jPacket
      /* end loop 'for jPacket' */
    }
    pPixIn +=nPackets*szSSEvec;
    pPixOut+=nPackets*szSSEvec;
    
    /* Convolute the rest one by one */
	{
      int countRest=nPixels - nPackets*szSSEvec;
      for (j=0; j<countRest; j++){
        float sum=0;
        for (k=0; k<szker; ++k) {
          sum+=pPixIn[k]*kernel[k];
        }
        *pPixOut++=sum;
        pPixIn++;
      }
	}

    /* Zero rightmost columns */
    for (j=0; j < radius; ++j){
      *pPixOut++ = 0.0;
    }

    pPixIn += 2* radius;
  }
#endif  
}

