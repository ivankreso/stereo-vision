// Copyright © Robert Spangenberg, 2014.
// See license.txt for more details

#include "iostream"

#include <smmintrin.h> // intrinsics
#include <emmintrin.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "StereoBMHelper.h"
#include "FastFilters.h"
#include "StereoSGM.h"

#include <vector>
#include <list>
#include <algorithm>
#include <numeric>

#include "MyImage.h"

void correctEndianness(uint16* input, uint16* output, uint32 size)
{
    uint8* outputByte = (uint8*)output;
    uint8* inputByte = (uint8*)input;

    for (uint32 i=0; i < size; i++) {
        *(outputByte+1) = *inputByte;
        *(outputByte) = *(inputByte+1);
        outputByte+=2;
        inputByte+=2;
    }
}

template <typename T>
void census5x5_t_SSE(T* source, uint32* dest, uint32 width, uint32 height)
{

}

template <>
void census5x5_t_SSE(uint8* source, uint32* dest, uint32 width, uint32 height)
{
    census5x5_SSE(source, dest, width, height);
}

template <>
void census5x5_t_SSE(uint16* source, uint32* dest, uint32 width, uint32 height)
{
    census5x5_16bit_SSE(source, dest, width, height);
}

template <typename T>
void census9x7_t(T* source, uint64* dest, uint32 width, uint32 height)
{

}

template <>
void census9x7_t(uint8* source, uint64* dest, uint32 width, uint32 height)
{
    census9x7_mode8(source, dest, width, height);
}

template <>
void census9x7_t(uint16* source, uint64* dest, uint32 width, uint32 height)
{
    census9x7_mode8_16bit(source, dest, width, height);
}

template <typename T>
void costMeasureCensus9x7_xyd_t(T* intermediate1, T* intermediate2,int height, int width, int dispCount, uint16* dsi, int threads)
{

}

template <>
void costMeasureCensus9x7_xyd_t(uint64* intermediate1, uint64* intermediate2,int height, int width, int dispCount, uint16* dsi, int threads)
{
    costMeasureCensus9x7_xyd_parallel(intermediate1, intermediate2,height, width, dispCount, dsi, threads);
}

template<typename T>
void processCensus5x5SGM(T* leftImg, T* rightImg, float32* output, float32* dispImgRight,
    int width, int height, int method, uint16 paths, const int numThreads, const int numStrips, const int dispCount)
{
    const int maxDisp = dispCount - 1;

    std::cout << std::endl << "- " << method << ", " << paths << ", " << numThreads << ", " << numStrips << ", " << dispCount << std::endl;

    // get memory and init sgm params
    uint32* leftImgCensus = (uint32*)_mm_malloc(width*height*sizeof(uint32), 16);
    uint32* rightImgCensus = (uint32*)_mm_malloc(width*height*sizeof(uint32), 16);

    StereoSGMParams_t params;
    params.lrCheck = true;
    params.MedianFilter = true;
    params.Paths = paths;
    params.subPixelRefine = 0;
    params.NoPasses = 2;
    params.rlCheck = false;

    if (method == 0){
        uint16* dsi = (uint16*)_mm_malloc(width*height*(maxDisp + 1)*sizeof(uint16), 32);
        StereoSGM<T> m_sgm16(width, height, maxDisp, params);

        census5x5_t_SSE(leftImg, leftImgCensus, width, height);
        census5x5_t_SSE(rightImg, rightImgCensus, width, height);

        costMeasureCensus5x5_xyd_SSE(leftImgCensus, rightImgCensus, height, width, dispCount, params.InvalidDispCost, dsi, numThreads);

        if (numThreads > 1) {
            m_sgm16.processParallel(dsi, leftImg, output, dispImgRight, numThreads);
        }
        else {
            m_sgm16.process(dsi, leftImg, output, dispImgRight);
        }
        _mm_free(dsi);

    }
    else if (method == 1) {
        uint16* dsi = (uint16*)_mm_malloc(width*height*(maxDisp + 1)*sizeof(uint16), 32);

        StripedStereoSGM<T> stripedStereoSGM(width, height, maxDisp, numStrips, 16, params);

        census5x5_t_SSE(leftImg, leftImgCensus, width, height);
        census5x5_t_SSE(rightImg, rightImgCensus, width, height);

        costMeasureCensus5x5_xyd_SSE(leftImgCensus, rightImgCensus, height, width, dispCount, params.InvalidDispCost, dsi, numThreads);

        stripedStereoSGM.process(leftImg, output, dispImgRight, dsi, numThreads);

        _mm_free(dsi);

    }
    else if (method == 2) {
        const int maxDisp2 = 95;
        const sint32 dispSubsample = 2;
        uint16* dsi = (uint16*)_mm_malloc(width*height*(maxDisp2 + 1)*sizeof(uint16), 32);

        StripedStereoSGM<T> stripedStereoSGM(width, height, maxDisp2, numStrips, 16, params);

        census5x5_t_SSE(leftImg, leftImgCensus, width, height);
        census5x5_t_SSE(rightImg, rightImgCensus, width, height);

        costMeasureCensusCompressed5x5_xyd_SSE(leftImgCensus, rightImgCensus, height, width, dispCount, params.InvalidDispCost, dispSubsample, dsi, numThreads);

        stripedStereoSGM.process(leftImg, output, dispImgRight, dsi, numThreads);

        uncompressDisparities_SSE(output, width, height, dispSubsample);

        _mm_free(dsi);
    }
    else if (method == 3) {
        const int maxDisp2 = 79;
        const sint32 dispSubsample = 4;
        uint16* dsi = (uint16*)_mm_malloc(width*height*(maxDisp2 + 1)*sizeof(uint16), 32);

        StripedStereoSGM<T> stripedStereoSGM(width, height, maxDisp2, numStrips, 16, params);

        census5x5_t_SSE(leftImg, leftImgCensus, width, height);
        census5x5_t_SSE(rightImg, rightImgCensus, width, height);

        costMeasureCensusCompressed5x5_xyd_SSE(leftImgCensus, rightImgCensus, height, width, dispCount, params.InvalidDispCost, dispSubsample, dsi, numThreads);

        stripedStereoSGM.process(leftImg, output, dispImgRight, dsi, numThreads);

        uncompressDisparities_SSE(output, width, height, dispSubsample);

        _mm_free(dsi);

    }
}

template<typename T>
void processCensus9x7SGM(T* leftImg, T* rightImg, float32* output, float32* dispImgRight,
    int width, int height, int method, uint16 paths, const int numThreads, const int numStrips, const int dispCount)
{
    const int maxDisp = dispCount - 1;

    std::cout << std::endl << "- " << method << ", " << paths << ", " << numThreads << ", " << numStrips << ", " << dispCount << std::endl;

    // get memory and init sgm params
    uint64* leftImgCensus = (uint64*)_mm_malloc(width*height*sizeof(uint64), 16);
    uint64* rightImgCensus = (uint64*)_mm_malloc(width*height*sizeof(uint64), 16);

    StereoSGMParams_t params;
    params.lrCheck = true;
    params.MedianFilter = true;
    params.Paths = paths;
    params.subPixelRefine = 0;
    params.NoPasses = 2;
    params.rlCheck = false;
    params.InvalidDispCost = 16;
    params.Gamma = 100;
    params.Alpha = 0.f;

    if (method == 1) {
        uint16* dsi = (uint16*)_mm_malloc(width*height*(maxDisp + 1)*sizeof(uint16), 32);

        StripedStereoSGM<T> stripedStereoSGM(width, height, maxDisp, numStrips, 16, params);

#pragma omp parallel num_threads(2)
        {
#pragma omp sections nowait
            {
#pragma omp section
                {
                    census9x7_t<T>(leftImg, leftImgCensus, width, height);
                }
#pragma omp section
                {
                    census9x7_t<T>(rightImg, rightImgCensus, width, height);
                }
            }
        }

        costMeasureCensus9x7_xyd_t(leftImgCensus, rightImgCensus, height, width, dispCount, dsi, numThreads);

        stripedStereoSGM.process(leftImg, output, dispImgRight, dsi, numThreads);

        _mm_free(dsi);
    }
}

int main(int argc, char **argv)
{
    const int verbose = 1;

    // load parameter
    if (argc!=6) {
        std::cout << "expected imL,imR,disp,bitdepth,demo as params"<< std::endl;
        return -1;
    }
    char *im1name = argv[1];
    char *im2name = argv[2];
    char *dispname= argv[3];
    uint32 bitDepth = atoi(argv[4]);
    uint32 demo = atoi(argv[5]);
   
    fillPopCount16LUT();

    if (bitDepth == 16) {
        // load images
        MyImage<uint16> myImg1, myImg2;
        readPGM(myImg1, im1name);
        readPGM(myImg2, im2name);

        std::cout << "image 1 " << myImg1.getWidth() << "x" << myImg1.getHeight() << std::endl;
        std::cout << "image 2 " << myImg2.getWidth() << "x" << myImg2.getHeight() << std::endl;

        if (myImg1.getWidth() % 16 != 0) {
            std::cout << "Image width must be a multiple of 16" << std::endl;
            return 0;
        }
            
        MyImage<uint8> disp(myImg1.getWidth(), myImg1.getHeight());

        uint16* leftImg = (uint16*)_mm_malloc(myImg1.getWidth()*myImg1.getHeight()*sizeof(uint16), 16);
        uint16* rightImg = (uint16*)_mm_malloc(myImg2.getWidth()*myImg2.getHeight()*sizeof(uint16), 16);
        correctEndianness((uint16*)myImg1.getData(), leftImg, myImg1.getWidth()*myImg1.getHeight());
        correctEndianness((uint16*)myImg2.getData(), rightImg, myImg1.getWidth()*myImg1.getHeight());

        float32* dispImg = (float32*)_mm_malloc(myImg1.getWidth()*myImg1.getHeight()*sizeof(float32), 16);
        float32* dispImgRight = (float32*)_mm_malloc(myImg1.getWidth()*myImg1.getHeight()*sizeof(float32), 16);

        // start processing
        switch (demo)  {
        case 0:
            // standard SGM, 2 threads, 64 disparities
            processCensus5x5SGM(leftImg, rightImg, dispImg, dispImgRight, myImg1.getWidth(), myImg1.getHeight(), 0, 8, 2, 2, 64);
            break;
        case 1:
            // striped SGM, 4 threadsm, 64 disparities
            processCensus5x5SGM(leftImg, rightImg, dispImg, dispImgRight, myImg1.getWidth(), myImg1.getHeight(), 1, 8, 4, 4, 64);
            break;
        case 2:
            // striped SGM, 4 threads, 128 disparities
            processCensus5x5SGM(leftImg, rightImg, dispImg, dispImgRight, myImg1.getWidth(), myImg1.getHeight(), 1, 8, 4, 4, 128);
            break;
        case 3:
            // striped SGM, 4 threads, disparity compression with sub-sampling 2 (only implemented for 128)
            processCensus5x5SGM(leftImg, rightImg, dispImg, dispImgRight, myImg1.getWidth(), myImg1.getHeight(), 2, 8, 4, 4, 128);
            break;
        case 4:
            // striped SGM, 4 threads, disparity compression with sub-sampling 4 (only implemented for 128)
            processCensus5x5SGM(leftImg, rightImg, dispImg, dispImgRight, myImg1.getWidth(), myImg1.getHeight(), 3, 8, 4, 4, 128);
            break;
        case 5:
            // 9x7 HCWS Census measure, striped SGM, 4 threads, 128 disparities
            processCensus9x7SGM(leftImg, rightImg, dispImg, dispImgRight, myImg1.getWidth(), myImg1.getHeight(), 1, 8, 4, 4, 128);
        default:

            break;
        }

        // write output
        uint8* dispOut = disp.getData();
        for (uint32 i = 0; i < myImg1.getWidth()*myImg1.getHeight(); i++) {
            if (dispImg[i]>0) {
                dispOut[i] = (uint8)dispImg[i];
            }
            else {
                dispOut[i] = 0;
            }
        }

        std::string dispnamePlusDisp = dispname;
        writePGM(disp, dispnamePlusDisp.c_str(), verbose);

        // cleanup
        _mm_free(leftImg);
        _mm_free(rightImg);
        _mm_free(dispImg);
        _mm_free(dispImgRight);
    } else if (bitDepth==8) {
        // load images
        //readPGM(myImg1, im1name);
        //readPGM(myImg2, im2name);
        cv::Mat img1, img2;
        img1 = cv::imread(im1name, CV_LOAD_IMAGE_GRAYSCALE);
        img2 = cv::imread(im2name, CV_LOAD_IMAGE_GRAYSCALE);
        MyImage<uint8> myImg1(img1.cols, img1.rows);
        MyImage<uint8> myImg2(img2.cols, img2.rows);

        std::cout << "image 1 " << myImg1.getWidth() << "x" << myImg1.getHeight() << std::endl;
        std::cout << "image 2 " << myImg2.getWidth() << "x" << myImg2.getHeight() << std::endl;

        if (myImg1.getWidth() % 16 != 0) {
            std::cout << "Image width must be a multiple of 16" << std::endl;
            return 0;
        }

        MyImage<uint8> disp(myImg1.getWidth(), myImg1.getHeight());

        // start processing
        float32* dispImg = (float32*)_mm_malloc(myImg1.getWidth()*myImg1.getHeight()*sizeof(float32), 16);
        float32* dispImgRight = (float32*)_mm_malloc(myImg1.getWidth()*myImg1.getHeight()*sizeof(float32), 16);

        uint8* leftImg = myImg1.getData();
        uint8* rightImg = myImg2.getData();

        // start processing
        switch (demo)  {
        case 0:
            // standard SGM, 2 threads, 64 disparities
            processCensus5x5SGM(leftImg, rightImg, dispImg, dispImgRight, myImg1.getWidth(), myImg1.getHeight(), 0, 8, 2, 2, 64);
            break;
        case 1:
            // striped SGM, 4 threads, 64 disparities
            processCensus5x5SGM(leftImg, rightImg, dispImg, dispImgRight, myImg1.getWidth(), myImg1.getHeight(), 1, 8, 4, 4, 64);
            break;
        case 2:
            // striped SGM, 4 threads, 128 disparities
            processCensus5x5SGM(leftImg, rightImg, dispImg, dispImgRight, myImg1.getWidth(), myImg1.getHeight(), 1, 8, 4, 4, 128);
            break;
        case 3:
            // striped SGM, 4 threads, disparity compression with sub-sampling 2 (only implemented for 128)
            processCensus5x5SGM(leftImg, rightImg, dispImg, dispImgRight, myImg1.getWidth(), myImg1.getHeight(), 2, 8, 4, 4, 128);
            break;
        case 4:
            // striped SGM, 4 threads, disparity compression with sub-sampling 4 (only implemented for 128)
            processCensus5x5SGM(leftImg, rightImg, dispImg, dispImgRight, myImg1.getWidth(), myImg1.getHeight(), 3, 8, 4, 4, 128);
            break;
        case 5:
            // 9x7 HCWS Census measure, striped SGM, 4 threads, 128 disparities
            processCensus9x7SGM(leftImg, rightImg, dispImg, dispImgRight, myImg1.getWidth(), myImg1.getHeight(), 1, 8, 4, 4, 128);
        default:

            break;
        }

        // write output
        uint8* dispOut = disp.getData();
        for (uint32 i = 0; i < myImg1.getWidth()*myImg1.getHeight(); i++) {
            if (dispImg[i]>0) {
                dispOut[i] = (uint8)dispImg[i];
            }
            else {
                dispOut[i] = 0;
            }
        }

        std::string dispnamePlusDisp = dispname;
        writePGM(disp, dispnamePlusDisp.c_str(), verbose);

        // cleanup
        _mm_free(dispImg);
        _mm_free(dispImgRight);
    }
    
    return 0;
}
