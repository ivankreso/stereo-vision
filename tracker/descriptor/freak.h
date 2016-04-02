//  freak.h
//
//	Copyright (C) 2011-2012  Signal processing laboratory 2, EPFL,
//	Kirell Benzi (kirell.benzi@epfl.ch),
//	Raphael Ortiz (raphael.ortiz@a3.epfl.ch),
//	Alexandre Alahi (alexandre.alahi@epfl.ch),
//	and Pierre Vandergheynst (pierre.vandergheynst@epfl.ch)
//
//  Redistribution and use in source and binary forms, with or without modification,
//  are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
//  This software is provided by the copyright holders and contributors "as is" and
//  any express or implied warranties, including, but not limited to, the implied
//  warranties of merchantability and fitness for a particular purpose are disclaimed.
//  In no event shall the Intel Corporation or contributors be liable for any direct,
//  indirect, incidental, special, exemplary, or consequential damages
//  (including, but not limited to, procurement of substitute goods or services;
//  loss of use, data, or profits; or business interruption) however caused
//  and on any theory of liability, whether in contract, strict liability,
//  or tort (including negligence or otherwise) arising in any way out of
//  the use of this software, even if advised of the possibility of such damage.

#ifndef FREAKORIG_H_INCLUDED
#define FREAKORIG_H_INCLUDED

#include <opencv2/core/core.hpp>
#include <string>

namespace descriptor {

class FREAKorig : public cv::DescriptorExtractor
{
public:
    /** Constructor
         * @param orientationNormalized enable orientation normalization
         * @param scaleNormalized enable scale normalization
         * @param patternScale scaling of the description pattern
         * @param nbOctave number of octaves covered by the detected keypoints
         * @param selectedPairs (optional) user defined selected pairs
    */
    explicit FREAKorig( bool orientationNormalized = true
           , bool scaleNormalized = true
           , float patternScale = 22.0f
           , int nbOctave = 4
           , const std::vector<int>& selectedPairs = std::vector<int>()
         );

    virtual ~FREAKorig();

    // Not used TODO
    virtual void read( const cv::FileNode& );
    virtual void write( cv::FileStorage& ) const;

    /** returns the descriptor length in bytes */
    virtual int descriptorSize() const;

    /** returns the descriptor type */
    virtual int descriptorType() const;

    /** select the 512 "best description pairs"
         * @param images grayscale images set
         * @param keypoints set of detected keypoints
         * @param corrThresh correlation threshold
         * @param verbose print construction information
         * @return list of best pair indexes
    */
    std::vector<int> selectPairs( const std::vector<cv::Mat>& images, std::vector<std::vector<cv::KeyPoint> >& keypoints,
                      const double corrThresh = 0.7, bool verbose = true );

//    AlgorithmInfo* info() const;

protected:
    virtual void computeImpl( const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) const;

protected:
    struct FREAKImpl* impl;

private:
    FREAKorig( const FREAKorig& rhs ); // do not allow copy constructor
    const FREAKorig& operator=( const FREAKorig& ); // nor assignement operator
};

} // END NAMESPACE CV

#endif // FREAK_H_INCLUDED
