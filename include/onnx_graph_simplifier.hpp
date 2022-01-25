// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2020, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
#ifndef __OPENCV_DNN_ONNX_SIMPLIFIER_HPP__
#define __OPENCV_DNN_ONNX_SIMPLIFIER_HPP__
#include "precomp.hpp"
#include "utils.hpp"
#include "opencv-onnx.pb.h"
namespace cv
{   namespace dnn{

    void simplifySubgraphs(opencv_onnx::GraphProto &net);
    
        
}
}
#endif