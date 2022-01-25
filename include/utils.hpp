#ifndef _UTILS_HPP_Mix
#define _UTILS_HPP_Mix
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/core/utils/trace.hpp>
#include "opencv-onnx.pb.h"
#include <opencv2/core/utils/logger.hpp>
using namespace cv;
template <typename T1, typename T2>

void convertInt64ToInt32(const T1 &src, T2 &dst, int size)
    {
        for (int i = 0; i < size; i++)
        {
            dst[i] = cv::saturate_cast<int>(src[i]);
        }
    }

Mat getMatFromTensor(opencv_onnx::TensorProto &tensor_proto)
{
    if (tensor_proto.raw_data().empty() && tensor_proto.float_data().empty() &&
        tensor_proto.double_data().empty() && tensor_proto.int64_data().empty() &&
        tensor_proto.int32_data().empty())
        return Mat();

    opencv_onnx::TensorProto_DataType datatype = tensor_proto.data_type();
    Mat blob;
    std::vector<int> sizes;
    for (int i = 0; i < tensor_proto.dims_size(); i++)
    {
        sizes.push_back(tensor_proto.dims(i));
    }
    if (sizes.empty())
        sizes.assign(1, 1);
    if (datatype == opencv_onnx::TensorProto_DataType_FLOAT)
    {

        if (!tensor_proto.float_data().empty())
        {
            const ::google::protobuf::RepeatedField<float> field = tensor_proto.float_data();
            Mat(sizes, CV_32FC1, (void *)field.data()).copyTo(blob);
        }
        else
        {
            char *val = const_cast<char *>(tensor_proto.raw_data().c_str());
            Mat(sizes, CV_32FC1, val).copyTo(blob);
        }
    }
    else if (datatype == opencv_onnx::TensorProto_DataType_DOUBLE)
    {
        const ::google::protobuf::RepeatedField<double> field = tensor_proto.double_data();
        CV_Assert(!field.empty());
        Mat(sizes, CV_64FC1, (void *)field.data()).convertTo(blob, CV_32FC1);
    }
    else if (datatype == opencv_onnx::TensorProto_DataType_INT32)
    {
        if (!tensor_proto.int32_data().empty())
        {
            const ::google::protobuf::RepeatedField<int32_t> field = tensor_proto.int32_data();
            Mat(sizes, CV_32SC1, (void *)field.data()).copyTo(blob);
        }
        else
        {
            char *val = const_cast<char *>(tensor_proto.raw_data().c_str());
            Mat(sizes, CV_32SC1, val).copyTo(blob);
        }
    }
    else if (datatype == opencv_onnx::TensorProto_DataType_INT64)
    {
        blob.create(sizes, CV_32SC1);
        int32_t *dst = reinterpret_cast<int32_t *>(blob.data);

        if (!tensor_proto.int64_data().empty())
        {
            ::google::protobuf::RepeatedField<::google::protobuf::int64> src = tensor_proto.int64_data();
            convertInt64ToInt32(src, dst, blob.total());
        }
        else
        {
            const char *val = tensor_proto.raw_data().c_str();
#if CV_STRONG_ALIGNMENT
            // Aligned pointer is required: https://github.com/opencv/opencv/issues/16373
            // this doesn't work: typedef int64_t CV_DECL_ALIGNED(1) unaligned_int64_t;
            AutoBuffer<int64_t, 16> aligned_val;
            if (!isAligned<sizeof(int64_t)>(val))
            {
                size_t sz = tensor_proto.raw_data().size();
                aligned_val.allocate(divUp(sz, sizeof(int64_t)));
                memcpy(aligned_val.data(), val, sz);
                val = (const char *)aligned_val.data();
            }
#endif
            const int64_t *src = reinterpret_cast<const int64_t *>(val);
            convertInt64ToInt32(src, dst, blob.total());
        }
    }
    else if (datatype == opencv_onnx::TensorProto_DataType_INT8 ||
             datatype == opencv_onnx::TensorProto_DataType_UINT8)
    {
        // TODO : Add support for uint8 weights and acitvations. For now, converting uint8 tensors to int8.
        int offset = datatype == opencv_onnx::TensorProto_DataType_INT8 ? 0 : -128;
        int depth = datatype == opencv_onnx::TensorProto_DataType_INT8 ? CV_8S : CV_8U;

        if (!tensor_proto.int32_data().empty())
        {
            const ::google::protobuf::RepeatedField<int32_t> field = tensor_proto.int32_data();
            Mat(sizes, CV_32SC1, (void *)field.data()).convertTo(blob, CV_8S, 1.0, offset);
        }
        else
        {
            char *val = const_cast<char *>(tensor_proto.raw_data().c_str());
            Mat(sizes, depth, val).convertTo(blob, CV_8S, 1.0, offset);
        }
    }
    else
    {
        std::string errorMsg = "Unsupported data type: " +
                               opencv_onnx::TensorProto_DataType_Name(datatype);

        CV_Error(Error::StsUnsupportedFormat, errorMsg);
        CV_LOG_ERROR(NULL, errorMsg);
        return blob;
    }
    if (tensor_proto.dims_size() == 0)
        blob.dims = 1; // To force 1-dimensional cv::Mat for scalars.
    return blob;
}
void releaseONNXTensor(opencv_onnx::TensorProto &tensor_proto)
{
    if (!tensor_proto.raw_data().empty())
    {
        delete tensor_proto.release_raw_data();
    }
}
#endif