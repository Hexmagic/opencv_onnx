#include "onnx_importer.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <opencv2/dnn.hpp>
#include <string>
using namespace cv;


Mat letterbox(Mat &img, Size new_shape, Scalar color, bool _auto, bool scaleFill, bool scaleup, int stride)
{
    float width = img.cols;
    float height = img.rows;
    float r = min(new_shape.width / width, new_shape.height / height);
    if (!scaleup)
        r = min(r, 1.0f);
    int new_unpadW = int(round(width * r));
    int new_unpadH = int(round(height * r));
    int dw = new_shape.width - new_unpadW;
    int dh = new_shape.height - new_unpadH;
    if (_auto)
    {
        dw %= stride;
        dh %= stride;
    }
    dw /= 2, dh /= 2;
    Mat dst;
    resize(img, dst, Size(new_unpadW, new_unpadH), 0, 0, INTER_LINEAR);
    int top = int(round(dh - 0.1));
    int bottom = int(round(dh + 0.1));
    int left = int(round(dw - 0.1));
    int right = int(round(dw + 0.1));
    copyMakeBorder(dst, dst, top, bottom, left, right, BORDER_CONSTANT, color);
    return dst;
}

int main(int argc, char **argv)
{
	GOOGLE_PROTOBUF_VERIFY_VERSION;
	std::string model_path;
	if (argc > 1)
	{
		model_path = argv[1];
	}
	else
	{
		model_path = "yolov5s.onnx";
	}
	CV_LOG_INFO(NULL, "Read Model From " << model_path)
	

	dnn::Net model =readONNX("yolov5s.onnx");
    Mat img0 = imread("images/bus.jpg");
    Mat img = letterbox(img0, Size(640, 640), Scalar(114, 114, 114), true, false, true, 32);
    Mat blob;    
    dnn::blobFromImage(img, blob, 1 / 255.0f, Size(img.cols, img.rows), Scalar(0, 0, 0), true, false);
    model.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);
    model.setPreferableTarget(dnn::DNN_TARGET_CPU);
    model.setInput(blob);
    std::vector<std::string> outLayerNames = model.getUnconnectedOutLayersNames();
    std::vector<Mat> result;
    model.forward(result, outLayerNames);
	Mat out = Mat(result[0].size[1],result[0].size[2],CV_32F,result[0].ptr<float>());
	CV_LOG_INFO(NULL, "Output Row "<<out.row(0));
	CV_LOG_INFO(NULL,"Native ONNX RUN PASS");

	return 0;
}