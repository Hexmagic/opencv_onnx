#include "onnx_importer.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <opencv2/dnn.hpp>
#include <string>
using namespace cv;


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
	Net net;
	ONNXImporter importer(net, model_path.c_str());
	net.setPreferableBackend(DNN_BACKEND_OPENCV);
	net.setPreferableTarget(DNN_TARGET_CPU);
	Mat img = imread("images/bus.jpg");
	resize(img,img,Size(640,640),0,0,INTER_LINEAR);
	Mat blob;
	blobFromImage(img,1/255.0f,Size(640,640),Scalar(0,0,0),true,false);
	net.setInput(blob);
	std::vector<Mat> outputs;
	std::vector<std::string> outLayerNames=net.getUnconnectedOutLayersNames();
	net.forward(outputs,outLayerNames);
	return 0;
}