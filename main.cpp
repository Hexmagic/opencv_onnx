#include "onnx_importer.hpp"
#include <string>

int main(int argc, char **argv)
{
	// GOOGLE_PROTOBUF_VERIFY_VERSION;
	std::string model_path;
	if (argc > 1)
	{
		model_path = argv[1];
	}
	else
	{
		model_path = "yolov5n.onnx";
	}
	CV_LOG_INFO(NULL, "Read Model From " << model_path)
	Net net;
	ONNXImporter importer(net, model_path.c_str());

	return 0;
}