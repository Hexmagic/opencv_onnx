#include "opencv-onnx.pb.h"
#include <opencv2/dnn.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/core/utils/trace.hpp>
#include <opencv2/dnn/all_layers.hpp>
#include "onnx_graph_simplifier.hpp"
using namespace cv;
using namespace cv::dnn;
using namespace std;

class LayerHandler
{
public:
	void addMissing(const std::string &name, const std::string &type);
	bool contains(const std::string &type) const;
	void printMissing();

protected:
	LayerParams getNotImplementedParams(const std::string &name, const std::string &op);

private:
	std::unordered_map<std::string, std::unordered_set<std::string>> layers;
};





Mutex &getLayerFactoryMutex()
{
	static Mutex *volatile instance = NULL;
	if (instance == NULL)
	{
		cv::AutoLock lock(getInitializationMutex());
		if (instance == NULL)
			instance = new Mutex();
	}
	return *instance;
}

typedef std::map<std::string, std::vector<cv::dnn::LayerFactory::Constructor>> LayerFactory_Impl;

static LayerFactory_Impl &getLayerFactoryImpl_()
{
	static LayerFactory_Impl impl;
	return impl;
}


LayerFactory_Impl &getLayerFactoryImpl()
{
	static LayerFactory_Impl *volatile instance = NULL;
	if (instance == NULL)
	{
		cv::AutoLock lock(getLayerFactoryMutex());
		if (instance == NULL)
		{
			instance = &getLayerFactoryImpl_();
			CV_TRACE_FUNCTION();
		}
	}
	return *instance;
}

void LayerHandler::addMissing(const std::string &name, const std::string &type)
{
	cv::AutoLock lock(getLayerFactoryMutex());
	auto &registeredLayers = getLayerFactoryImpl();

	// If we didn't add it, but can create it, it's custom and not missing.
	if (layers.find(type) == layers.end() && registeredLayers.find(type) != registeredLayers.end())
	{
		return;
	}

	layers[type].insert(name);
}

bool LayerHandler::contains(const std::string &type) const
{
	return layers.find(type) != layers.end();
}

void LayerHandler::printMissing()
{
	if (layers.empty())
	{
		return;
	}

	std::stringstream ss;
	ss << "DNN: Not supported types:\n";
	for (const auto &type_names : layers)
	{
		const auto &type = type_names.first;
		ss << "Type='" << type << "', affected nodes:\n[";
		for (const auto &name : type_names.second)
		{
			ss << "'" << name << "', ";
		}
		ss.seekp(-2, std::ios_base::end);
		ss << "]\n";
	}
	CV_LOG_ERROR(NULL, ss.str());
}

LayerParams LayerHandler::getNotImplementedParams(const std::string &name, const std::string &op)
{
	LayerParams lp;
	lp.name = name;
	lp.type = "NotImplemented";
	lp.set("type", op);

	return lp;
}

Mat getMatFromTensor(opencv_onnx::TensorProto& tensor_proto)
{
    if (tensor_proto.raw_data().empty() && tensor_proto.float_data().empty() &&
        tensor_proto.double_data().empty() && tensor_proto.int64_data().empty() &&
        tensor_proto.int32_data().empty())
        return Mat();

    opencv_onnx::TensorProto_DataType datatype = tensor_proto.data_type();
    Mat blob;
    std::vector<int> sizes;
    for (int i = 0; i < tensor_proto.dims_size(); i++) {
            sizes.push_back(tensor_proto.dims(i));
    }
    if (sizes.empty())
        sizes.assign(1, 1);
    if (datatype == opencv_onnx::TensorProto_DataType_FLOAT) {

        if (!tensor_proto.float_data().empty()) {
            const ::google::protobuf::RepeatedField<float> field = tensor_proto.float_data();
            Mat(sizes, CV_32FC1, (void*)field.data()).copyTo(blob);
        }
        else {
            char* val = const_cast<char*>(tensor_proto.raw_data().c_str());
            Mat(sizes, CV_32FC1, val).copyTo(blob);
        }
    }
    else if (datatype == opencv_onnx::TensorProto_DataType_DOUBLE)
    {
        const ::google::protobuf::RepeatedField<double> field = tensor_proto.double_data();
        CV_Assert(!field.empty());
        Mat(sizes, CV_64FC1, (void*)field.data()).convertTo(blob, CV_32FC1);
    }
    else if (datatype == opencv_onnx::TensorProto_DataType_INT32)
    {
        if (!tensor_proto.int32_data().empty())
        {
            const ::google::protobuf::RepeatedField<int32_t> field = tensor_proto.int32_data();
            Mat(sizes, CV_32SC1, (void*)field.data()).copyTo(blob);
        }
        else
        {
            char* val = const_cast<char*>(tensor_proto.raw_data().c_str());
            Mat(sizes, CV_32SC1, val).copyTo(blob);
        }
    }
    else if (datatype == opencv_onnx::TensorProto_DataType_INT64)
    {
        blob.create(sizes, CV_32SC1);
        int32_t* dst = reinterpret_cast<int32_t*>(blob.data);

        if (!tensor_proto.int64_data().empty()) {
            ::google::protobuf::RepeatedField< ::google::protobuf::int64> src = tensor_proto.int64_data();
            convertInt64ToInt32(src, dst, blob.total());
        }
        else
        {
            const char* val = tensor_proto.raw_data().c_str();
#if CV_STRONG_ALIGNMENT
            // Aligned pointer is required: https://github.com/opencv/opencv/issues/16373
            // this doesn't work: typedef int64_t CV_DECL_ALIGNED(1) unaligned_int64_t;
            AutoBuffer<int64_t, 16> aligned_val;
            if (!isAligned<sizeof(int64_t)>(val))
            {
                size_t sz = tensor_proto.raw_data().size();
                aligned_val.allocate(divUp(sz, sizeof(int64_t)));
                memcpy(aligned_val.data(), val, sz);
                val = (const char*)aligned_val.data();
            }
#endif
            const int64_t* src = reinterpret_cast<const int64_t*>(val);
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
            Mat(sizes, CV_32SC1, (void*)field.data()).convertTo(blob, CV_8S, 1.0, offset);
        }
        else
        {
            char* val = const_cast<char*>(tensor_proto.raw_data().c_str());
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
        blob.dims = 1;  // To force 1-dimensional cv::Mat for scalars.
    return blob;
}
void releaseONNXTensor(opencv_onnx::TensorProto& tensor_proto)
{
    if (!tensor_proto.raw_data().empty()) {
        delete tensor_proto.release_raw_data();
    }
}

class ONNXImporter
{
protected:
	std::unique_ptr<LayerHandler> layerhandler;
	Net &dstNet;

	opencv_onnx::GraphProto graph_proto;
	std::string framework_name;

	std::map<std::string, Mat> constBlobs;

	std::map<std::string, MatShape> outShapes; // List of internal blobs shapes.
	bool hasDynamicShapes;					   // Whether the model has inputs with dynamic shapes
	typedef std::map<std::string, MatShape>::iterator IterShape_t;

	
	int onnx_opset;
private:
	opencv_onnx::ModelProto model_proto;

public:
	ONNXImporter(Net &net, const char *onnxFile);
	void populateNet();
	std::map<std::string, Mat> getGraphTensors(
		const opencv_onnx::GraphProto &graph_proto);
};

std::map<std::string, Mat> ONNXImporter::getGraphTensors(
	const opencv_onnx::GraphProto &graph_proto)
{
	opencv_onnx::TensorProto tensor_proto;
	std::map<std::string, Mat> layers_weights;

	for (int i = 0; i < graph_proto.initializer_size(); i++)
	{
		tensor_proto = graph_proto.initializer(i);
		Mat mat = getMatFromTensor(tensor_proto);
		releaseONNXTensor(tensor_proto);

		if (mat.empty())
			continue;

		layers_weights.insert(std::make_pair(tensor_proto.name(), mat));
	}
	return layers_weights;
}

ONNXImporter::ONNXImporter(Net &net, const char *onnxFile)
	: layerhandler(new LayerHandler()), dstNet(net), onnx_opset(0)
{
	hasDynamicShapes = false;
	CV_Assert(onnxFile);
	CV_LOG_DEBUG(NULL, "DNN/ONNX: processing ONNX model from file: " << onnxFile);

	std::fstream input(onnxFile, std::ios::in | std::ios::binary);
	if (!input)
	{
		CV_Error(Error::StsBadArg, cv::format("Can't read ONNX file: %s", onnxFile));
	}

	if (!model_proto.ParseFromIstream(&input))
	{
		CV_Error(Error::StsUnsupportedFormat, cv::format("Failed to parse ONNX model: %s", onnxFile));
	}
	cout << model_proto.producer_name() << endl;
	populateNet();
}



void ONNXImporter::populateNet()
{
    CV_Assert(model_proto.has_graph());
    graph_proto = model_proto.graph();
	cout<<graph_proto.input_size() << endl;
    std::string framework_version;
    if (model_proto.has_producer_name())
        framework_name = model_proto.producer_name();
    if (model_proto.has_producer_version())
        framework_version = model_proto.producer_version();

    CV_LOG_INFO(NULL, "DNN/ONNX: loading ONNX"
            << (model_proto.has_ir_version() ? cv::format(" v%d", (int)model_proto.ir_version()) : cv::String())
            << " model produced by '" << framework_name << "'"
            << (framework_version.empty() ? cv::String() : cv::format(":%s", framework_version.c_str()))
            << ". Number of nodes = " << graph_proto.node_size()
            << ", inputs = " << graph_proto.input_size()
            << ", outputs = " << graph_proto.output_size()
            );

    simplifySubgraphs(graph_proto);

    const int layersSize = graph_proto.node_size();
    CV_LOG_DEBUG(NULL, "DNN/ONNX: graph simplified to " << layersSize << " nodes");

    constBlobs = getGraphTensors(graph_proto);
    // Add all the inputs shapes. It includes as constant blobs as network's inputs shapes.
    for (int i = 0; i < graph_proto.input_size(); ++i)
    {
        const opencv_onnx::ValueInfoProto& valueInfoProto = graph_proto.input(i);
        CV_Assert(valueInfoProto.has_name());
        CV_Assert(valueInfoProto.has_type());
        opencv_onnx::TypeProto typeProto = valueInfoProto.type();
        CV_Assert(typeProto.has_tensor_type());
        opencv_onnx::TypeProto::Tensor tensor = typeProto.tensor_type();
        CV_Assert(tensor.has_shape());
        opencv_onnx::TensorShapeProto tensorShape = tensor.shape();

        MatShape inpShape(tensorShape.dim_size());
        for (int j = 0; j < inpShape.size(); ++j)
        {
            inpShape[j] = tensorShape.dim(j).dim_value();
            // NHW, NCHW(NHWC), NCDHW(NDHWC); do not set this flag if only N is dynamic
            if (!tensorShape.dim(j).dim_param().empty() && !(j == 0 && inpShape.size() >= 3))
                hasDynamicShapes = true;
        }
        if (!inpShape.empty() && !hasDynamicShapes)
        {
            inpShape[0] = std::max(inpShape[0], 1); // It's OK to have undetermined batch size
        }
        outShapes[valueInfoProto.name()] = inpShape;
    }

    // create map with network inputs (without const blobs)
    // fill map: push layer name, layer id and output id
    std::vector<String> netInputs;
    for (int j = 0; j < graph_proto.input_size(); j++)
    {
        const std::string& name = graph_proto.input(j).name();
        if (constBlobs.find(name) == constBlobs.end()) {
            netInputs.push_back(name);
            // layer_id.insert(std::make_pair(name, LayerInfo(0, netInputs.size() - 1)));
        }
    }
    dstNet.setInputsNames(netInputs);

    if (1) {
        CV_LOG_INFO(NULL, "DNN/ONNX: start diagnostic run!");
        //layerhandler->fillRegistry(graph_proto);
    }

    // for(int li = 0; li < layersSize; li++)
    // {
    //     const opencv_onnx::NodeProto& node_proto = graph_proto.node(li);
    //     handleNode(node_proto);
    // }

    CV_LOG_DEBUG(NULL,  "DNN/ONNX: diagnostic run completed! DNN/ONNX: import completed!");
}

int main(int argc, char **argv)
{
	// GOOGLE_PROTOBUF_VERIFY_VERSION;
	string model_path;
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
	ONNXImporter importer(net, "exp.onnx");

	return 0;
}