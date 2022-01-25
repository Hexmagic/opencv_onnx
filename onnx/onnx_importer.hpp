#ifndef __ONNX_IMPORTER_HPP_
#define __ONNX_IMPORTER_HPP_
#include "opencv-onnx.pb.h"

#include "layer_handler.hpp"
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <opencv2/core/utils/logger.hpp>
#include <fstream>
using namespace cv;
using namespace cv::dnn;
class ONNXLayerHandler;

inline void replaceLayerParam(LayerParams& layerParams, const String& oldKey, const String& newKey)
{
    if (layerParams.has(oldKey)) {
        layerParams.set(newKey, layerParams.get(oldKey));
        layerParams.erase(oldKey);
    }
}


void runLayer(LayerParams& params, const std::vector<Mat>& inputs,
              std::vector<Mat>& outputs)
{
    Ptr<Layer> layer = LayerFactory::createLayerInstance(params.type, params);
    CV_Assert((bool)layer);

    std::vector<MatShape> inpShapes(inputs.size());
    int ddepth = params.get<int>("depth", CV_32F);
    for (size_t i = 0; i < inputs.size(); ++i)
    {
        inpShapes[i] = shape(inputs[i]);
        if (i > 0 && ddepth != inputs[i].depth())
            CV_Error(Error::StsNotImplemented, "Mixed input data types.");
        ddepth = inputs[i].depth();
    }

    std::vector<MatShape> outShapes, internalShapes;
    layer->getMemoryShapes(inpShapes, 0, outShapes, internalShapes);

    std::vector<Mat> internals(internalShapes.size());
    outputs.resize(outShapes.size());
    for (size_t i = 0; i < outShapes.size(); ++i)
        outputs[i].create(outShapes[i], ddepth);
    for (size_t i = 0; i < internalShapes.size(); ++i)
        internals[i].create(internalShapes[i], ddepth);

    layer->finalize(inputs, outputs);
    layer->forward(inputs, outputs, internals);
}

class ONNXImporter
{
 public:
    Mat getBlob(const opencv_onnx::NodeProto& node_proto, int index);
    Mat getBlob(const std::string& input_name);

    LayerParams getLayerParams(const opencv_onnx::NodeProto& node_proto);
    bool isCeilMode(const LayerParams& layerParams);

    void addConstant(const std::string& name, const Mat& blob);
    void addLayer(LayerParams& layerParams,
                  const opencv_onnx::NodeProto& node_proto);
    void handleQuantizedNode(LayerParams& layerParams,
                             const opencv_onnx::NodeProto& node_proto);

    void expandMid(const std::string& prefix, opencv_onnx::NodeProto& node_proto,
                   const std::string& input, size_t n);
    struct LayerInfo {
        int layerId;
        int outputId;
        LayerInfo(int _layerId = 0, int _outputId = 0) : layerId(_layerId), outputId(_outputId) {}
    };             
        typedef std::map<std::string, LayerInfo>::iterator IterLayerId_t;       
     std::map<std::string, LayerInfo> layer_id;
protected:
    std::unique_ptr<ONNXLayerHandler> layerhandler;
    Net &dstNet;

    opencv_onnx::GraphProto graph_proto;
    std::string framework_name;

    std::map<std::string, Mat> constBlobs;

    std::map<std::string, MatShape> outShapes; // List of internal blobs shapes.
    bool hasDynamicShapes;                     // Whether the model has inputs with dynamic shapes
    typedef std::map<std::string, MatShape>::iterator IterShape_t;

    int onnx_opset;

private:
    friend class ONNXLayerHandler;
    opencv_onnx::ModelProto model_proto;
    void parseMaxPool(LayerParams &layerParams, const opencv_onnx::NodeProto &node_proto);
    void parseAveragePool(LayerParams &layerParams, const opencv_onnx::NodeProto &node_proto);
    void parseReduce(LayerParams &layerParams, const opencv_onnx::NodeProto &node_proto);
    void parseSlice(LayerParams &layerParams, const opencv_onnx::NodeProto &node_proto);
    void parseSplit(LayerParams &layerParams, const opencv_onnx::NodeProto &node_proto);
    void parseBias(LayerParams &layerParams, const opencv_onnx::NodeProto &node_proto);
    void parsePow(LayerParams &layerParams, const opencv_onnx::NodeProto &node_proto);
    void parseMinMax(LayerParams &layerParams, const opencv_onnx::NodeProto &node_proto);
    void parseNeg(LayerParams &layerParams, const opencv_onnx::NodeProto &node_proto);
    void parseConstant(LayerParams &layerParams, const opencv_onnx::NodeProto &node_proto);
    void parseLSTM(LayerParams &layerParams, const opencv_onnx::NodeProto &node_proto);
    void parseGRU(LayerParams &layerParams, const opencv_onnx::NodeProto &node_proto);
    void parseImageScaler(LayerParams &layerParams, const opencv_onnx::NodeProto &node_proto);
    void parseClip(LayerParams &layerParams, const opencv_onnx::NodeProto &node_proto);
    void parseLeakyRelu(LayerParams &layerParams, const opencv_onnx::NodeProto &node_proto);
    void parseRelu(LayerParams &layerParams, const opencv_onnx::NodeProto &node_proto);
    void parseElu(LayerParams &layerParams, const opencv_onnx::NodeProto &node_proto);
    void parseTanh(LayerParams &layerParams, const opencv_onnx::NodeProto &node_proto);
    void parsePRelu(LayerParams &layerParams, const opencv_onnx::NodeProto &node_proto);
    void parseLRN(LayerParams &layerParams, const opencv_onnx::NodeProto &node_proto);
    void parseInstanceNormalization(LayerParams &layerParams, const opencv_onnx::NodeProto &node_proto);
    void parseBatchNormalization(LayerParams &layerParams, const opencv_onnx::NodeProto &node_proto);
    void parseGemm(LayerParams &layerParams, const opencv_onnx::NodeProto &node_proto);
    void parseMatMul(LayerParams &layerParams, const opencv_onnx::NodeProto &node_proto);
    void parseMul(LayerParams &layerParams, const opencv_onnx::NodeProto &node_proto);
    void parseConv(LayerParams &layerParams, const opencv_onnx::NodeProto &node_proto);
    void parseConvTranspose(LayerParams &layerParams, const opencv_onnx::NodeProto &node_proto);
    void parseTranspose(LayerParams &layerParams, const opencv_onnx::NodeProto &node_proto);
    void parseSqueeze(LayerParams &layerParams, const opencv_onnx::NodeProto &node_proto);
    void parseFlatten(LayerParams &layerParams, const opencv_onnx::NodeProto &node_proto);
    void parseUnsqueeze(LayerParams &layerParams, const opencv_onnx::NodeProto &node_proto);
    void parseExpand(LayerParams &layerParams, const opencv_onnx::NodeProto &node_proto);
    void parseReshape(LayerParams &layerParams, const opencv_onnx::NodeProto &node_proto);
    void parsePad(LayerParams &layerParams, const opencv_onnx::NodeProto &node_proto);
    void parseShape(LayerParams &layerParams, const opencv_onnx::NodeProto &node_proto);
    void parseCast(LayerParams &layerParams, const opencv_onnx::NodeProto &node_proto);
    void parseConstantFill(LayerParams &layerParams, const opencv_onnx::NodeProto &node_proto);
    void parseGather(LayerParams &layerParams, const opencv_onnx::NodeProto &node_proto);
    void parseConcat(LayerParams &layerParams, const opencv_onnx::NodeProto &node_proto);
    void parseResize(LayerParams &layerParams, const opencv_onnx::NodeProto &node_proto);
    void parseUpsample(LayerParams &layerParams, const opencv_onnx::NodeProto &node_proto);
    void parseSoftMax(LayerParams &layerParams, const opencv_onnx::NodeProto &node_proto);
    void parseDetectionOutput(LayerParams &layerParams, const opencv_onnx::NodeProto &node_proto);
    void parseCumSum(LayerParams &layerParams, const opencv_onnx::NodeProto &node_proto);
    void parseQuantDequant(LayerParams &layerParams, const opencv_onnx::NodeProto &node_proto);
    void parseQConv(LayerParams &layerParams, const opencv_onnx::NodeProto &node_proto);
    void parseQMatMul(LayerParams &layerParams, const opencv_onnx::NodeProto &node_proto);
    void parseQEltwise(LayerParams &layerParams, const opencv_onnx::NodeProto &node_proto);
    void parseQLeakyRelu(LayerParams &layerParams, const opencv_onnx::NodeProto &node_proto);
    void parseQSigmoid(LayerParams &layerParams, const opencv_onnx::NodeProto &node_proto);
    void parseQAvgPool(LayerParams &layerParams, const opencv_onnx::NodeProto &node_proto);
    void parseQConcat(LayerParams &layerParams, const opencv_onnx::NodeProto &node_proto);

    void parseCustomLayer(LayerParams &layerParams, const opencv_onnx::NodeProto &node_proto);

public:
    typedef void (ONNXImporter::*ONNXImporterNodeParser)(LayerParams &layerParams, const opencv_onnx::NodeProto &node_proto);
    typedef std::map<std::string, ONNXImporterNodeParser> DispatchMap;
    const DispatchMap dispatch;
    static const ONNXImporter::DispatchMap buildDispatchMap();
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

class ONNXLayerHandler : public LayerHandler
{
public:
    explicit ONNXLayerHandler(ONNXImporter *importer_);

    void fillRegistry(const opencv_onnx::GraphProto &net);

protected:
    ONNXImporter *importer;
};

ONNXLayerHandler::ONNXLayerHandler(ONNXImporter *importer_) : importer(importer_) {}

void ONNXLayerHandler::fillRegistry(const opencv_onnx::GraphProto &net)
{
    int layersSize = net.node_size();
    for (int li = 0; li < layersSize; li++)
    {
        const opencv_onnx::NodeProto &node_proto = net.node(li);
        const std::string &name = node_proto.output(0);
        const std::string &type = node_proto.op_type();
        if (importer->dispatch.find(type) == importer->dispatch.end())
        {
            addMissing(name, type);
        }
    }
    printMissing();
}

ONNXImporter::ONNXImporter(Net &net, const char *onnxFile)
    : layerhandler(new ONNXLayerHandler(this)), dstNet(net), onnx_opset(0), dispatch(buildDispatchMap())
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
    CV_LOG_INFO(NULL, "producer Name" << model_proto.producer_name());
    populateNet();
}

const ONNXImporter::DispatchMap ONNXImporter::buildDispatchMap()
{
    DispatchMap dispatch;

    dispatch["MaxPool"] = &ONNXImporter::parseMaxPool;
    dispatch["AveragePool"] = &ONNXImporter::parseAveragePool;
    dispatch["GlobalAveragePool"] = dispatch["GlobalMaxPool"] = dispatch["ReduceMean"] = dispatch["ReduceSum"] =
        dispatch["ReduceMax"] = &ONNXImporter::parseReduce;
    dispatch["Slice"] = &ONNXImporter::parseSlice;
    dispatch["Split"] = &ONNXImporter::parseSplit;
    dispatch["Add"] = dispatch["Sum"] = dispatch["Sub"] = &ONNXImporter::parseBias;
    dispatch["Pow"] = &ONNXImporter::parsePow;
    dispatch["Min"] = dispatch["Max"] = &ONNXImporter::parseMinMax;
    dispatch["Neg"] = &ONNXImporter::parseNeg;
    dispatch["Constant"] = &ONNXImporter::parseConstant;
    dispatch["LSTM"] = &ONNXImporter::parseLSTM;
    dispatch["GRU"] = &ONNXImporter::parseGRU;
    dispatch["ImageScaler"] = &ONNXImporter::parseImageScaler;
    dispatch["Clip"] = &ONNXImporter::parseClip;
    dispatch["LeakyRelu"] = &ONNXImporter::parseLeakyRelu;
    dispatch["Relu"] = &ONNXImporter::parseRelu;
    dispatch["Elu"] = &ONNXImporter::parseElu;
    dispatch["Tanh"] = &ONNXImporter::parseTanh;
    dispatch["PRelu"] = &ONNXImporter::parsePRelu;
    dispatch["LRN"] = &ONNXImporter::parseLRN;
    dispatch["InstanceNormalization"] = &ONNXImporter::parseInstanceNormalization;
    dispatch["BatchNormalization"] = &ONNXImporter::parseBatchNormalization;
    dispatch["Gemm"] = &ONNXImporter::parseGemm;
    dispatch["MatMul"] = &ONNXImporter::parseMatMul;
    dispatch["Mul"] = dispatch["Div"] = &ONNXImporter::parseMul;
    dispatch["Conv"] = &ONNXImporter::parseConv;
    dispatch["ConvTranspose"] = &ONNXImporter::parseConvTranspose;
    dispatch["Transpose"] = &ONNXImporter::parseTranspose;
    dispatch["Squeeze"] = &ONNXImporter::parseSqueeze;
    dispatch["Flatten"] = &ONNXImporter::parseFlatten;
    dispatch["Unsqueeze"] = &ONNXImporter::parseUnsqueeze;
    dispatch["Expand"] = &ONNXImporter::parseExpand;
    dispatch["Reshape"] = &ONNXImporter::parseReshape;
    dispatch["Pad"] = &ONNXImporter::parsePad;
    dispatch["Shape"] = &ONNXImporter::parseShape;
    dispatch["Cast"] = &ONNXImporter::parseCast;
    dispatch["ConstantFill"] = dispatch["ConstantOfShape"] = &ONNXImporter::parseConstantFill;
    dispatch["Gather"] = &ONNXImporter::parseGather;
    dispatch["Concat"] = &ONNXImporter::parseConcat;
    dispatch["Resize"] = &ONNXImporter::parseResize;
    dispatch["Upsample"] = &ONNXImporter::parseUpsample;
    dispatch["SoftMax"] = dispatch["LogSoftmax"] = &ONNXImporter::parseSoftMax;
    dispatch["DetectionOutput"] = &ONNXImporter::parseDetectionOutput;
    dispatch["CumSum"] = &ONNXImporter::parseCumSum;
    dispatch["QuantizeLinear"] = dispatch["DequantizeLinear"] = &ONNXImporter::parseQuantDequant;
    dispatch["QLinearConv"] = &ONNXImporter::parseQConv;
    dispatch["QLinearMatMul"] = &ONNXImporter::parseQMatMul;
    dispatch["QLinearAdd"] = dispatch["QLinearMul"] = &ONNXImporter::parseQEltwise;
    dispatch["QLinearLeakyRelu"] = &ONNXImporter::parseQLeakyRelu;
    dispatch["QLinearSigmoid"] = &ONNXImporter::parseQSigmoid;
    dispatch["QLinearAveragePool"] = dispatch["QLinearGlobalAveragePool"] = &ONNXImporter::parseQAvgPool;
    dispatch["QLinearConcat"] = &ONNXImporter::parseQConcat;

    return dispatch;
}

void ONNXImporter::populateNet()
{
    CV_Assert(model_proto.has_graph());
    graph_proto = model_proto.graph();
    CV_LOG_INFO(NULL, "Grah Proto Input Size" << graph_proto.input_size());
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
                          << ", outputs = " << graph_proto.output_size());
    const int org_layersSize = graph_proto.node_size();
    simplifySubgraphs(graph_proto);

    const int layersSize = graph_proto.node_size();
    CV_LOG_INFO(NULL, "DNN/ONNX: graph org size " << org_layersSize << "  simplified to " << layersSize << " nodes");
    // weights
    constBlobs = getGraphTensors(graph_proto);
    // Add all the inputs shapes. It includes as constant blobs as network's inputs shapes.
    for (int i = 0; i < graph_proto.input_size(); ++i)
    {
        const opencv_onnx::ValueInfoProto &valueInfoProto = graph_proto.input(i);
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
        const std::string &name = graph_proto.input(j).name();
        if (constBlobs.find(name) == constBlobs.end())
        {
            netInputs.push_back(name);
            // layer_id.insert(std::make_pair(name, LayerInfo(0, netInputs.size() - 1)));
        }
    }
    dstNet.setInputsNames(netInputs);

    if (1)
    {
        CV_LOG_INFO(NULL, "DNN/ONNX: start diagnostic run!");
        // layerhandler->fillRegistry(graph_proto);
    }

    // for(int li = 0; li < layersSize; li++)
    // {
    //     const opencv_onnx::NodeProto& node_proto = graph_proto.node(li);
    //     handleNode(node_proto);
    // }

    CV_LOG_DEBUG(NULL, "DNN/ONNX: diagnostic run completed! DNN/ONNX: import completed!");
}

#endif //