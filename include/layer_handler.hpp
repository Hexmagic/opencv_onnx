#ifndef _LAYER_HANDLER_HPP_
#define _LAYER_HANDLER_HPP_
#include <opencv2/opencv.hpp>
#include <string>
#include <set>
#include "utils.hpp"
#include <opencv2/dnn.hpp>
using namespace cv;
using namespace cv::dnn;
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
static Mutex& getInitializationMutex()
        {
            static Mutex initializationMutex;
            return initializationMutex;
        }

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

#endif