// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2020, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef __GRAPH_SIMPLIFIER_HPP__
#define __GRAPH_SIMPLIFIER_HPP__

#include <string>
#include "utils.hpp"
#include <opencv2/core.hpp>

namespace cv
{
    namespace dnn
    {

        class ImportNodeWrapper
        {
        public:
            virtual ~ImportNodeWrapper(){};

            virtual int getNumInputs() const = 0;

            virtual std::string getInputName(int idx) const = 0;

            virtual std::string getType() const = 0;

            virtual void setType(const std::string &type) = 0;

            virtual void setInputNames(const std::vector<std::string> &inputs) = 0;
        };

        class ImportGraphWrapper
        {
        public:
            virtual ~ImportGraphWrapper(){};

            virtual Ptr<ImportNodeWrapper> getNode(int idx) const = 0;

            virtual int getNumNodes() const = 0;

            virtual int getNumOutputs(int nodeId) const = 0;

            virtual std::string getOutputName(int nodeId, int outId) const = 0;

            virtual void removeNode(int idx) = 0;
        };

        class Subgraph // Interface to match and replace subgraphs.
        {
        public:
            virtual ~Subgraph();

            // Add a node to be matched in the origin graph. Specify ids of nodes that
            // are expected to be inputs. Returns id of a newly added node.
            // TODO: Replace inputs to std::vector<int> in C++11
            int addNodeToMatch(const std::string &op, int input_0 = -1, int input_1 = -1,
                               int input_2 = -1, int input_3 = -1);

            int addNodeToMatch(const std::string &op, const std::vector<int> &inputs_);

            // Specify resulting node. All the matched nodes in subgraph excluding
            // input nodes will be fused into this single node.
            // TODO: Replace inputs to std::vector<int> in C++11
            void setFusedNode(const std::string &op, int input_0 = -1, int input_1 = -1,
                              int input_2 = -1, int input_3 = -1, int input_4 = -1,
                              int input_5 = -1);

            void setFusedNode(const std::string &op, const std::vector<int> &inputs_);

            static int getInputNodeId(const Ptr<ImportGraphWrapper> &net,
                                      const Ptr<ImportNodeWrapper> &node,
                                      int inpId);

            // Match TensorFlow subgraph starting from <nodeId> with a set of nodes to be fused.
            // Const nodes are skipped during matching. Returns true if nodes are matched and can be fused.
            virtual bool match(const Ptr<ImportGraphWrapper> &net, int nodeId,
                               std::vector<int> &matchedNodesIds,
                               std::vector<int> &targetNodesIds);

            // Fuse matched subgraph.
            void replace(const Ptr<ImportGraphWrapper> &net, const std::vector<int> &matchedNodesIds,
                         const std::vector<int> &targetNodesIds);

            virtual void finalize(const Ptr<ImportGraphWrapper> &net,
                                  const Ptr<ImportNodeWrapper> &fusedNode,
                                  std::vector<Ptr<ImportNodeWrapper>> &inputs);

        private:
            std::vector<std::string> nodes;       // Nodes to be matched in the origin graph.
            std::vector<std::vector<int>> inputs; // Connections of an every node to it's inputs.

            std::string fusedNodeOp;          // Operation name of resulting fused node.
            std::vector<int> fusedNodeInputs; // Inputs of fused node.
        };

        Subgraph::~Subgraph() {}

        int Subgraph::addNodeToMatch(const std::string &op, int input_0, int input_1,
                                     int input_2, int input_3)
        {
            int nodeInputs[] = {input_0, input_1, input_2, input_3};
            int numInputs = 0;
            for (int i = 0; i < 4; ++i)
            {
                numInputs += (int)(nodeInputs[i] != -1);
            }
            return addNodeToMatch(op, std::vector<int>(&nodeInputs[0], &nodeInputs[0] + numInputs));
        }

        int Subgraph::addNodeToMatch(const std::string &op, const std::vector<int> &inputs_)
        {
            for (int i = 0; i < inputs_.size(); ++i)
            {
                CV_Assert(inputs_[i] < (int)nodes.size());
            }
            nodes.push_back(op);
            inputs.push_back(inputs_);
            return nodes.size() - 1;
        }

        void Subgraph::setFusedNode(const std::string &op, int input_0, int input_1,
                                    int input_2, int input_3, int input_4, int input_5)
        {
            int nodeInputs[] = {input_0, input_1, input_2, input_3, input_4, input_5};
            int numInputs = 0;
            for (int i = 0; i < 6; ++i)
            {
                CV_Assert(nodeInputs[i] < (int)nodes.size());
                numInputs += (int)(nodeInputs[i] != -1);
            }
            setFusedNode(op, std::vector<int>(&nodeInputs[0], &nodeInputs[0] + numInputs));
        }

        void Subgraph::setFusedNode(const std::string &op, const std::vector<int> &inputs_)
        {
            fusedNodeInputs = inputs_;
            fusedNodeOp = op;
        }

        int Subgraph::getInputNodeId(const Ptr<ImportGraphWrapper> &net,
                                     const Ptr<ImportNodeWrapper> &node,
                                     int inpId)
        {
            CV_Assert(inpId < node->getNumInputs());
            std::string name = node->getInputName(inpId);
            const int numNodes = net->getNumNodes();
            for (int i = 0; i < numNodes; ++i)
            {
                const int numOutputs = net->getNumOutputs(i);
                for (int j = 0; j < numOutputs; j++)
                {
                    if (net->getOutputName(i, j) == name)
                        return i;
                }
            }
            CV_Error(Error::StsParseError, "Input node with name " + name + " not found");
        }

        bool Subgraph::match(const Ptr<ImportGraphWrapper> &net, int nodeId,
                             std::vector<int> &matchedNodesIds,
                             std::vector<int> &targetNodesIds)
        {
            matchedNodesIds.clear();
            targetNodesIds.clear();

            std::queue<int> nodesToMatch;
            std::queue<int> targetNodes;
            nodesToMatch.push(nodeId);
            targetNodes.push(nodes.size() - 1);
            while (!nodesToMatch.empty())
            {
                int nodeToMatch = nodesToMatch.front();
                int targetNodeId = targetNodes.front();
                nodesToMatch.pop();
                targetNodes.pop();

                if (std::find(matchedNodesIds.begin(), matchedNodesIds.end(), nodeToMatch) !=
                    matchedNodesIds.end())
                    continue;

                const Ptr<ImportNodeWrapper> node = net->getNode(nodeToMatch);
                if (node->getType() != nodes[targetNodeId])
                    return false;

                std::vector<int> &inputNodes = inputs[targetNodeId];
                if (inputNodes.size() != node->getNumInputs())
                    return false;

                for (int j = 0; j < inputNodes.size(); ++j)
                {
                    if (nodes[inputNodes[j]].empty()) // Unknown input node type.
                        continue;
                    nodeId = getInputNodeId(net, node, j);
                    const Ptr<ImportNodeWrapper> inpNode = net->getNode(nodeId);
                    if (inpNode->getType() != "Const" && inpNode->getType() != "Constant")
                    {
                        nodesToMatch.push(nodeId);
                        targetNodes.push(inputNodes[j]);
                    }
                    else if (nodes[inputNodes[j]] != "Const" && nodes[inputNodes[j]] != "Constant")
                        return false;
                }
                matchedNodesIds.push_back(nodeToMatch);
                targetNodesIds.push_back(targetNodeId);
            }

            const int n = matchedNodesIds.size();
            std::vector<std::pair<int, int>> elements(n);
            for (int i = 0; i < n; ++i)
                elements[i] = std::make_pair(matchedNodesIds[i], targetNodesIds[i]);
            std::sort(elements.begin(), elements.end());
            for (int i = 0; i < n; ++i)
            {
                matchedNodesIds[i] = elements[i].first;
                targetNodesIds[i] = elements[i].second;
            }
            return true;
        }

        void Subgraph::replace(const Ptr<ImportGraphWrapper> &net, const std::vector<int> &matchedNodesIds,
                               const std::vector<int> &targetNodesIds)
        {
            // Extract names of input nodes.
            std::vector<std::string> inputsNames(fusedNodeInputs.size());
            for (int i = 0; i < fusedNodeInputs.size(); ++i)
            {
                std::string inpName;
                // Find input node name looking at inputs of fused nodes.
                for (int j = 0; j < matchedNodesIds.size() && inpName.empty(); ++j)
                {
                    Ptr<ImportNodeWrapper> node = net->getNode(matchedNodesIds[j]);
                    std::vector<int> &inpIndices = inputs[targetNodesIds[j]];

                    CV_Assert(node->getNumInputs() == inpIndices.size());
                    for (int k = 0; k < inpIndices.size(); ++k)
                    {
                        if (inpIndices[k] == fusedNodeInputs[i])
                        {
                            inpName = node->getInputName(k);
                            break;
                        }
                    }
                }
                CV_Assert(!inpName.empty());
                inputsNames[i] = inpName;
            }

            // Remove matched nodes except the last one. Indices in ascending order are expected.
            Ptr<ImportNodeWrapper> node = net->getNode(matchedNodesIds.back());
            for (int i = matchedNodesIds.size() - 2; i >= 0; --i)
                net->removeNode(matchedNodesIds[i]);

            // Modify the last node to be a fused one.
            node->setType(fusedNodeOp);
            node->setInputNames(inputsNames);

            std::vector<Ptr<ImportNodeWrapper>> inputNodes(inputsNames.size());
            for (int i = 0; i < inputsNames.size(); ++i)
            {
                inputNodes[i] = net->getNode(getInputNodeId(net, node, i));
            }
            finalize(net, node, inputNodes);
        }

        void Subgraph::finalize(const Ptr<ImportGraphWrapper> &net,
                                const Ptr<ImportNodeWrapper> &fusedNode,
                                std::vector<Ptr<ImportNodeWrapper>> &inputs) {}

        void simplifySubgraphs(const Ptr<ImportGraphWrapper> &net,
                               const std::vector<Ptr<Subgraph>> &patterns)
        {
            int numNodes = net->getNumNodes();
            std::vector<int> matchedNodesIds, targetNodesIds;
            for (int j = 0; j < patterns.size(); ++j)
            {
                for (int i = 0; i < numNodes; ++i)
                {
                    if (patterns[j]->match(net, i, matchedNodesIds, targetNodesIds))
                    {
                        patterns[j]->replace(net, matchedNodesIds, targetNodesIds);
                        numNodes -= matchedNodesIds.size() - 1; // #matchedNodes removed and one added.
                    }
                }
            }
        }

    }
} // namespace dnn, namespace cv

#endif // __OPENCV_DNN_GRAPH_SIMPLIFIER_HPP__
