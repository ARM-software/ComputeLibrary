Vanilla_Transformer
|
|-  Graph Representation ---> examples/graph_vanilla_transformer.cpp
|
|
|- Graph -------------------> arm_compute/graph/frontend/Layers.cpp
|                                                        |-> PositionalEncodingLayer(Add sequence information)
|                                                        |-> MultiHeadAttentionLayer
|                                                        |-> ScaleDotProductionAttentionLayer
|                                                        |-> LayerNormLayer
|                                                        |-> FeedForwardLayer
|
|
|- Nodes -------------------> src/graph/nodes/PositionalEncodingNode.cpp
|                                            /MultiHeadAttentionNode.cpp
|                                            /ScaleDotProductionAttentionNode.cpp
|                                            /LayerNormNode.cpp
|                                            /FeedForwardNode.cpp
|
|
|- Function ----------------> src/graph/backends/NEFunctionFactory.cpp
            |                                    |-> TBA
            |
            |
            |---------------> arm_compute/runtime/NEON/functions/NEPositionalEncodingLayer.h
            |
            |---------------> src/runtime/NEON/functions/NEPositionalEncodingLayer.cpp

