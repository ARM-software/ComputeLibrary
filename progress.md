Vanilla_Transformer
|
|- Graph Representation ----> examples/graph_vanilla_transformer.cpp
|
|
|- Graph -------------------> arm_compute/graph/Types.h Add support layer type enum
|        |
|        |------------------> arm_compute/graph/frontend/Layers.h
|        |                                               |-> TokenEmbeddingLayer
|        |                                               |-> PositionalEncodingLayer(Add sequence information)
|        |                                                   TODO : moveable? data type
|        |                                               |-> MultiHeadAttentionLayer(Wraps over Scale Dot Production)
|        |                                               |-> ScaleDotProductionAttentionLayer
|        |                                               |-> LayerNormLayer
|        |                                               |-> FeedForwardLayer
|        |
|        |-----------------> arm_compute/graph/GraphBuilder.h
|                                              |-> add_tkemb_node
|
|
|- Nodes ------------------> arm_compute/graph/nodes/TokenEmbeddingLayerNode.h
|        |                                     |--> /PositionalEncodingNode.h
|        |                                     |--> /MultiHeadAttentionNode.h
|        |                                     |--> /ScaleDotProductionAttentionNode.h
|        |                                     |--> /LayerNormNode.h
|        |                                     |--> /FeedForwardNode.h
|        |
|        |
|        |-----------------> src/graph/nodes/TokenEmbeddingNode.cpp
|                                      |--> /PositionalEncodingNode.cpp
|                                      |--> /MultiHeadAttentionNode.cpp
|                                      |--> /ScaleDotProductionAttentionNode.cpp
|                                      |--> /LayerNormNode.cpp
|                                      |--> /FeedForwardNode.cpp
|
|
|- Function ----------------> src/graph/backends/NEON/NEFunctionFactory.cpp
|           |                                         |-> TBA
|           |
|           |
|           |---------------> arm_compute/runtime/NEON/functions/NETokenEmbeddingLayer.h
|           |                                          |------> /NEPositionalEncodingLayer.h
|           |                                          |------> /NEMultiHeadAttentionLayer.h
|           |                                          |------> /NEScaleDotProductionAttentionLayer.h
|           |                                          |------> /NELayerNormLayer.h
|           |                                          |------> /NEFeedForwardLayer.h
|           |
|           |
|           |---------------> src/runtime/NEON/functions/NETokenEmbeddingLayer.cpp
|                                              |------> /NEPositionalEncodingLayer.cpp
|                                              |------> /NEMultiHeadAttentionLayer.cpp
|                                              |------> /NEScaleDotProductionAttentionLayer.cpp
|                                              |------> /NELayerNormLayer.cpp
|                                              |------> /NEFeedForwardLayer.cpp
|
|
|- Core --------------------> arm_compute/core/Types.h
                                               |-> TokenEmbeddingLayerInfo
                                               |-> PositionalEncodingLayerInfo
                                               |-> MultiHeadAttentionLayerInfo
                                               |-> ScaleDotProductionAttentionLayerInfo
                                               |-> LayerNormLayerInfo
                                               |-> FeedForwardLayerInfo

