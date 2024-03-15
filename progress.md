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
|- Nodes ------------------> arm_compute/graph/nodes/nodes.h: Include all nodes 
|        |
|        | ----------------> arm_compute/graph/nodes/TokenEmbeddingLayerNode.h
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
|           |                                         |-> NodeType::TokenEmbeddingLayer
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
|       |                                      |-> TokenEmbeddingLayerInfo
|       |                                      |-> PositionalEncodingLayerInfo
|       |                                      |-> MultiHeadAttentionLayerInfo
|       |                                      |-> ScaleDotProductionAttentionLayerInfo
|       |                                      |-> LayerNormLayerInfo
|       |                                      |-> FeedForwardLayerInfo
|       |
|       |-------------------> arm_compute/core/CoreTypes.h
|       |                                      |-> TextFormat: utf-8
|       |
|       |-------------------> arm_compute/core/TensorInfo.h
|       |                                      |-> TensorInfo: 1D tensor info for text input 
|       |
|       |-------------------> src/core/TensorInfo.cpp
|       |                              |-> TensorInfo: wrapper over init
|       |                              |-> init
|       |
|       |-------------------> arm_compute/core/utils/DataTypeUtils.h
|                                                    |-> data_type_from_format: configure tensor data type
|
|
|- Operator ----------------> src/cpu/operators/CpuTokenEmbed.h.cppp
|
|
|- Kernel ------------------> 
|
|
|- Utils -------------------> utils/GraphUtils.h
         |                          |-> TextAccessor
         |                          |-> get_input_accessor : add txt reader
         |                          |-> WordPiecePreprocessor: constructor, preprcessor, preprocessor_typed
         |------------------> utils/GraphUtils.cpp
         |                          |-> TextAccessor
         |                          |-> get_input_accessor : add txt reader
         |                          |-> WordPiecePreprocessor: constructor, preprcessor, preprocessor_typed     
         |                              TODO: Currenting read file in using unsigned char, but processing using F32.
         |                                      Datatype mismatch.
         |                              TODO: Implement maximum input length(throughout), and implement padding. 
         |                                  Currently manual add space.
         |
         |
         |------------------> utils/TextLoader.h
         |                          |-> TextDataFeeder
         |                          |-> TXTLoader
         |                          |-> TextLoaderFactory
         |                          |-> ITextLoader   
         |
         |
         |------------------> utils/CommonGraphOptions.h
         |                          |-> add text type
         |
         |------------------> utils/Utils.h
         |                          |-> TextType
         |                          |-> parse_txt_header: TO BE USE?
         |                          |-> get_text_type_from_file
         |                              Note: NPY loader reinterpert char
         |------------------> utils/Utils.cpp
                                    |-> parse_txt_header: TO BE USE?
                                    |-> get_text_type_from_file: TODO: now just return Default UTF-8



Compatability:
            1: All function only support NEON right now.
            2. Only support UTF-8 encoding (U8)

Functionality:
            1. Segment token:
            2. Token vectorize


