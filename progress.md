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
|        |                                     |--> /LinearLayerNode.h
|        |                                     |--> /SimpleForwardLayerNode.h: TODO forward_descriptors
|        |                                     |--> /LayerNormNode.h
|        |                                     |--> /FeedForwardNode.h
|        |
|        |
|        |-----------------> src/graph/nodes/TokenEmbeddingNode.cpp
|                                      |--> /PositionalEncodingNode.cpp
|                                      |--> /MultiHeadAttentionNode.cpp
|                                      |--> /ScaleDotProductionAttentionNode.cpp
|                                      |     _input_edges.resize(1, EmptyEdgeID);
|                                      |     _outputs.resize(1, NullTensorID);
|                                      |
|                                      |--> /SimpleForwardLayerNode.cpp
|                                      |--> /LinearLayerNode.cpp
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
|                                              |         |-> TODO: validate
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
|- Operator ----------------> src/cpu/operators/CpuTokenEmbed.h.cpp
|
|
|- Kernel ------------------> Token Embedding
|         |                   |-src/cpu/kernels/CpuTokenEmbedKernel.h.cpp -> Have been replaced by CpuVectorizeKernel.h.cpp
|         |                   |                 |-> using dst::datatype for kernel selection.
|         |                   |                     #TODO: data compability
|         |                   |-src/cpu/kernels/tokenembed/generic/neon/fp32.cpp
|         |                                                             |-> Improve kernel using intrinsics
|         |
|         |-----------------> Linear Layer
|         |                   |-src/cpu/kernels/CpuLinearKernel.h.cpp
|         |                                           |-> Interface for the kernel to perform linear operation 
|         |                   
|         |
|         |-----------------> src/cpu/kernels/CpuKernelSelectionTypes.h:
|                                             |->TokenEmbedKernelDataTypeISASelectorData & Ptr: 
|                                                For selecting kernel implmentation
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


Program structure

Input
  |-->WordPiecePreprocessor: Word token to numerical representation
  |-->atoiPreprocessor: Get sentence segmentation and convert into numerical
  |
Embedding
  |-->TokenEmbeddingLayerNode --> cpu::CpuTokenEmbed --> kernels::CpuVectorizeKernel : Numerical token into pretrained vector
  |                                                  |-> kernels::CpuPositionalEncodingKernel : Compute positional encoding 
  |
  |-->SegmentEmbeddingLayerNode --> cpu::CpuSegmentEmbed --> kernels::CpuVectorizeKernel : Segemnt token into pretrained vector
  |
  |-->EltwiseLayerNode --> EltwiseOperation::Add: Sum all three token embedding, segemnt embedding and positional embedding
  |
Linear
  |--> LinearLayerNode: Input vector * value weight(pre-trained) + value bias(pre-trained)
  |--> LinearLayerNode: Input vector * key weight(pre-trained) + key bias(pre-trained)
  |--> LinearLayerNode: Input vector * query weight(pre-trained) + query bias(pre-trained)
  |
  |--> SimpleForwardLayerNode : hold all three above output tensor in one node
  |
Attention
  |
  |


Potential problem:
            1.utils/GraphUtils.cpp: Text Preprocess input/output, configure, runtime tensor shape may mismatch
              src/cpu/kernels/tokenembed/generic/neon/fp32.cpp: neon_token_embed_char_2_float32:
                  (const unsigned int window_end_x     = src->info()->tensor_shape().x();)
          
Compatability:
            1: All function only support NEON right now.
            2. Input only support UTF-8 encoding (U8) input
            3. Interpret token_embedding npy in float right now 
            4. data layout ND
            5. Linear Layer Now only have "src/core/NEON/kernels" implementation
            6. Update arm_compute/graph/nodes/NodesFwd.h
            7. Only support lower case input eg. "I, i" 1045 
            8. Add ARM_COMPUTE_LOG_GRAPH_INFO to FunctionHelpers.h

Functionality:
            1. Segment token, currently only support 1 sentence input
            2. Token vectorize
            3. Postion Embedding from pretained model dont really need src input :
                        src/cpu/kernels/CpuPositionEmbeddingKernel.cpp
            4. Pytorch positional embdding is implemented using pretrained model, but this calcualtes.
            5. Deallocate Simple forward original output tensor
            6. Potential: src/cpu/operators/CpuScaleDotProduction.cpp Run tensor pack re indexed
            7. CpuGemmMatrixMultiplyKernel Requires 4*4transpose and 1w transpose
            8. src/cpu/operators/CpuScaleDotProduction.cpp: {ACL_SRC, const_cast<const ITensor*>(scaled_output.get())} causes 
                    free(): invalid next size (normal) Aborted
            9. There is difference from pytroch bert, which has at the end of embedding
                      embeddings = self.LayerNorm(embeddings)
                      embeddings = self.dropout(embeddings)
                while out inplementation does not
            10. segemnt embedding using verctorize kernel, produeces (*,2) shape, should be (*,1)


Optimization: 
            1. window collapse
            2. Every kernel
            3. GEMM run optimized


Input                                                           char U8                          (Len_seq, ...)
  |             
utils/GraphUtils.cpp(preprocess)                            U8 -> unsigned int                  (Len_seq*, ...) *reshape
  |
Token Embedding                                            unsigned int -> FP32                 (Len_seq*, d_model, ...)
  |
Query,Key,Value                                               FP32 -> FP32                      (Len_seq*, d_model, ...)
  |
Scale Dot Production                                          FP32 -> FP32


output_descriptor shape:  13 768 1 1 1 1
DataLayout: NCHW
DataType::F32

Tensor Shape

Input                               (Len_seq, ...)
Vocabulary                          (d_vocab, d_model, ...)
Query,Key,Value Weight              (d_model, d_model, ...)
Query,Key,Value Bias                (1, d_model, ...)


Pytroch modificationL:
Embedding:
embeddings = self.LayerNorm(embeddings)
embeddings = self.dropout(embeddings)

Self Attention:
attention_probs = self.dropout(attention_probs)

BertSelfOutput
hidden_states = self.dense(hidden_states)
hidden_states = self.dropout(hidden_states)

Layer Norm class LayerNorm(Module):
self.elementwise_affine = elementwise_affine -> False

Intermedia 

Output
hidden_states = self.dropout(hidden_states)
hidden_states = self.LayerNorm(hidden_states + input_tensor)






Current:

Linear layer