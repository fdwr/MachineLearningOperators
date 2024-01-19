This may have typos, incorrect connections, and incomplete nodes.

```mermaid
%%{ init: { 'flowchart': { 'curve': 'linear' } } }%%
flowchart-elk TB
    subgraph ML_Operator_Set_Definitions
        ONNX[<a href='https://onnx.ai/onnx/operators/index.html'><abbr title='Open Neural Network Exchange'>ONNX</abbr></a>]
        ARM_TOSA[<a href='https://www.mlplatform.org/tosa/tosa_spec.html'><abbr title='Advanced RISC Machines Tensor Operator Set Architecture'>ARM TOSA</abbr></a>]
        OpenXLA_Stable_HLO[<a href='https://github.com/openxla/stablehlo/blob/main/docs/spec.md'><abbr title='Open Accelerated Linear Algebra Stable High Level Optimizer'>OpenXLA StableHLO</abbr></a>]
    end

    subgraph Model_Formats
        TF2_Saved_Model_File
        TF1_Hub_Format_File
        TFLite_File
        PyTorch_PTH_File
        ONNX_File[<a href='https://onnx.ai/onnx/operators/index.html'><abbr title='Open Neural Network Exchange'>ONNX</abbr> File</a>]
        ORT_File[<a href='https://onnxruntime.ai/docs/performance/model-optimizations/ort-format-models.html'>Microsoft <abbr title='Open Neural Network Exchange Runtime'>ORT</abbr> File</a>]
        NCNN_Param_File[<a href='https://github.com/Tencent/ncnn/wiki/param-and-model-file-structure'>Tencent <abbr title='Next/New/Naive/Neon Convolutional Neural Network'>NCNN</abbr> Param File</a>]
    end

    subgraph Machine_Learning_Libraries_High_Level
        Microsoft_WinML[<a href='https://learn.microsoft.com/en-us/windows/ai/windows-ml/'>Microsoft <abbr title='Windows Machine Learning'>WinML</abbr></a>]
        Microsoft_ONNX_Runtime[<a href='https://github.com/microsoft/onnxruntime'>Microsoft <abbr title='Open Neural Network Exchange'>ONNX</abbr> Runtime</a>]
        PyTorch
        Google_TensorFlow
        Apple_CoreML[<a href='https://developer.apple.com/documentation/coreml'>Microsoft <abbr title='Apple Core Machine Learning'>Apple CoreML</abbr></a>]
        subgraph ORT_EP_subgraph[ ]
            ORT_EPs[EPs: CPU DML CUDA OpenVINO ROCM WebNN WebGPU QNN CoreML]
        end
        ONEIROS_Keras
        Facebook_Caffe2
        MILA_Theano[<a href='https://github.com/Theano/Theano'><abbr title='Montreal Institute for Learning Algorithms Theano'>MILA Theano</abbr></a>]
        PyMC_PyTensor
    end

    subgraph Machine_Learning_Libraries_Low_Level
        Microsoft_DirectML[<a href='https://learn.microsoft.com/en-us/windows/ai/directml/dml-intro'>Microsoft <abbr title='Direct Machine Learning'>DirectML</abbr></a>]
        Intel_OpenVINO[<a href='https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html'>Intel <abbr title='Open Visual Inference and Neural Network Optimization'>OpenVINO</abbr></a>]
        Intel_OneDNN[<a href='https://oneapi-src.github.io/oneDNN/'>Intel <abbr title='One Deep Neural Network library'>OneDNN</abbr></a>]
        AMD_ROCM[<abbr title='Advanced Micro Devices Radeon Open Compute Platform'>AMD ROCM</abbr>]
        Google_XNNPack[<a href='https://github.com/google/XNNPACK/blob/master/include/xnnpack.h'>Google <abbr title='X Neural Network Package'>XNNPack</abbr></a>]
        Apple_MPS[<a href='https://developer.apple.com/documentation/metalperformanceshadersgraph/mpsgraph'>Apple <abbr title='Metal Performance Shaders'>MPS</abbr></a>]
        Apple_BNNS[<a href='https://developer.apple.com/documentation/accelerate/bnns'>Apple <abbr title='Basic Neural Network Subroutines'>BNNS</abbr></a>]
        Google_NNAPI[<a href='https://developer.android.com/ndk/guides/neuralnetworks'>Google <abbr title='Android Neural Network API'>NNAPI</abbr></a>]
        Qualcomm_QNN
        NVidia_cuDNN[NVidia <abbr title='CUDA Deep Neural Network'>cuDNN</abbr>]
        Microsoft_CNTK[Microsoft <abbr title='Cognitive Tooklit'>CNTK</abbr>]
        Tencent_NCNN[<a href='https://github.com/Tencent/ncnn/wiki#faq'>Tencent <abbr title='Next/New/Naive/Neon Convolutional Neural Network'>NCNN</abbr></a>]
        W3C_WebNN[<a href='https://www.w3.org/TR/webnn/'><abbr title='World Wide Web, Web Neural Network'>W3C WebNN</abbr></a>]
        NVidia_TensorRT[NVidia <abbr title='Tensor Runtime'>TensorRT</abbr>]
    end

    subgraph Compute_Shading_Languages
        HLSL[<abbr title='High Level Shading Language'>HLSL</abbr>]
        WGSL[<abbr title='WebGPU Shading Language'>WGSL</abbr>]
        GLSL[<abbr title='Open Graphics Library Shading Language'>GLSL</abbr>]
        MSL[<abbr title='Metal Shading Language'>MSL</abbr>]
        SPIRV[<abbr title='Standard Portable Intermediate Representation Vulkan'>SPIRV</abbr>]
        CUDA_CPP[<abbr title='Compute Unified Device Architecture'>CUDA</abbr>]
        SYCL_CPP[<abbr title='SYstem-wide Compute Language'>SYCL</abbr>]
    end

    subgraph Graphics_Compute_Libraries
        W3C_WebGPU[<abbr title='World Wide Web, Web Graphics Processing Unit'>W3C WebGPU</abbr>]
        Microsoft_Direct3D
        Khronos_OpenGL[Khronos <abbr title='Open Graphics Library'>OpenGL</abbr>]
        Khronos_Vulkan
        Khronos_SYCL[Khronos <abbr title='SYstem-wide Compute Language'>SYCL</abbr>]
        Khronos_OpenCL[Khronos <abbr title='Open Computing Language'>OpenCL</abbr>]
        Apple_Metal
        NVidia_CUDA[NVidia <abbr title='Compute Unified Device Architecture'>CUDA</abbr>]
    end

    subgraph Graphics_Drivers
        Microsoft_WARP
        NVidia_GPU_Driver
        AMD_GPU_Driver
        Qualcomm_GPU_Driver
        Intel_GPU_Driver
    end

    subgraph Hardware
        Apple_ANE[<a href='https://machinelearning.apple.com/research/neural-engine-transformers'>Apple <abbr title='Apple Neural Engine'>ANE</abbr></a>]
        NVidia_GPU
        AMD_GPU
        Qualcomm_GPU
        Intel_GPU
    end

    subgraph Unknowns
        Apache_TVM[<a href='https://tvm.apache.org/docs/'>Apache <abbr title='Tensor Virtual Machine'>TVM</abbr></a>]
        Halide[<a href='https://halide-lang.org/'>Halide</a>]
        OpenXLA_XLA[<a href='https://github.com/openxla/xla'><abbr title='Open Accelerated Linear Algebra'>OpenXLA XLA</abbr></a>]
        LLVM_MLIR[<a href='https://mlir.llvm.org/'><abbr title='Low Level Virtual Machine Multi-level Intermediate Representation'>LLVM MLIR</abbr></a>]
        NVidia_Triton_MLIR
        Google_JAX[<a href='https://jax.readthedocs.io/en/latest/'>Google <abbr title='Just After Execution'>JAX</abbr></a>]
    end

    ONNX --> ONNX_File
    OpenXLA_Stable_HLO --> TF2_Saved_Model_File
    NCNN_Param_File --> Tencent_NCNN

    ONNX_File --> Microsoft_ONNX_Runtime
    ONNX_File --> Microsoft_WinML
    ORT_File --> Microsoft_ONNX_Runtime
    PyTorch_PTH_File --> PyTorch
    TFLite_File --> Google_TensorFlow
    TF1_Hub_Format_File --> Google_TensorFlow
    TF2_Saved_Model_File --> Google_TensorFlow

    Microsoft_WinML --> Microsoft_ONNX_Runtime
    Microsoft_ONNX_Runtime --> ORT_EPs

    ORT_EPs --> Microsoft_DirectML
    ORT_EPs --> NVidia_CUDA
    ORT_EPs --> Intel_OpenVINO
    ORT_EPs --> AMD_ROCM
    ORT_EPs --> W3C_WebNN
    ORT_EPs --> WGSL
    ORT_EPs --> Apple_CoreML
    ORT_EPs --> Qualcomm_QNN

    Microsoft_DirectML --> HLSL
    W3C_WebNN --> Microsoft_DirectML
    W3C_WebNN --> Google_XNNPack
    W3C_WebGPU --> Microsoft_Direct3D
    W3C_WebGPU --> Khronos_Vulkan
    W3C_WebGPU --> Apple_Metal
    WGSL --> W3C_WebGPU
    HLSL --> Microsoft_Direct3D
    GLSL --> Khronos_OpenGL
    MSL --> Apple_Metal
    SPIRV --> Khronos_Vulkan
    SYCL_CPP --> Khronos_SYCL
    CUDA_CPP --> NVidia_CUDA
    Apple_CoreML --> Apple_BNNS
    Apple_CoreML --> Apple_MPS
    Apple_CoreML --> Apple_ANE
    Apple_MPS --> MSL
    PyTorch --> NVidia_CUDA
    Google_TensorFlow --> NVidia_CUDA
    PyTorch --> Microsoft_DirectML
    Google_TensorFlow --> Microsoft_DirectML
    NVidia_TensorRT --> NVidia_CUDA
    NVidia_cuDNN --> NVidia_CUDA
    Facebook_Caffe2 --> NVidia_CUDA

    Microsoft_Direct3D --> NVidia_GPU_Driver
    Microsoft_Direct3D --> AMD_GPU_Driver
    Microsoft_Direct3D --> Microsoft_WARP
    Microsoft_Direct3D --> Intel_GPU_Driver
    Microsoft_Direct3D --> Qualcomm_GPU_Driver
    Khronos_OpenGL --> NVidia_GPU_Driver
    Khronos_OpenGL --> AMD_GPU_Driver
    Khronos_OpenGL --> Intel_GPU_Driver
    Khronos_OpenGL --> Qualcomm_GPU_Driver

    NVidia_CUDA --> NVidia_GPU_Driver
    AMD_ROCM --> AMD_GPU_Driver
    Intel_OpenVINO --> Intel_GPU_Driver
    Intel_OneDNN --> Intel_GPU_Driver

    NVidia_GPU_Driver --> NVidia_GPU
    AMD_GPU_Driver --> AMD_GPU
    Qualcomm_GPU_Driver --> Qualcomm_GPU
    Intel_GPU_Driver --> Intel_GPU
```

<!--
TODO: Figure out how to add links to Mermaid diagram.

- Apple Model Intermediate Language Model Intermediate Language
- Apple ANE extra link https://github.com/hollance/neural-engine
- Google TPU Tensor Processing Unit https://en.wikipedia.org/wiki/Tensor_Processing_Unit
- ONNX https://github.com/onnx/onnx/blob/main/docs/Operators.md#Sqrt

TODO: Integrate all/some of these into diagram...

- PyTorch https://pytorch.org/docs/stable/generated/
- TensorFlow https://www.tensorflow.org/api_docs/python/
- ONNX Runtime https://onnxruntime.ai/
- DirectML https://learn.microsoft.com/en-us/windows/ai/directml/dml-intro https://learn.microsoft.com/en-us/windows/win32/api/directml/ns-directml-dml_element_wise_sqrt_operator_desc
- NVIDIA® CUDA® Deep Neural Network LIbrary (cuDNN) " is a GPU-accelerated library of primitives for deep neural networks. It provides highly tuned implementations of operations arising frequently - in DNN applications." https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html
- Intel plaidML "PlaidML is a portable tensor compiler." https://www.intel.com/content/www/us/en/artificial-intelligence/plaidml.html
- LLVM IR
- CUDA
- AMD Vitis ORT EP https://github.com/Xilinx/Vitis-AI, https://onnxruntime.ai/docs/execution-providers/Vitis-AI-ExecutionProvider.html
- Vitis AI DPU Deep Learning Processor Unit
- https://mlir.llvm.org/docs/Dialects/Linalg/
- OpenHLO?
- IREE team? OpenXLA initiative.
- BLAS

TODO: Add links:
    https://halide-lang.org/
    https://github.com/halide/Halide
TODO: Add links:
    ONEIROS_Keras
        JAX
        TF
        PyTorch

- High level: ONNX, PT, TF
- Low level instructions: x86, HLSL, CUDA...
-->

<!--
Resources:
https://mermaid.js.org/syntax/flowchart.html
https://mermaid.live/edit
-->
