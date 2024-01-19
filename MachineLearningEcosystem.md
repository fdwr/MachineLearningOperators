This may have typos, incorrect connections, and incomplete nodes.

```mermaid
%%{ init: { 'flowchart': { 'curve': 'linear' } } }%%
flowchart-elk TB
    subgraph ML_Operator_Set_Definitions
        ONNX[<a href='https://onnx.ai/onnx/operators/index.html'><abbr title='Open Neural Network Exchange'>ONNX</abbr></a>]
        ARM_TOSA[<a href='https://www.mlplatform.org/tosa/tosa_spec.html'><abbr title='Tensor Operator Set Architecture'>ARM TOSA</abbr></a>]
        OpenXLA_Stable_HLO[<a href='https://github.com/openxla/stablehlo/blob/main/docs/spec.md'><abbr title='OpenXLA Stable High Level Optimizer'>OpenXLA StableHLO</abbr></a>]
    end

    subgraph Model_Formats
        TF2_Saved_Model_File
        TF1_Hub_Format_File
        TFLite_File
        PyTorch_PTH_File
        ONNX_File
        ORT_File
        NCNN_File
    end

    subgraph Machine_Learning_Model_Libraries
        Microsoft_WinML
        Microsoft_ONNX_Runtime[<a href='https://github.com/microsoft/onnxruntime'>Microsoft <abbr title='Open Neural Network Exchange'>ONNX</abbr> Runtime</a>]
        PyTorch
        Google_TensorFlow
        Apple_CoreML[<a href='https://developer.apple.com/documentation/coreml'>Apple CoreML</a>]
        subgraph ORT_EPs
            ORT_CPU_EP[CPU]
            ORT_DirectML_EP[DML]
            ORT_CUDA_EP[CUDA]
            ORT_OpenVINO_EP[OpenVINO]
            ORT_ROCM_EP[ROCM]
            ORT_WebNN_EP[WebNN]
            ORT_WebGPU_EP[WebGPU]
            ORT_QNN_EP[QNN]
            ORT_CoreML_EP[CoreML]
        end
    end

    subgraph Machine_Learning_APIs
        Microsoft_DirectML[<a href="https://learn.microsoft.com/en-us/windows/ai/directml/dml-intro">Microsoft <abbr title='Direct Machine Learning'>DirectML</abbr></a>]
        Intel_OpenVINO[<a href='https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html'><abbr title='Intel Open Visual Inference and Neural Network Optimization'>Intel OpenVINO</abbr></a>]
        Intel_OneDNN[<a href='https://oneapi-src.github.io/oneDNN/'><abbr title='Intel One Deep Neural Network library'>Intel OneDNN</abbr></a>]
        AMD_ROCM[<abbr title='Advanced Micro Devices Radeon Open Compute Platform'>AMD ROCM</abbr>]
        Google_XNNPack[<a href='https://github.com/google/XNNPACK/blob/master/include/xnnpack.h'>Google <abbr title='X Neural Network Package'>XNNPack</abbr></a>]
        Apple_MPS[<a href='https://developer.apple.com/documentation/metalperformanceshadersgraph/mpsgraph'>Apple <abbr title='Metal Performance Shaders'>MPS</abbr></a>]
        Apple_BNNS[<a href='https://developer.apple.com/documentation/accelerate/bnns'>Apple <abbr title='Basic Neural Network Subroutines'>BNNS</abbr></a>]
        Android_NN[Google Android <abbr title='Neural Network'>NN</abbr>]
        Qualcomm_QNN
        Microsoft_CNTK[Microsoft <abbr title='Cognitive Tooklit'>CNTK</abbr>]
        Tencent_NCNN[Tencent <abbr title='Next/New/Naive/Neon Convolutional Neural Network'><a href='https://github.com/Tencent/ncnn/wiki#faq'>NCNN</a></abbr>]
        W3C_WebNN[<a href="https://www.w3.org/TR/webnn/"><abbr title='World Wide Web Web Neural Network'>W3C WebNN</abbr></a>]
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
        W3C_WebGPU
        Microsoft_Direct3D
        Khronos_OpenGL
        Khronos_Vulkan
        Khronos_SYCL
        Apple_Metal
        NVidia_CUDA
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

    ONNX --> ONNX_File
    OpenXLA_Stable_HLO --> TF2_Saved_Model_File
    NCNN_File --> Tencent_NCNN

    ONNX_File --> Microsoft_ONNX_Runtime
    ONNX_File --> Microsoft_WinML
    ORT_File --> Microsoft_ONNX_Runtime
    PyTorch_PTH_File --> PyTorch
    TFLite_File --> Google_TensorFlow
    TF1_Hub_Format_File --> Google_TensorFlow
    TF2_Saved_Model_File --> Google_TensorFlow

    Microsoft_WinML --> Microsoft_ONNX_Runtime
    Microsoft_ONNX_Runtime --> ORT_CPU_EP
    Microsoft_ONNX_Runtime --> ORT_DirectML_EP
    Microsoft_ONNX_Runtime --> ORT_CUDA_EP
    Microsoft_ONNX_Runtime --> ORT_WebNN_EP
    Microsoft_ONNX_Runtime --> ORT_WebGPU_EP
    Microsoft_ONNX_Runtime --> ORT_OpenVINO_EP
    Microsoft_ONNX_Runtime --> ORT_ROCM_EP
    Microsoft_ONNX_Runtime --> ORT_QNN_EP
    Microsoft_ONNX_Runtime --> ORT_CoreML_EP

    ORT_DirectML_EP --> Microsoft_DirectML
    ORT_CUDA_EP --> NVidia_CUDA
    ORT_OpenVINO_EP --> Intel_OpenVINO
    ORT_ROCM_EP --> AMD_ROCM
    ORT_WebNN_EP --> W3C_WebNN
    ORT_WebGPU_EP --> WGSL
    ORT_CoreML_EP --> Apple_CoreML

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
    ORT_QNN_EP --> Qualcomm_QNN

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

- TVM
- Halide
- XLA
- MLIR
- Triton MLIR
- PyTorch https://pytorch.org/docs/stable/generated/torch.sqrt.html
- TensorFlow https://www.tensorflow.org/api_docs/python/tf/math/sqrt
- ONNX Runtime https://onnxruntime.ai/
- DirectML https://learn.microsoft.com/en-us/windows/ai/directml/dml-intro https://learn.microsoft.com/en-us/windows/win32/api/directml/ns-directml-dml_element_wise_sqrt_operator_desc
- NVIDIA® CUDA® Deep Neural Network LIbrary (cuDNN) " is a GPU-accelerated library of primitives for deep neural networks. It provides highly tuned implementations of operations arising frequently - in DNN applications." https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html
- AMD ROCM
- Intel plaidML "PlaidML is a portable tensor compiler." https://www.intel.com/content/www/us/en/artificial-intelligence/plaidml.html
- OpenCL
- LLVM IR
- CUDA
- AMD Vitis ORT EP https://github.com/Xilinx/Vitis-AI, https://onnxruntime.ai/docs/execution-providers/Vitis-AI-ExecutionProvider.html
- Vitis AI DPU Deep Learning Processor Unit
- https://mlir.llvm.org/docs/Dialects/Linalg/
- OpenHLO?
- IREE team? OpenXLA initiative.
- BLAS

- High level: ONNX, PT, TF
- Low level instructions: x86, HLSL, CUDA...
-->

<!--
Resources:
https://mermaid.js.org/syntax/flowchart.html
https://mermaid.live/edit
-->
