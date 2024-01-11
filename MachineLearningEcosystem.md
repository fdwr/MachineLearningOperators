```mermaid
flowchart TB
    subgraph ML_Operator_Set_Definitions
        ONNX[<a href='https://onnx.ai/onnx/operators/index.html'><abbr title='Open Neural Network Exchange'>ONNX</abbr></a>]
        ARM_TOSA[<a href='https://www.mlplatform.org/tosa/tosa_spec.html'><abbr title='Tensor Operator Set Architecture'>ARM TOSA</abbr></a>]
        OpenXLA_Stable_HLO[<a href='https://github.com/openxla/stablehlo/blob/main/docs/spec.md'><abbr title='OpenXLA Stable High Level Optimizer'>OpenXLA StableHLO</abbr></a>]
    end

    subgraph ML_Libraries
        WinML
        Microsoft_ONNX_Runtime[<a href='https://github.com/microsoft/onnxruntime'>Microsoft <abbr title='Open Neural Network Exchange'>ONNX</abbr> Runtime</a>]
        PyTorch
        Google_TensorFlow
    end

    %% Graphics libraries
    subgraph Graphics_Libraries
        WebGPU
        Microsoft_Direct3D
        Khronos_OpenGL
        Khronos_Vulkan
        Apple_Metal
    end

    subgraph Shader_Languages
        HLSL
        WGSL
        GLSL
        MSL
        SPIRV
    end

    subgraph Machine_Learning_APIs
        Microsoft_DirectML
        Intel_OpenVINO[<a href='https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html'><abbr title='Intel Open Visual Inference and Neural Network Optimization'>Intel OpenVINO</abbr></a>]
        NVidia_CUDA
        AMD_ROCM
        Google_XNNPack[<a href='https://github.com/google/XNNPACK/blob/master/include/xnnpack.h'>Google <abbr title='X Neural Network Package'>XNNPack</abbr></a>]
        Apple_CoreML[<a href='https://developer.apple.com/documentation/coreml'>Apple CoreML</a>]
        Apple_MPS[<a href='https://developer.apple.com/documentation/metalperformanceshadersgraph/mpsgraph'>Apple <abbr title='Metal Performance Shaders'>MPS</abbr></a>]
        Apple_BNNS[<a href='https://developer.apple.com/documentation/accelerate/bnns'>Apple <abbr title='Basic Neural Network Subroutines'>BNNS</abbr></a>]
        Android_NN
        Qualcomm_QNN
        Microsoft_CNTK
        Tencent_NCNN
        W3C_WebNN
    end

    subgraph ORT_EPs
        ORT_CUDA_EP
        ORT_DirectML_EP
        ORT_OpenVINO_EP
        ORT_ROCM_EP
        ORT_CPU_EP
        ORT_WebNN_EP
        ORT_WebGPU_EP
        ORT_QNN_EP
        ORT_CoreML_EP
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

    WinML --- Microsoft_ONNX_Runtime
    Microsoft_ONNX_Runtime --- ORT_CUDA_EP
    Microsoft_ONNX_Runtime --- ORT_OpenVINO_EP
    Microsoft_ONNX_Runtime --- ORT_DirectML_EP
    Microsoft_ONNX_Runtime --- ORT_ROCM_EP
    Microsoft_ONNX_Runtime --- ORT_CPU_EP
    Microsoft_ONNX_Runtime --- ORT_WebNN_EP
    Microsoft_ONNX_Runtime --- ORT_WebGPU_EP
    Microsoft_ONNX_Runtime --- ORT_QNN_EP
    Microsoft_ONNX_Runtime --- ORT_CoreML_EP

    ORT_OpenVINO_EP --- Intel_OpenVINO
    ORT_DirectML_EP --- Microsoft_DirectML
    ORT_CUDA_EP --- NVidia_CUDA
    ORT_ROCM_EP --- AMD_ROCM
    ORT_WebNN_EP --- W3C_WebNN
    ORT_WebGPU_EP --- WGSL
    ORT_CoreML_EP --- Apple_CoreML

    Microsoft_DirectML --- HLSL
    W3C_WebNN --- Microsoft_DirectML
    W3C_WebNN --- Google_XNNPack
    WGSL --- WebGPU
    WebGPU --- Microsoft_Direct3D
    WebGPU --- Khronos_Vulkan
    WebGPU --- Apple_Metal
    HLSL --- Microsoft_Direct3D
    GLSL --- Khronos_OpenGL
    MSL --- Apple_Metal
    SPIRV --- Khronos_Vulkan
    Apple_CoreML --- Apple_BNNS
    Apple_CoreML --- Apple_MPS
    Apple_CoreML --- Apple_ANE
    Apple_MPS --- MSL
    PyTorch --- NVidia_CUDA
    Google_TensorFlow --- NVidia_CUDA
    PyTorch --- Microsoft_DirectML
    Google_TensorFlow --- Microsoft_DirectML

    Microsoft_Direct3D --- NVidia_GPU_Driver
    Microsoft_Direct3D --- AMD_GPU_Driver
    Microsoft_Direct3D --- Microsoft_WARP
    Microsoft_Direct3D --- Intel_GPU_Driver
    Microsoft_Direct3D --- Qualcomm_GPU_Driver
    Khronos_OpenGL --- NVidia_GPU_Driver
    Khronos_OpenGL --- AMD_GPU_Driver
    Khronos_OpenGL --- Intel_GPU_Driver
    Khronos_OpenGL --- Qualcomm_GPU_Driver

    NVidia_CUDA --- NVidia_GPU_Driver
    AMD_ROCM --- AMD_GPU_Driver
    Intel_OpenVINO --- Intel_GPU_Driver
    ORT_QNN_EP --- Qualcomm_QNN

    NVidia_GPU_Driver --- NVidia_GPU
    AMD_GPU_Driver --- AMD_GPU
    Qualcomm_GPU_Driver --- Qualcomm_GPU
    Intel_GPU_Driver --- Intel_GPU
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
