---
title: Gather Multiaxis Operator
author: Dwayne Robinson
published-on: 2025-02-19
date: 2025-02-19
---

# Multaxis Gather Operator

ML libraries have a confusing mess of various gather/scatter operators, and it always takes me a few minutes to recall the little differences between every `gather*` variant out there, even just in ONNX ([Gather](https://onnx.ai/onnx/operators/onnx__Gather.html), [GatherElements](https://onnx.ai/onnx/operators/onnx__GatherElements.html), [GatherND](https://onnx.ai/onnx/operators/onnx__GatherND.html)) let alone all the other ML libraries. Many are woefully underdocumentated on behavior too (e.g. [TOSA gather](https://mlir.llvm.org/docs/Dialects/TOSA/#tosagather-mlirtosagatherop) and [StableHLO gather](https://github.com/openxla/stablehlo/blob/main/docs/spec.md)). I had an epiphany that most of these are really just the same operator with annoyingly minor differences of rank and implicit axis that could be generalized by passing explicit parameters for `axes` and coordinate size (1 for 1D indices, 3 for 3D indices...). Then you don't need to remember all the little differences nor need hacks like a `batch_dims` parameter.

```javascript
dictionary MLGatherMultiaxisOptions : MLOperatorOptions {
  sequence<unsigned long> axes; // 3D coordinates in indices would hold 3 axes.
};

const result = graphBuilder.gatherMultiaxis(input, indices, options);
```

The `input`, `indices`, and `output` tensors all have the same rank. The `indices` shape must be broadcastable to the `input` shape for all dimensions that aren't active `axes`. The output tensor shape matches the `indices` tensor shape along any corresponding `axes`, and it matches the `input` shape for other axes.

So in the example below, the `indices` shape `[1,X,1,2]` (ignoring the axis 1) is broadcastable to the `input` shape `[4,X,2,2]`, and the `indices` shape `[X,3,X,X]` (using axis 1) is combined to the final `output` shape `[4,3,2,2]`.

| Parameter         | Value               |
|-------------------|---------------------|
| axes              | `[1]`               |
| coordinateSize    | 1                   |
| input shape       | shape = `[4,2,2,2]` |
| indices shape     | shape = `[1,3,1,2]` |
| output shape      | shape = `[4,3,2,2]` |

This is equivalent to combining GatherElements/gather_along_axis and broadcasting (Expand).

For multiple axes, the shape computation becomes more complicated since every coordinate uses multiple index elements. One approach would be to append another dimension (so indices with 2D coordinate would have a shape `[M,...,N,2]`) which would throw off the rank consistency across tensors and possibly exceed the maximum tensor rank. Another approach would be to fold the coordinate size into the last dimension (so indices with 2D coordinate would have a shape `[M,...,N*2]`). Neither are perfect, but the latter is slightly easier for implementation and avoids exceeding maximum rank limitations.

## Pseudocode implementation

```javascript // not actually python, but closest match
function gatherMultiaxis(input, indices, axes)
    assert(input.rank == indices.rank)
    assert(allValuesLess(axes, input.rank))
    assert(areUnique(axes))
    assert(axes.empty || indices.shape.at(-1) % axes.length == 0)

    coordinateSize = axes.length

    // Create output tensor, taking the dimensions from indices that are in axes
    // and the dimension in input that are not in axes.
    outputShape = input.shape
    logicalIndicesShape = indices.shape
    logicalIndicesShape.at(-1) /= coordinateSize
    for i in range(axes.length)
        axis = axes[i]
        outputShape[axis] = logicalIndicesShape[i]
    endfor
    outputShapeTimesCoordinateSize = outputShape
    outputShapeTimesCoordinateSize.at(-1) *= coordinateSize
    output = new tensor(input.dataType, outputShape)

    // Broadcast the indices for any non-gather dimensions like leading batches.
    // (note an efficient implementation would avoid the memory copy and just
    // use broadcasted zero strides)
    broadcastIndices = broadcast(indices, outputShapeTimesCoordinateSize)

    for each outputCoordinate in output.coordinates
        // Determine corresponding input coordinate given the current
        // coordinate and indices tensor.
        inputCoordinate = outputCoordinate
        indexCoordinate = outputCoordinate
        indexCoordinate.at(-1) *= coordinateSize
        elementIndices = broadcastIndices[indexCoordinate .. indexCoordinate + coordinateSize]
        for i in range(axes.length)
            axis = axes[i]
            inputCoordinate[axis] = elementIndices[i]
        endfor
        output[outputCoordinate] = input[inputCoordinate]
    endfor
endfunction
```

# Equivalents

## Single-axis gather
- [ONNX GatherElements](https://onnx.ai/onnx/operators/onnx__GatherElements.html),
- [PyTorch gather](https://pytorch.org/docs/stable/generated/torch.gather.html),
- [PyTorch take_along_dim](https://pytorch.org/docs/stable/generated/torch.take_along_dim.html),
- [numpy.take_along_axis](https://numpy.org/doc/stable/reference/generated/numpy.take_along_axis.html),
- [CoreML gather_along_axis](https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.ops.defs.html#coremltools.converters.mil.mil.ops.defs.iOS17.scatter_gather.gather_along_axis)

```javascript
// Nearly directly compatible.
function gatherSingleAxis(input, indices, axis)
    return gatherMultiaxis(input, indices, axes: [axis], coordinateSize: 1)
endfunction
```

## Single-axis 1D gather
- [PyTorch take](https://pytorch.org/docs/stable/generated/torch.take.html)

1D only, reinterpreting the input as 1D.

```javascript
// Nearly directly compatible.
function gatherForced1D(input, indices)
    inputReshaped = reshape(input, [input.elementCount])
    indicesReshaped = reshape(indices, [indices.elementCount])
    return gatherMultiaxis(inputReshaped, indicesReshaped, axes: [0], coordinateSize: 1)
endfunction
```

## Gather
- [ONNX Gather](https://onnx.ai/onnx/operators/onnx__Gather.html)
- [numpy.take](https://numpy.org/doc/stable/reference/generated/numpy.take.html)
- [TensorFlow gather](https://www.tensorflow.org/api_docs/python/tf/gather)
- [CoreML gather](https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.ops.defs.html#coremltools.converters.mil.mil.ops.defs.iOS17.scatter_gather.gather)

```javascript
// Nearly directly compatible.
function gatherBlocks(input, indices, axis)
    // TODO: Reshape input and indices to be output compatible.
    //       Determine whether input or indices is bigger.
    inputReshaped = reshape(input, ...)
    indicesReshaped = reshape(indices, ...)
    return gatherMultiaxis(inputReshaped, indicesReshaped, axes: [axis], coordinateSize: 1)
endfunction
```

```
input of shape [4,3]:
  [[ 0,  1,  2],
   [10, 11, 12],
   [20, 21, 22],
   [30, 31, 32]]
axis = 0 (default)
indices of shape [2]:
  [3,1]
output of shape [2,3]:
  [[30, 31, 32],
   [10, 11, 12]]

intermediate processing values:
input shape   = [4,3]
indices shape = [2,1]
output shape  = [2,3]
axes          = [0]
```

```
input of shape [4,3]:
  [[ 0,  1,  2],
   [10, 11, 12],
   [20, 21, 22],
   [30, 31, 32]]
axis = 1
indices of shape [3]:
  [2,1,1]
output of shape [4,3]:
  [[ 2,  1,  1],
   [12, 11, 11],
   [22, 21, 21],
   [32, 31, 31]]

TODO: Verify correctness of this example

intermediate processing values:
input shape   = [4,3]
indices shape = [3]
output shape  = [4,3]
axes          = [...]
```

```
input of shape [4,3]:
  [[ 0,  1,  2],
   [10, 11, 12],
   [20, 21, 22],
   [30, 31, 32]]
axis = 1
indices of shape [2,2]:
  [[0, 1],
   [1, 2]]
output of shape [4,2,2]:
  [[[ 0,  1], [ 1,  2]],
   [[10, 11], [11, 12]],
   [[20, 21], [21, 22]],
   [[30, 31], [31, 32]]]

intermediate processing values:
input shape   = [4,3]
indices shape = [2,2]
output shape  = [4,2,2]
axes          = [...]
```

## ND gather
- [ONNX GatherND](https://onnx.ai/onnx/operators/onnx__GatherND.html)
- [ONNX gather_nd](https://www.tensorflow.org/api_docs/python/tf/gather_nd)
- [CoreML gather_nd](https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.ops.defs.html#coremltools.converters.mil.mil.ops.defs.iOS17.scatter_gather.gather_nd)
- PyTorch gather_nd missing, see [here](https://discuss.pytorch.org/t/implement-tf-gather-nd-in-pytorch/37502) and [here](https://discuss.pytorch.org/t/how-to-do-the-tf-gather-nd-in-pytorch/6445/13)

```javascript
function gatherND(input, indices, batchDimensions)
    coordinateSize = indices.shape.at(-1)
    axes = iota(batchDimensions, batchDimensions + coordinateSize) // So 3D would yield axes [0,1,2].
    // TODO: Reshape input and indices to be output compatible.
    //       Determine whether input or indices is bigger.
    inputReshaped = reshape(input, ...)
    indicesReshaped = reshape(indices, ...)
    return gatherMultiaxis(inputReshaped, indicesReshaped, axes, coordinateSize)
endfunction
```

```
input of shape [2,2,2]:
  [[[0, 1],
    [2, 3]],
   [[4, 5],
    [6, 7]]]
indices of shape [2,2]:
  [[0, 1],
   [1, 0]]
output of shape [2,2]:
  [[2, 3],   <= row [2, 3] from input coordinates [0, 1, *]
   [4, 5]]   <= row [4, 5] from input coordinates [1, 0, *]

intermediate processing values:
input shape   = [2,2,2]
indices shape = [1,2,1*2]
output shape  = [1,2,2]
axes          = [0,1]
```

```
input of shape [2,2,2]:
  [[[0, 1],
    [2, 3]],
   [[4, 5],
    [6, 7]]]
indices of shape [3,1]:
  [[1],
   [0],
   [1]]
output of shape [3,2,2]:
  [[[4, 5],   <= block [[4, 5], [6, 7]] from input coordinates [1, *, *]
    [6, 7]],
   [[0, 1],   <= block [[0, 1], [2, 3]] from input coordinates [0, *, *]
    [2, 3]],
   [[4, 5],   <= block [[4, 5], [6, 7]] from input coordinates [1, *, *]
    [6, 7]]]

intermediate processing values:
input shape   = [2,2,2]
indices shape = [3,1,1]
output shape  = [3,2,2]
axes          = [0]
```

```
input of shape [2,2,2]:
  [[[0,1],[2,3]],[[4,5],[6,7]]]
indices of shape [2,1]:
  [[1],[0]]
output of shape [2,2]:
  [[2,3],[4,5]]
batch_dims = 1

intermediate processing values:
input shape   = [2,2,2]
indices shape = [2,1,1]
output shape  = [2,1,2]
axes          = [1]
```

## Unknown
- [TOSA linalg gather](https://mlir.llvm.org/docs/Dialects/TOSA/#tosagather-mlirtosagatherop)
- [TOSA tensor gather](https://mlir.llvm.org/docs/Dialects/TensorOps/#tensorgather-tensorgatherop)
- [StableHLO gather](https://github.com/openxla/stablehlo/blob/main/docs/spec.md)

The documentation does not enlighten. 🤷‍♂️
