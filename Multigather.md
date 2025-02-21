---
title: Gather Multiaxis Operator
author: Dwayne Robinson
published-on: 2025-02-19
date: 2025-02-19
---

🚧 ROUGH DRAFT 🚧

# Multaxis Gather Operator

> *"One operator to gather them all, to bring them together, and in the darkness bind them."*

ML libraries have a confusing mess of various gather/scatter operators, and it always takes me a few minutes to recall the brain-bending differences between every `gather*` variant out there, even just in ONNX ([Gather](https://onnx.ai/onnx/operators/onnx__Gather.html), [GatherElements](https://onnx.ai/onnx/operators/onnx__GatherElements.html), [GatherND](https://onnx.ai/onnx/operators/onnx__GatherND.html)) let alone all the other ML libraries. Many are woefully underdocumentated on behavior too (e.g. [TOSA gather](https://mlir.llvm.org/docs/Dialects/TOSA/#tosagather-mlirtosagatherop) and [StableHLO gather](https://github.com/openxla/stablehlo/blob/main/docs/spec.md)).

It always bothered me after implementing `DML_OPERATOR_GATHER`, `DML_OPERATOR_GATHER_ELEMENTS`, and `DML_OPERATOR_GATHER_ND` at the API level (for the corresponding ONNX `Gather`, `GatherElements`, and `GatherND` operators) that there wasn't a more elegant DML operator to encompass them all, because after implementing each in GPU shaders, I realized every one could actually use the *same shader* after normalizing the tensor ranks/strides before reaching the GPU. So surely there was a more general API form too hiding behind those differences, after some massaging (1) normalizing input and indices tensor ranks consistently, (2) passing `axes` explicitly (like how `reduce*` and `resample` take axes) instead of letting them be partially *inferred* from shapes, and (3) utilizing existing broadcasting definitions like those used by elementwise operators. With that, you don't need to re-remember the divergences between them nor need hacks like an extra `batch_dims` parameter.

## Equivalence Classes

Gather operators can be grouped so:

| Category                                                             | Library Names | Notes        |
|----------------------------------------------------------------------|---------------|--------------|
| Single axis element gather             | [ONNX GatherElements](https://onnx.ai/onnx/operators/onnx__GatherElements.html)<br/>[PyTorch gather](https://pytorch.org/docs/stable/generated/torch.gather.html)<br/>[PyTorch take_along_dim](https://pytorch.org/docs/stable/generated/torch.take_along_dim.html)<br/>[numpy.take_along_axis](https://numpy.org/doc/stable/reference/generated/numpy.take_along_axis.html)<br/>[CoreML gather_along_axis](https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.ops.defs.html#coremltools.converters.mil.mil.ops.defs.iOS17.scatter_gather.gather_along_axis) | All tensors have the same rank. All dimensions in input and indices have the same size except the active axis.
| Single axis element gather 1D          | [PyTorch take](https://pytorch.org/docs/stable/generated/torch.take.html) | Same as above, but tensors are flattened to 1D first.
| Single axis block gather               | [ONNX Gather](https://onnx.ai/onnx/operators/onnx__Gather.html)<br/>[numpy.take](https://numpy.org/doc/stable/reference/generated/numpy.take.html)<br/>[TensorFlow gather](https://www.tensorflow.org/api_docs/python/tf/gather)<br/>[CoreML gather](https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.ops.defs.html#coremltools.converters.mil.mil.ops.defs.iOS17.scatter_gather.gather) | Dimensions are selected at a given axis and any trailing dimensions copy entire blocks to the output (as if those dimensions in indices were broadcast to the input shape).
| Multiple axes block gather             | [ONNX GatherND](https://onnx.ai/onnx/operators/onnx__GatherND.html)<br/>[ONNX gather_nd](https://www.tensorflow.org/api_docs/python/tf/gather_nd)<br/>[CoreML gather_nd](https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.ops.defs.html#coremltools.converters.mil.mil.ops.defs.iOS17.scatter_gather.gather_nd) | Axes are indirectly implied by correspondence of input and indices shapes and the last dimension of indices. Axes start at 0 in the input or after the batch dimension count if provided.
| Indeterminate from documentation 🤷‍♂️    | [TOSA linalg gather](https://mlir.llvm.org/docs/Dialects/TOSA/#tosagather-mlirtosagatherop)<br/>[TOSA tensor gather](https://mlir.llvm.org/docs/Dialects/TensorOps/#tensorgather-tensorgatherop)<br/>[StableHLO gather](https://github.com/openxla/stablehlo/blob/main/docs/spec.md) | TOSA's gather is probably equivalent to one of the above, but there are no useful examples. StableHLO's gather looks monstrous, like some hybrid slice/gather chimera 😯 - it's out of scope.

## API

```javascript
partial interface MLGraphBuilder
{
    MLOperand gatherMultiaxis(MLOperand input, MLOperand indices, sequence<unsigned long> axes);
};

const output = graphBuilder.gatherMultiaxis(input, indices, axes);
```

- The `input`, `indices`, and `output` tensors all have the same rank.
- The `input` and `indices` shapes may differ for dimensions in `axes`, and they must be bidirectionally broadcastable for any other dimensions.
- The output tensor shape takes logical dimensions from the `indices` tensor shape that are in `axes`, and any other dimensions are taken from the the `input` broadcasted with `indices`.

Given `axes` `[1]` and tensor shapes below, the `input` shape `[4,X,1,2]` (note axis 1 is ignorable here since it's replaced later anyway) is broadcastable with `indices` shape `[1,X,2,2]` to form `[4,X,2,2]`. Then axis 1 is taken from `indices` shape `[X,3,X,X]` to form the final `output` shape `[4,3,2,2]`.

| Parameter         | Value                  |
|-------------------|------------------------|
| axes              | `[1]` (1D coordinates) |
| input shape       | `[4,2,1,2]`            |
| indices shape     | `[1,3,2,2]`            |
| output shape      | `[4,3,2,2]`            |

For 1D cases (`axes.length == 1`), this can implement (`ONNX GatherElements`, `take_along_axis`, `gather_along_axis`) and (`ONNX/TF/CoreML Gather`, `numpy.take`).

For multiple axes (2D/3D/ND coordinates where `axes.length > 1`), shape computation is more complex since each coordinate consume multiple index elements. One approach would be to append *another* trailing dimension onto the indices (so an indices shape `[2,4]` with 3D coordinates would become shape `[2,4,3]`), which would throw off the rank consistency across tensors and possibly exceed the maximum tensor rank for higher dimension cases. Another approach would be to fold the coordinate size into the last dimension (so indices shape `[2,4]` with 3D coordinates would become shape `[2,4*3]`, and the last dimension must be an exact multiple of `axes.length`). Neither are perfect, but the latter is somewhat easier for the implementation when mapping coordinates between output/indices/input and avoids exceeding maximum rank limitations. The output in either case remains shape `[2,4]` following the *logical* indices size.

## Possible implementation

```javascript // not actually python, but closest match
import {Tensor} from './tensor.js'; // https://fdwr.github.io/MachineLearningOperators/tensor.js

// Implementation of a multiaxis gather, which can satisfy ONNX Gather, GatherElements, and GatherND.
function gatherMultiaxis(/*Tensor*/ input, /*Tensor*/ indices, /*Array*/ axes)
{
    const coordinateSize = Math.max(axes.length, 1);

    console.assert(input.rank == indices.rank);
    console.assert(axes.length <= input.rank); // There can't be more axes than input dimensions.
    console.assert(indices.size == 0 || input.size != 0); // Input cannot be empty if indices are given.
    console.assert(axes.every((value) => value < input.rank)); // Ensure valid axes.
    console.assert(axes.every((value, index) => axes.indexOf(value) == index)); // Ensure uniqueness.
    console.assert(indices.shape.at(-1) % coordinateSize == 0);

    // Bail out early for scalar case to simplify later logic.
    if (indices.rank == 0)
    {
        return new Tensor(indices.shape, input.data);
    }

    // Compute output shape, and create output tensor, taking the dimensions from indices
    // that are in axes and bidirectionally broadcasting input with logical indices dimensions.
    //
    // e.g. Given input shape [2,3,1], indices shape [1,1,4], and axes [1],
    //      first broadcast the input shape to intermediate shape [2,3,4],
    //      then take axes [1] from indices shape [_,1,_] to yield output shape [2,1,4].
    // e.g. Given input shape [10,9,8], indices shape [1,5,6*2], and axes = [1,2],
    //      first broadcast the input with logical indices shape to make [10,_,_],
    //      then take axes [1,2] from logical shape [_,5,6] to yield [10,5,6].
    let logicalIndicesShape = [...indices.shape];
    logicalIndicesShape[logicalIndicesShape.length - 1] /= coordinateSize;
    let outputShape = broadcastShapeWith(input.shape, logicalIndicesShape);
    axes.forEach((axis) => outputShape[axis] = logicalIndicesShape[axis]);
    let output = new Tensor(outputShape);

    // Create broadcasting masks to avoid creating large broadcasted temporaries
    // wastes memory and time.
    const inputCoordinateMask = makeBroadcastingMask(input.shape);
    const indicesCoordinateMask = makeBroadcastingMask(logicalIndicesShape);

    for (let outputCoordinate of output.coordinates)
    {
        // Determine the corresponding input coordinate to read from given the
        // current output coordinate and indices tensor coordinate. Achieve in-place
        // broadcasting of the input and/or indices by masking the output coordinate.
        let inputCoordinate = getMaskedCoordinate(outputCoordinate, inputCoordinateMask);
        let indexCoordinate = getMaskedCoordinate(outputCoordinate, indicesCoordinateMask);
        indexCoordinate[indexCoordinate.length - 1] *= coordinateSize;
        for (const axis of axes)
        {
            inputCoordinate[axis] = indices.at(indexCoordinate);
            indexCoordinate[indexCoordinate.length - 1]++;
        }
        const inputValue = input.at(inputCoordinate);
        output.setAt(outputCoordinate, inputValue);
    }

    return output;
}

// Mask any dimensions of length 1 to 0, so that when coordinates are masked via getMaskedCoordinate
// that they have no contribution to the element location, allowing trivial broadcasting.
// e.g. shape [3,1,4] yields a mask [1,0,1].
function makeBroadcastingMask(/*Array*/ shape)
{
    return shape.slice().map((value) => value > 1 ? 1 : 0);
}

// Apply the mask to the coordinate.
// e.g. coordinate [1,2,3] with mask [1,0,1] yields coordinate [1,0,3].
function getMaskedCoordinate(/*Array*/ coordinate, /*Array*/ mask)
{
    console.assert(coordinate.length == mask.length);
    return coordinate.slice().map((value, index) => value * mask[index]);
}
```

# Operator mappings

## Single-axis same rank for all tensors
- [ONNX GatherElements](https://onnx.ai/onnx/operators/onnx__GatherElements.html),
- [PyTorch gather](https://pytorch.org/docs/stable/generated/torch.gather.html),
- [PyTorch take_along_dim](https://pytorch.org/docs/stable/generated/torch.take_along_dim.html),
- [numpy.take_along_axis](https://numpy.org/doc/stable/reference/generated/numpy.take_along_axis.html),
- [CoreML gather_along_axis](https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.ops.defs.html#coremltools.converters.mil.mil.ops.defs.iOS17.scatter_gather.gather_along_axis)

### Mapping
```javascript
// GatherElements is basically directly compatible with multiaxis gather.
function gatherSingleAxisElements(input, indices, axis)
{
    return gatherMultiaxis(input, indices, axes: [axis]);
}
```

### Examples
```
input of shape [4,3]:
  [[ 0,  1,  2],
   [10, 11, 12],
   [20, 21, 22],
   [30, 31, 32]]
indices of shape [2,3]:
  [[3, 1, 1],
   [2, 0, 3]]
axis = 0 (default)
output of shape [2,3]:
  [[30, 11, 12],
   [20,  1, 32]]
```

```
input of shape [4,3]:
  [[ 0,  1,  2],
   [10, 11, 12],
   [20, 21, 22],
   [30, 31, 32]]
indices of shape [4,1]:
  [[2],
   [1],
   [0],
   [2]],
axis = 1
output of shape [4,1]:
  [[ 2],
   [11],
   [20],
   [32]]
```

```
input of shape [4,2,2]:
  [[[  0,   1],
    [ 10,  11]],
   [[100, 101],
    [110, 111]],
   [[200, 201],
    [210, 211]],
   [[300, 301],
    [310, 311]],]
indices of shape [1,2,2]:
  [[[0, 2],
    [1, 3]]],
axis = 0
output of shape [1,2,2]:
  [[[  0, 201],
    [110, 311]]]
```

## Single-axis coerced to 1D
- [PyTorch take](https://pytorch.org/docs/stable/generated/torch.take.html)

1D only, reinterpreting the input as 1D.

### Mapping
```javascript
function gatherForced1D(input, indices)
{
    const inputReshaped = reshape(input, [input.elementCount]);
    const indicesReshaped = reshape(indices, [indices.elementCount]);
    return gatherMultiaxis(inputReshaped, indicesReshaped, axes: [0]);
}
```

## Single-axis blocks
- [ONNX Gather](https://onnx.ai/onnx/operators/onnx__Gather.html)
- [numpy.take](https://numpy.org/doc/stable/reference/generated/numpy.take.html)
- [TensorFlow gather](https://www.tensorflow.org/api_docs/python/tf/gather)
- [CoreML gather](https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.ops.defs.html#coremltools.converters.mil.mil.ops.defs.iOS17.scatter_gather.gather)

### Mapping
```javascript
function gatherSingleAxisBlocks(input, indices, axis)
{
    // TODO: Reshape input and indices to be output compatible.
    //       Determine whether input or indices is bigger.
    let inputReshaped = reshape(input, ...);
    let indicesReshaped = reshape(indices, ...);
    return gatherMultiaxis(inputReshaped, indicesReshaped, axes: [axis]);
}
```

### Examples
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
indices of shape [5]:
  [2,1,0,1,2]
output of shape [4,5]:
  [[ 2,  1,  0,  1,  2],
   [12, 11, 10, 11, 12],
   [22, 21, 20, 21, 22],
   [32, 31, 30, 31, 32]]

intermediate processing values:
input shape   = [4,3]
indices shape = [1,5]
output shape  = [4,5]
axes          = [1]
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
input shape   = [4,1,3]
indices shape = [1,2,2]
output shape  = [4,2,2]
axes          = [2]
```

```
input:
  [[1 2]
   [3 4]]
indices:
  1
axis = 0
output:
  [3 4]

intermediate processing values:
input shape   = [2,2]
indices shape = [2,1]
output shape  = [2,2]
axes          = [0]
```

```
input:
  [[1 2]
   [3 4]]
indices:
  [1 0]
axis = 0
output:
  [[3 4]
   [1 2]]

intermediate processing values:
    input shape   = [2,2]
    indices shape = [2,1]
    output shape  = [2,2]
    axes          = [0]
```

```
input:
  [[1 2]
   [3 4]]
indices:
  [[1 0]
   [0 1]]
axis = 0
output:
  [[[3 4]
    [1 2]]
   [[1 2]
    [3 4]]]

intermediate processing values:
input shape   = [2,1,2]
indices shape = [2,2,1]
output shape  = [2,2,2]
axes          = [0]
```

## Multiple axes blocks
- [ONNX GatherND](https://onnx.ai/onnx/operators/onnx__GatherND.html)
- [ONNX gather_nd](https://www.tensorflow.org/api_docs/python/tf/gather_nd)
- [CoreML gather_nd](https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.ops.defs.html#coremltools.converters.mil.mil.ops.defs.iOS17.scatter_gather.gather_nd)
- PyTorch gather_nd missing, see [here](https://discuss.pytorch.org/t/implement-tf-gather-nd-in-pytorch/37502) and [here](https://discuss.pytorch.org/t/how-to-do-the-tf-gather-nd-in-pytorch/6445/13)

### Mapping
```javascript
function gatherNdBlocks(input, indices, batchDimensionCount)
{
    const coordinateSize = indices.shape.at(-1);
    const axes = iota(batchDimensionCount, batchDimensionCount + coordinateSize); // So 3D would yield axes [0,1,2].
    // TODO: Reshape input and indices to be output compatible.
    //       Determine whether input or indices is bigger.
    const inputReshaped = reshape(input, ...);
    const indicesReshaped = reshape(indices, ...);
    return gatherMultiaxis(inputReshaped, indicesReshaped, axes);
}
```

### Examples
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

```
input of shape [2,2,2]:
  [[[0,1],[2,3]],[[4,5],[6,7]]]
indices of shape [5,3]:
  [[0,0,1],
   [0,1,0],
   [1,0,0],
   [1,1,0],
   [1,1,1]]
output of shape [5]:
  [1,2,4,6,7]

intermediate processing values:
input shape   = [2,2,2]
indices shape = [5,1,1*3]
output shape  = [5,1,1]
axes          = [0,1,2]
```

## Unknown behavior
- [TOSA linalg gather](https://mlir.llvm.org/docs/Dialects/TOSA/#tosagather-mlirtosagatherop)
- [TOSA tensor gather](https://mlir.llvm.org/docs/Dialects/TensorOps/#tensorgather-tensorgatherop)
- [StableHLO gather](https://github.com/openxla/stablehlo/blob/main/docs/spec.md)

The documentation does not enlighten. 🤷‍♂️

# Reference

Test code:

```python
# numpy==1.24.3
import numpy as np

def gather(data, indices, axis=0, mode='gather'):
    print("gather", mode)
    print("data:\n", data)
    print("indices:\n", indices)
    print("output:")
    if mode == 'gather':
        return np.take(data, indices, axis=axis)
    elif mode == 'gather_elements':
        result = np.zeros_like(indices)
        for idx, val in np.ndenumerate(indices):
            result[idx] = data[tuple(idx[:axis] + (val,) + idx[axis+1:])]
        return result
    elif mode == 'gather_nd':
        return data[tuple(indices.T)]
    else:
        raise ValueError("Unsupported mode. Use 'gather', 'gather_elements', or 'gather_nd'.")
    #endif
#enddef

# Example usage:
data = np.array([[1, 2], [3, 4]])
indices = np.array(1)
print(gather(data, indices, axis=0, mode='gather'), '\n')  # Output: [3 4]

data = np.array([[1, 2], [3, 4]])
indices = np.array([1, 0])
print(gather(data, indices, axis=0, mode='gather'), '\n')  # Output: [[[3 4], [1 2]], [[1 2], [3 4]]]

data = np.array([[1, 2], [3, 4]])
indices = np.array([[1, 0], [0, 1]])
print(gather(data, indices, axis=0, mode='gather'), '\n')

data = np.array([[ 0,  1,  2], [10, 11, 12], [20, 21, 22], [30, 31, 32]])
indices = np.array([2,1,0,1,2])
print(gather(data, indices, axis=1, mode='gather'), '\n')

indices = np.array([[0, 1], [1, 0]])
print(gather(data, indices, axis=1, mode='gather_elements'), '\n')  # Output: [[1, 2], [4, 3]]

indices = np.array([[0, 0], [1, 1]])
print(gather(data, indices, mode='gather_nd'), '\n')  # Output: [1, 4]
```
