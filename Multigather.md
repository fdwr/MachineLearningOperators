---
title: Gather Multiaxis Operator
author: Dwayne Robinson
published-on: 2025-02-19
date: 2025-02-19
---

# 🚧 ROUGH DRAFT 🚧

# Multaxis Gather Operator

> *"One operator to gather them all, to bring them together, and in the darkness bind them."*

ML libraries have a confusing zoo of various gather/scatter operators, and it always takes me a few minutes to recall the brain-bending differences between every `gather*` variant out there, even just in ONNX ([Gather](https://onnx.ai/onnx/operators/onnx__Gather.html), [GatherElements](https://onnx.ai/onnx/operators/onnx__GatherElements.html), [GatherND](https://onnx.ai/onnx/operators/onnx__GatherND.html)) let alone all the other ML libraries. Many are woefully underdocumentated on behavior too (e.g. [TOSA gather](https://mlir.llvm.org/docs/Dialects/TOSA/#tosagather-mlirtosagatherop) and [StableHLO gather](https://github.com/openxla/stablehlo/blob/main/docs/spec.md)).

It always bothered me after implementing `DML_OPERATOR_GATHER`, `DML_OPERATOR_GATHER_ELEMENTS`, and `DML_OPERATOR_GATHER_ND` (for the corresponding ONNX `Gather`, `GatherElements`, and `GatherND` operators) that there wasn't a more elegant DML operator to encompass them all at the *API level*, because at the GPU implementation level, every operator used the *same shader* after normalizing the tensor ranks/strides to be rank-compatible and broadcastable (which made the implementation much simpler and reusable). So surely there was a more general API form too hiding behind those differences, after some massaging:
- (1) set input and indices tensor ranks consistently, padding with 1's where needed
- (2) pass `axes` explicitly (like how `reduce*` and `resample` take axes) instead of letting them be partially *inferred* from shapes
- (3) use existing broadcasting definitions like those from elementwise operators

With those normalizations, you don't need to re-remember the divergences between each of them, nor need hacks like an extra `batch_dims` parameter.

## Equivalence Classes

Gather operators can be grouped so:

| Category                                                             | Library Names | Notes        |
|----------------------------------------------------------------------|---------------|--------------|
| Single axis element gather             | [ONNX GatherElements](https://onnx.ai/onnx/operators/onnx__GatherElements.html)<br/>[PyTorch gather](https://pytorch.org/docs/stable/generated/torch.gather.html)<br/>[PyTorch take_along_dim](https://pytorch.org/docs/stable/generated/torch.take_along_dim.html)<br/>[numpy.take_along_axis](https://numpy.org/doc/stable/reference/generated/numpy.take_along_axis.html)<br/>[CoreML gather_along_axis](https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.ops.defs.html#coremltools.converters.mil.mil.ops.defs.iOS17.scatter_gather.gather_along_axis) | All tensors have the same rank. All dimensions in input and indices have the same size except the active axis.<br/><sup>`input.rank == indices.rank == output.rank`<br/>`input.shape = [leading dimensions..., input axis dimension, trailing dimensions...]`<br/>`indices.shape = [leading dimensions..., output axis dimension, trailing dimensions...]`<br/>`axis = 0..(input.rank - 1)`<br/>`output.shape = [leading dimensions..., output axis dimension, trailing dimensions...]`<br/>`output.shape[axis] = indices.shape[axis]`</sup>
| Single axis element gather 1D          | [PyTorch take](https://pytorch.org/docs/stable/generated/torch.take.html) | Same as above, but tensors are flattened to 1D first.<br/><sup>`input.shape = [input axis dimension]`<br/>`indices.shape = [output axis dimension]`<br/>`axis = always 0`<br/>`output.shape = [output axis dimension]`</sup>
| Single axis block gather               | [ONNX Gather](https://onnx.ai/onnx/operators/onnx__Gather.html)<br/>[numpy.take](https://numpy.org/doc/stable/reference/generated/numpy.take.html)<br/>[TensorFlow gather](https://www.tensorflow.org/api_docs/python/tf/gather)<br/>[CoreML gather](https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.ops.defs.html#coremltools.converters.mil.mil.ops.defs.iOS17.scatter_gather.gather) | Dimensions are selected at a given axis and any trailing dimensions copy entire blocks to the output (as if those dimensions in indices were broadcast to the input.shape).<br/><sup>`input.shape = [leading dimensions..., input axis dimension, trailing dimensions...]`<br/>`indices.shape = [index dimensions...]`<br/>`axis = 0..(input.rank - 1)`<br/>`output.shape = [leading dimensions..., index dimensions..., trailing dimensions...]`<br/>`output.shape = input.shape[0..axis] ~ indices.shape ~ input.shape[axis+1..input.rank]`</sup>
| Multiple contiguous axes block gather  | [ONNX GatherND](https://onnx.ai/onnx/operators/onnx__GatherND.html)<br/>[ONNX gather_nd](https://www.tensorflow.org/api_docs/python/tf/gather_nd)<br/>[CoreML gather_nd](https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.ops.defs.html#coremltools.converters.mil.mil.ops.defs.iOS17.scatter_gather.gather_nd) | Axes are indirectly implied by correspondence of input and indices shapes, the batch dimension count, and the size of the last dimension in indices (the lookup coordinate size). Axes start at dimension 0 in the input or after the batch dimension count if nonzero, and the number of indexable input dimensions depends on the coordinate size.<br/><sup>`input.shape = [batch dimensions..., indexable dimensions..., trailing dimensions...]`<br/>`indices.shape = [batch dimensions..., index dimensions..., coordinate size]`<br/>`block dimension count < min(input.rank, indices.rank)`<br/>`output.shape = [batch dimensions..., index dimensions..., trailing dimensions...]`</sup>
| Multiaxis gather                       | None known, emulatable via reshape + transpose + gatherND | Multiple noncontiguous axes are supported to gather from the input.<br/><sup>`input.shape = [batch dimensions..., indexable dimensions..., trailing dimensions...]`<br/>`indices.shape = [batch dimensions..., index dimensions..., coordinate size]`<br/>`block dimension count < min(input.rank, indices.rank)`<br/>`output.shape = [batch dimensions..., index dimensions..., trailing dimensions...]`</sup>
| Indeterminate from documentation 🤷‍♂️    | [TOSA linalg gather](https://mlir.llvm.org/docs/Dialects/TOSA/#tosagather-mlirtosagatherop)<br/>[TOSA tensor gather](https://mlir.llvm.org/docs/Dialects/TensorOps/#tensorgather-tensorgatherop)<br/>[StableHLO gather](https://github.com/openxla/stablehlo/blob/main/docs/spec.md) | TOSA's gather is probably equivalent to one of the above, but the docs lack insight. StableHLO's gather looks quite complex, like some hybrid slice/gather chimera 😯 - it's out of scope.

They have the following properties:

| Category                                                             | GatherElements | Gather(blocks) | GatherND(blocks) | Gather Axes |
|----------------------------------------------------------------------|----------------|----------------|------------------|-------------|
| Multiple axes                                                        | ❌             | ❌            | ✅               | ✅         |
| Non-contiguous axes (like N and C in NHWC layout)                    | ❌             | ❌            | ❌               | ✅         |
| Custom coordinate ordering (like [x,y] or [y,x])                     | ❌             | ❌            | ❌               | ✅         |
| Supports input < indices broadcasting before axes                    | ❌             | ❌            | ✅               | ✅         |
| Supports indices < input broadcasting before axes¹                   | ❌             | ✅            | ❌               | ✅         |
| Supports indices < input broadcasting after axes                     | ❌             | ✅            | ✅               | ✅         |
| Supports trailing broadcasting (after axes)                          | ❌             | ✅            | ✅               | ✅         |
| Trivial implementation²                                              | ✅             | ❌            | ❌               | ✅         |

- ¹ Unsure if it's supposed to or not, but ORT 2024-11-26 crashes with a divizion by zero when trying.
- ² Trivial implementations reduce the chances of bugs.

## Multigather Operator API

This function implements the above:

```javascript
partial interface MLGraphBuilder
{
    MLOperand gatherMultiaxis(MLOperand input, MLOperand indices, sequence<unsigned long> axes);
};

const output = graphBuilder.gatherMultiaxis(input, indices, axes);
```

- **Consistent rank**: `input.rank` == `indices.rank` == `output.rank`.
- **Broadcastability**: The `input` shape and `indices` logical shape may differ for any dimensions in `axes`, but they must be bidirectionally broadcastable for any other dimensions.
- **Logical indices shape**: The `indices` have an *actual shape* and a *logical shape*, as the last dimension of `indices` must be a multiple of the `axes` length since the coordinates are folded into that dimension. For 1D cases (`axes.length == 1`), the logical and actual shapes are identical anyway, and this is directly equivalent to {`ONNX GatherElements`, `take_along_axis`, `gather_along_axis`}. For 2D/3D/ND cases (`axes.length > 1`), the logical shape has the last dimension divided by `axes` length. e.g An `indices` logical shape `[2,4]` with 3D coordinates would have an actual shape `[2,4*3]`.
- **Output shape**: The output tensor shape takes dimensions of `indices` logical shape that are in `axes`, and any other dimensions are taken from `input` broadcasted with `indices`.
- **Axes**: Axes do not need to be contiguous and strictly in order, unlike ONNX Gather and ONNX GatherND. For example, a 2D coordinate [1,2] could be mapped to y=1 and x=2 dimensions respectively, or mapped to the (more commonly found in graphics) the x=1 and y=2 dimensions. GatherND is unable to accomodate this without either transposes on the input or reversal of the indices.

## Example

Given the following...

| Parameter         | Value                  |
|-------------------|------------------------|
| input shape       | `[4,2,1,2]`            |
| indices shape     | `[1,3,2,2]`            |
| axes              | `[1]` (1D coordinates) |
| output shape      | `[4,3,2,2]`            |

...the `input` shape `[4,_,1,2]` is broadcastable with `indices` shape `[1,_,2,2]` to form `[4,_,2,2]` (note the ignorable `_`'s in place of dimensions in `axes`, since those are replaced later anyway). Then axis 1 is taken from `indices` shape `[_,3,_,_]` to form the final `output` shape `[4,3,2,2]`.

### Alternate design considerations

For multiple axes (2D/3D/ND coordinates where `axes.length > 1`), shape computation is more complex since each coordinate consumes multiple index elements. There are a few possible approaches:
  - **indices.rank = input.rank + 1**: Append *another* trailing dimension onto the indices. So an `indices` logical shape `[2,4]` with 3D coordinates yield an actual shape `[2,4,3]`.
  - **indices.rank = input.rank and fold last dimension**: Fold the coordinate size into the last dimension. So an `indices` logical shape `[2,4]` with 3D coordinates yield an actual shape `[2,4*3]` (the last dimension is always an exact multiple of `axes.length`).

| Consideration                                                                       | input.rank + 1  | input.rank and fold last dimension |
|-------------------------------------------------------------------------------------|-----------------|--------------------------------------|
| Enables you to change the lengths of non-`axes` dimensions.                         | ✅              | ✅                                  |
| Avoids rank limitation issues.                                                      | ❌              | ✅                                  |
| Implementation correspondence between tensor coordinates is straight-forward.       | ✅              | ✅ (one extra multiply)             |

No approach is ideal, but the latter enables you to change existing dimension sizes, enables full rank usage up to backend limits (in other words, a `gatherElements` that already worked up to 5D in an implementation could be implemented by this operator), has a simple rank consistency validation rule (`input.rank == indices.rank == output.rank`), and is fairly easy for the implementation when mapping coordinates between output/indices/input.

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
- [ONNX GatherElements](https://onnx.ai/onnx/operators/onnx__GatherElements.html)
- [PyTorch gather](https://pytorch.org/docs/stable/generated/torch.gather.html)
- [PyTorch take_along_dim](https://pytorch.org/docs/stable/generated/torch.take_along_dim.html)
- [numpy.take_along_axis](https://numpy.org/doc/stable/reference/generated/numpy.take_along_axis.html)
- [CoreML gather_along_axis](https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.ops.defs.html#coremltools.converters.mil.mil.ops.defs.iOS17.scatter_gather.gather_along_axis)

### Mapping
```javascript
// GatherElements is basically directly compatible with multiaxis gather.
function gatherSingleAxisElements(input, indices, axis)
{
    return gatherMultiaxis(input, indices, [axis]);
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
    return gatherMultiaxis(inputReshaped, indicesReshaped, [0]);
}
```

## Single-axis blocks
- [ONNX Gather](https://onnx.ai/onnx/operators/onnx__Gather.html)
- [numpy.take](https://numpy.org/doc/stable/reference/generated/numpy.take.html)
- [TensorFlow gather](https://www.tensorflow.org/api_docs/python/tf/gather)
- [CoreML gather](https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.ops.defs.html#coremltools.converters.mil.mil.ops.defs.iOS17.scatter_gather.gather)

### Mapping
```javascript
// Translates ONNX Gather operator into a generic multiaxis gather by reshaping tensors.
// https://onnx.ai/onnx/operators/onnx__Gather.html

// data shape | indices shape  | axis | output shape | output equation                             | reshaped input/indices/output
// ---------- | -------------- | ---- | ------------ | ------------------------------------------- | --------------------------------------
// (P, Q)     | ( ) (a scalar) | 0    | (Q)          | output[q] = data[indices, q]                | (P, Q)       (1, 1)       (1, Q)
// (P, Q, R)  | ( ) (a scalar) | 1    | (P, R)       | output[p, r] = data[p, indices, r]          | (P, Q, R)    (1, 1, 1)    (P, 1, R)
// (P, Q, R)  | (S)            | 1    | (P, S, R)    | output[p, s, r] = data[p, indices[s], r]    | (P, Q, 1, R) (1, 1, S, 1) (P, 1, S, R)
// (P, Q)     | (R, S)         | 0    | (R, S, Q)    | output[r, s, q] = data[[indices[r, s], q]   | (P, 1, 1, Q) (1, R, S, 1) (1, R, S, Q)
// (P, Q)     | (R, S)         | 1    | (P, R, S)    | output[p, r, s] = data[p, indices[r, s]]    | (P, Q, 1, 1) (1, 1, R, S) (P, 1, R, S)
//
function gatherBlocks(input, indices, axis)
{
    console.assert(input.rank > 0);
    console.assert(axis >= 0);
    console.assert(axis < input.rank);

    // Split the shapes for input and output into leading/trailing parts.
    //
    // input shape [ leading input dimensions..., axis dimension, trailing non-indexed input dimensions...]
    //             <---------------leading part---------------->  <------------ trailing part ----------->
    //                                                          |
    //                                            alignment filler inserted here
    //
    // indices shape [ index dimensions... ]
    //                |                   |
    //  leading filler inserted here      trailing filler inserted here
    //
    const leadingInputShape     = input.shape.slice(0, axis + 1);
    const trailingInputShape    = input.shape.slice(axis + 1);
    const leadingOutputShape    = input.shape.slice(0, axis);
    const trailingOutputShape   = trailingInputShape;
    const inputShapeFiller      = new Array(indices.rank).fill(1);
    const leadingIndicesFiller  = new Array(axis + 1).fill(1);
    const trailingIndicesFiller = new Array(input.rank - axis - 1).fill(1);

    // Compute normalized shapes for input and indices, then the output shape, which starts with the input shape,
    // removes the dimension of the active axis from input, and substitutes it with the indices shape.
    //
    //      output.shape = input.shape[0..axis] ~ indices.shape ~ input.shape[axis + 1..input.rank]
    //
    const newInputShape   = [...leadingInputShape,    ...inputShapeFiller, ...trailingInputShape   ];
    const newIndicesShape = [...leadingIndicesFiller, ...indices.shape,    ...trailingIndicesFiller];
    const outputShape     = [...leadingOutputShape,   ...indices.shape,    ...trailingOutputShape  ];

    const inputReshaped = input.asShape(newInputShape);
    const indicesReshaped = indices.asShape(newIndicesShape);
    const axes = [axis];
    const output = gatherMultiaxis(inputReshaped, indicesReshaped, axes);
    return output.asShape(outputShape);
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
// Maps GatherND to multiaxis gather by reshaping tensors.
// https://onnx.ai/onnx/operators/onnx__GatherND.html
//
// Example 1
//
// batch_dims = 0
// data    = [[0,1],[2,3]]   # data_shape    = [2, 2]
// indices = [[0,0],[1,1]]   # indices_shape = [2, 2]
// output  = [0,3]           # output_shape  = [2]
///
// Example 2
//
// batch_dims = 0
// data    = [[0,1],[2,3]]  # data_shape    = [2, 2]
// indices = [[1],[0]]      # indices_shape = [2, 1]
// output  = [[2,3],[0,1]]  # output_shape  = [2, 2]
//
// Example 3
//
// batch_dims = 0
// data    = [[[0,1],[2,3]],[[4,5],[6,7]]] # data_shape    = [2, 2, 2]
// indices = [[0,1],[1,0]]                 # indices_shape = [2, 2]
// output  = [[2,3],[4,5]]                 # output_shape  = [2, 2]
//
// Example 4
//
// batch_dims = 0
// data    = [[[0,1],[2,3]],[[4,5],[6,7]]] # data_shape    = [2, 2, 2]
// indices = [[[0,1]],[[1,0]]]             # indices_shape = [2, 1, 2]
// output  = [[[2,3]],[[4,5]]]             # output_shape  = [2, 1, 2]
//
// Example 5
//
// batch_dims = 1
// data    = [[[0,1],[2,3]],[[4,5],[6,7]]] # data_shape    = [2, 2, 2]
// indices = [[1],[0]]                     # indices_shape = [2, 1]
// output  = [[2,3],[4,5]]                 # output_shape  = [2, 2]
//
function gatherNdBlocks(input, indices, batchDimensionCount)
{
    console.assert(input.rank > 0);
    console.assert(indices.rank > 0);
    console.assert(batchDimensionCount >= 0);
    console.assert(batchDimensionCount <= input.rank);
    console.assert(batchDimensionCount <= indices.rank);
    console.assert(indices.shape.at(-1) >= 0); // 0 is allowed.
    console.assert(batchDimensionCount + indices.shape.at(-1) <= input.rank); // Can't touch elements that don't exist.

    const coordinateSize = indices.shape.at(-1);
    const axes = Array.from({length: coordinateSize}, (_, i) => batchDimensionCount + i); // Sequence. e.g. 0,1,2

    // Split the input shape into leading/trailing parts.
    //
    // input shape [ batch dimensions..., indexable input dimensions..., non-indexed input dimensions...]
    //             <-----------------leading part--------------------->  <------- trailing part ------->
    //
    // Split the indices shape into leading/trailing parts (the trailing part is chopped off and multiplied back later
    // as coordinateSize).
    //
    // indices shape [ batch dimensions..., index dimensions..., coordinate tuple size dimension]
    //                <--------------leading part------------->  <------- trailing part ------->
    //
    const leadingInputShape = input.shape.slice(0, batchDimensionCount + coordinateSize); // Exclude trailing non-indexed dimensions.
    const trailingInputShape = input.shape.slice(batchDimensionCount + coordinateSize); // Exclude batch and indexed dimensions.
    const leadingIndicesShape = indices.shape.slice(0, -1); // Exclude last dimension.

    // Align the shape fragments, so that the indexable input dimensions and the index dimensions are consistent.
    // Insert filler as needed to make them correspondent and broadcastable to the output.
    //
    // input shape   [ batch, indexable dimensions, <--- filler here --->, non-indexable dimensions]
    // indices shape [ batch, indices dimensions,   <---------------- filler here ---------------->]
    //
    const maxLeadingShapingLength = Math.max(leadingInputShape.length, leadingIndicesShape.length);
    const newShapeLength = maxLeadingShapingLength + trailingInputShape.length;
    const inputShapeFiller = new Array(maxLeadingShapingLength - leadingInputShape.length).fill(1);
    const indicesShapeFiller = new Array(newShapeLength - leadingIndicesShape.length).fill(1);

    // The output shape consists of any leading batch dimensions from the indices, intermediate indices dimensions
    // (excluding the last dimension which is the coordinate size), and any residual input dimensions after consuming
    // the number of coordinates from the leading side.
    //
    //      output.shape = [ batch dimensions ... indices dimensions ... residual input dimensions]

    const newInputShape   = [...leadingInputShape,   ...inputShapeFiller,   ...trailingInputShape];
    let   newIndicesShape = [...leadingIndicesShape, ...indicesShapeFiller, /* set below */      ];
    const outputShape     = [...leadingIndicesShape, /* no filler */        ...trailingInputShape];
    newIndicesShape[newIndicesShape.length - 1] *= coordinateSize;

    const inputReshaped = input.asShape(newInputShape);
    const indicesReshaped = indices.asShape(newIndicesShape);
    const output = gatherMultiaxis(inputReshaped, indicesReshaped, axes);
    return output.asShape(outputShape);
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

// Demonstrates broadcasting of leading input dimensions (since 1, not 2)
batch_dims = 1
input of shape [1,3]
  [[0,1,2]]
indices of shape [2,1]
  [[1],[2]],
output of shape [2]
  [1,2],
```

## Unknown behavior
- [TOSA linalg gather](https://mlir.llvm.org/docs/Dialects/TOSA/#tosagather-mlirtosagatherop)
- [TOSA tensor gather](https://mlir.llvm.org/docs/Dialects/TensorOps/#tensorgather-tensorgatherop) (possibly GatherND?)
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
