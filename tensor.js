// From https://github.com/webmachinelearning/webnn-baseline/blob/main/src/lib/tensor.js.

'use strict';

/**
 * Compute the number of elements given a shape.
 * @param {Array} shape
 * @return {Number}
 */
function sizeOfShape(shape)
{
    return shape.reduce((accumulator, currentValue) => accumulator * currentValue, 1);
}

// Mask any dimensions of length 1 to 0, so that when coordinates are masked via getMaskedCoordinate
// that they have no contribution to the element location, allowing trivial broadcasting.
// e.g. shape [3,1,4] yields a mask [1,0,1].
function makeBroadcastingMask(shape)
{
    return shape.slice().map((value) => value > 1 ? 1 : 0);
}

// Apply the mask to the coordinate.
// e.g. coordinate [1,2,3] with mask [1,0,1] yields coordinate [1,0,3].
function getMaskedCoordinate(coordinate, mask)
{
    console.assert(coordinate.length == mask.length);
    return coordinate.slice().map((value, index) => value * mask[index]);
}

// Bidirectional broadcasting between two shapes.
// Any dimensions of size 1 in the first tensor will be broadcast to the dimension of the other tensor.
// e.g. first [1,3,1] with second [2,1,4] yields shape [2,3,4].
// If two corresponding coordinates are > 1, the first one wins (not an error). e.g. [2,1] and [3,4] yield [2,4].
function broadcastShapeWith(first, second)
{
    console.assert(first.length == second.length);
    return first.slice().map((value, index) => (value == 1) ? second[index] : value);
}

/**
 * TensorElementIterator: iterator over elements.
 */
class TensorCoordinateIterator extends Iterator
{
    index;        // Current logical element index.
    size;         // Total number of elements.
    shape;        // Dimensions. e.g. a 3D tensor [2,3,4]
    coordinate;   // Current coordinate.

    constructor(shape)
    {
        super();
        this.index = 0;
        this.size = sizeOfShape(shape);
        this.coordinate = new Array(shape.length).fill(0);
        this.shape = [...shape];
    }

    next()
    {
        if (this.index >= this.size)
        {
            // The last coordinate was already read, or this is the first call and 0 dimensions were present.
            return {done: true, value: undefined};
        }
        else if (this.index == 0)
        {
            // Unlike most language iterators, JS calls next() the first iteration too, which would yield the wrong
            // value, skipping past coordinate [0,0,0]. So treat the first call specially.
            this.index++;
            return {done: false, value: this.coordinate};
        }
        else
        {
            // Advance to next coordinate. e.g.
            // Given shape [2,3,4]:
            // Coordinate [0,1,2] -> [0,1,3]
            // Coordinate [0,1,3] -> [1,2,0]
            this.index++;
            for (let i = this.shape.length; i > 0; )
            {
                --i;
                this.coordinate[i]++;
                if (this.coordinate[i] < this.shape[i])
                    break;

                this.coordinate[i] = 0;
            }
            return {done: false, value: this.coordinate};
        }
    }
}

/**
 * Tensor: the multidimensional array.
 */
class Tensor
{
    data;
    shape;
    strides;

    /**
     * Construct a Tensor object
     * @param {Array} shape
     * @param {Array} [data]
     */
    constructor(shape, data = undefined, shouldCopy = true)
    {
        const size = sizeOfShape(shape);
        if (data !== undefined)
        {
            if (size !== data.length)
            {
                throw new Error(`The length of data ${data.length} is invalid. Expected ${size}.`);
            }

            if (shouldCopy)
            {
                this.data = data.slice();
            }
            else
            {
                this.data = data;
            }
        }
        else
        {
            this.data = new Array(size).fill(0);
        }

        // Copy the shape, and calculate the strides.
        this.shape = shape.slice();
        this.strides = new Array(this.rank);

        let stride = 1;
        for (let i = this.rank; i > 0; )
        {
            this.strides[--i] = stride;
            stride *= this.shape[i];
        }
    }

    get rank()
    {
        return this.shape.length;
    }

    get size()
    {
        return this.data.length;
    }

    /**
     * Get index in the flat array given the location.
     * @param {Array} location
     * @return {Number}
     */
    indexFromLocation(location)
    {
        if (location.length !== this.rank)
        {
            throw new Error(`The location length ${location.length} is not equal to rank ${this.rank}.`);
        }
        let index = 0;
        for (let i = 0; i < this.rank; ++i)
        {
            if (location[i] >= this.shape[i])
            {
                throw new Error(`The location value ${location[i]} at axis ${i} is invalid.`);
            }
            index += this.strides[i] * location[i];
        }
        return index;
    }

    /**
     * Get location from the index of the flat array.
     * @param {Number} index
     * @return {Array}
     */
    locationFromIndex(index)
    {
        if (index >= this.size)
        {
            throw new Error('The index is invalid.');
        }
        const location = new Array(this.rank);
        for (let i = 0; i < location.length; ++i)
        {
            location[i] = Math.floor(index / this.strides[i]);
            index -= location[i] * this.strides[i];
        }
        return location;
    }

    /**
     * Set value given the location.
     * @param {Array} location
     * @param {Number} value
     */
    setValueByLocation(location, value)
    {
        this.data[this.indexFromLocation(location)] = value;
    }

    /**
     * Get value given the location.
     * @param {Array} location
     * @return {Number}
     */
    getValueByLocation(location)
    {
        return this.data[this.indexFromLocation(location)];
    }

    at(location)
    {
        return this.data[this.indexFromLocation(location)];
    }

    setAt(location, value)
    {
        this.data[this.indexFromLocation(location)] = value;
    }

    /**
     * Set value given the index.
     * @param {Number} index
     * @param {Number} value
     */
    setValueByIndex(index, value)
    {
        if (index >= this.size)
        {
            throw new Error('The index is invalid.');
        }
        this.data[index] = value;
    }

    /**
     * Get value given the index.
     * @param {Number} index
     * @return {Number}
     */
    getValueByIndex(index)
    {
        if (index >= this.size)
        {
            throw new Error('The index is invalid.');
        }
        return this.data[index];
    }

    /*TensorCoordinateIterator*/ get coordinates()
    {
        return new TensorCoordinateIterator(this.shape);
    }

    // Return new tensor view of existing data.
    /*Tensor*/ asShape(/*Array*/ shape)
    {
        // Create a new view of the data, but do not copy the data.
        return new Tensor(shape, this.data, false);
    }
}

/**
 * Scalar: a helper class to create a Tensor with a single value.
 */
class Scalar extends Tensor
{
    /**
     * Construct a Tensor with a single value.
     * @param {Number} value
     */
    constructor(value)
    {
        super([1], [value]);
    }
}
