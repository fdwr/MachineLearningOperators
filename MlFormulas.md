---
Title: ML Operators
Author: Dwayne Robinson
Date: 2019-04-19
---

# Operator Equations

<table border=1 cellspacing=0 cellpadding=1 style='border-collapse:collapse; border:none; font-family:Calibri; font-size: 8em;'>
 <tr>
  <th style="width:15em">Category</td>
  <th>Name</td>
  <th>ONNX Name</td>
  <th>PyTorch Name</td>
  <th>DML Name</td>
  <th>Formula</td>
  <th>Notes</td>
 </tr>
 <tr>
  <td>Generic</td>
  <td>Identity</td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#Identity">Identity</a></td>
  <td></td>
  <td>DML_OPERATOR_ELEMENT_WISE_IDENTITY or DML_ACTIVATION_IDENTITY</td>
  <td>f(x) = x</td>
  <td></td>
 </tr>
 <tr>
  <td>Generation</td>
  <td>Constant</td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#Constant">Constant</a></td>
  <td></td>
  <td>NA just provide the tensor data</td>
  <td>f() = value</td>
  <td></td>
 </tr>
 <tr>
  <td>Generation</td>
  <td>Constant Fill Like</td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/rel-1.4.0/docs/Operators.md#ConstantOfShape">ConstantOfShape</a></td>
  <td></td>
  <td></td>
  <td></td>
  <td></td>
 </tr>
 <tr>
  <td>Generation</td>
  <td>Constant Fill</td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#ConstantFill">ConstantFill</a></td>
  <td></td>
  <td>NA experimental (can use trivially strides for broadcasting though)</td>
  <td>f() = value</td>
  <td></td>
 </tr>
 <tr>
  <td>Generation</td>
  <td>Constant Fill</td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#GivenTensorFill">GivenTensorFill</a></td>
  <td></td>
  <td>NA experimental</td>
  <td></td>
  <td></td>
 </tr>
 <tr>
  <td>Generation Random</td>
  <td>Random Normal</td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#RandomNormal">RandomNormal</a></td>
  <td>---</td>
  <td></td>
  <td></td>
  <td></td>
 </tr>
 <tr>
  <td>Generation Random</td>
  <td>Random Normal Like</td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#RandomNormalLike">RandomNormalLike</a></td>
  <td>---</td>
  <td></td>
  <td></td>
  <td></td>
 </tr>
 <tr>
  <td>Generation Random</td>
  <td></td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#RandomUniform">RandomUniform</a></td>
  <td>---</td>
  <td></td>
  <td></td>
  <td></td>
 </tr>
 <tr>
  <td>Generation Random</td>
  <td>Random Uniform Normal</td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#RandomUniformLike">RandomUniformLike</a></td>
  <td>---</td>
  <td></td>
  <td></td>
  <td></td>
 </tr>
 <tr>
  <td>Generation Random</td>
  <td>Random Multinomial</td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#Multinomial">Multinomial</a></td>
  <td>---</td>
  <td></td>
  <td></td>
  <td></td>
 </tr>
 <tr>
  <td>Generation Matrix Multiplication</td>
  <td>Diagonal Matrix</td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/rel-1.4.0/docs/Operators.md#EyeLike">EyeLike</a></td>
  <td></td>
  <td></td>
  <td>output[i, j] = if i + k == j then 1 else 0</td>
  <td>Diagonal matrix initializer. TODO: verify equation</td>
 </tr>
 <tr>
  <td>Elementwise Math</td>
  <td>Add</td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#Add">Add</a></td>
  <td></td>
  <td>DML_OPERATOR_ELEMENT_WISE_ADD</td>
  <td>f(x, y) = x + y</td>
  <td></td>
 </tr>
 <tr>
  <td>Elementwise Math</td>
  <td>Subtract</td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#Sub">Sub</a></td>
  <td></td>
  <td>DML_OPERATOR_ELEMENT_WISE_SUBTRACT</td>
  <td>f(x, y) = x - y</td>
  <td></td>
 </tr>
 <tr>
  <td>Elementwise Math</td>
  <td>Multiply</td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#Mul">Mul</a></td>
  <td></td>
  <td>DML_OPERATOR_ELEMENT_WISE_MULTIPLY</td>
  <td>f(x, y) = x * y</td>
  <td></td>
 </tr>
 <tr>
  <td>Elementwise Math</td>
  <td>Divide</td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#Div">Div</a></td>
  <td></td>
  <td>DML_OPERATOR_ELEMENT_WISE_DIVIDE</td>
  <td>f(x, y) = x / y</td>
  <td></td>
 </tr>
 <tr>
  <td>Elementwise Math</td>
  <td>Square root</td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#Sqrt">Sqrt</a></td>
  <td></td>
  <td>DML_OPERATOR_ELEMENT_WISE_SQRT</td>
  <td>f(x) = sqrt(x * scale + bias)</td>
  <td></td>
 </tr>
 <tr>
  <td>Elementwise Math</td>
  <td>Reciprocal</td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#Reciprocal">Reciprocal</a></td>
  <td></td>
  <td>DML_OPERATOR_ELEMENT_WISE_RECIP</td>
  <td>f(x) = 1 / (x * scale + bias)</td>
  <td></td>
 </tr>
 <tr>
  <td>Elementwise Math</td>
  <td>Power</td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#Pow">Pow</a></td>
  <td></td>
  <td>DML_OPERATOR_ELEMENT_WISE_POW</td>
  <td>f(x, exponent) = pow(x * scale + bias,
  exponent)</td>
  <td></td>
 </tr>
 <tr>
  <td>Elementwise Math</td>
  <td>Expnonent</td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#Exp">Exp</a></td>
  <td></td>
  <td>DML_OPERATOR_ELEMENT_WISE_EXP</td>
  <td>f(x) = exp(x * scale + bias)</td>
  <td></td>
 </tr>
 <tr>
  <td>Elementwise Math</td>
  <td>Logarithm</td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#Log">Log</a></td>
  <td></td>
  <td>DML_OPERATOR_ELEMENT_WISE_LOG</td>
  <td>f(x) = log(x * scale + bias)</td>
  <td></td>
 </tr>
 <tr>
  <td>Elementwise Math</td>
  <td>Absolute</td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#Abs">Abs</a></td>
  <td></td>
  <td>DML_OPERATOR_ELEMENT_WISE_ABS</td>
  <td>f(x) = abs(x * scale + bias)</td>
  <td></td>
 </tr>
 <tr>
  <td>Elementwise Math</td>
  <td>Negative</td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#Neg">Neg</a></td>
  <td></td>
  <td>DML_OPERATOR_ELEMENT_WISE_IDENTITY
  with scale = -1</td>
  <td>f(x) = -x</td>
  <td></td>
 </tr>
 <tr>
  <td>Elementwise Math</td>
  <td>Ceiling</td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#Ceil">Ceil</a></td>
  <td></td>
  <td>DML_OPERATOR_ELEMENT_WISE_CEIL</td>
  <td>f(x) = ceil(x * scale + bias)</td>
  <td></td>
 </tr>
 <tr>
  <td>Elementwise Math</td>
  <td>Floor</td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#Floor">Floor</a></td>
  <td></td>
  <td>DML_OPERATOR_ELEMENT_WISE_FLOOR</td>
  <td>f(x) = floor(x * scale + bias)</td>
  <td></td>
 </tr>
 <tr>
  <td>Elementwise Math</td>
  <td>Clamp</td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#Clip">Clip</a></td>
  <td></td>
  <td>DML_OPERATOR_ELEMENT_WISE_CLIP</td>
  <td>f(x) = clamp(x, min, max)</td>
  <td></td>
 </tr>
 <tr>
  <td>Elementwise Math</td>
  <td>Threshold</td>
  <td>NA</td>
  <td></td>
  <td>DML_OPERATOR_ELEMENT_WISE_THRESHOLD</td>
  <td>f(x) = max(minValue, x)</td>
  <td></td>
 </tr>
 <tr>
  <td>Elementwise Math</td>
  <td>Error Function</td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/rel-1.4.0/docs/Operators.md#Erf">Erf</a></td>
  <td></td>
  <td></td>
  <td>erf(x) = 1/sqrt(pi) * integrate(i = -x to x, e ^ -(i^2))

    // constants
    double a1 =  0.254829592;
    double a2 = -0.284496736;
    double a3 =  1.421413741;
    double a4 = -1.453152027;
    double a5 =  1.061405429;
    double p  =  0.3275911;

    // Save the sign of x
    int sign = 1;
    if (x < 0)
        sign = -1;
    x = fabs(x);

    // A&S formula 7.1.26
    double t = 1.0/(1.0 + p*x);
    double y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*exp(-x*x);
    return sign * y;
  </td>
  <td></td>
 </tr>
 <tr>
  <td>Elementwise Math</td>
  <td>Is Not a Number</td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/rel-1.4.0/docs/Operators.md#IsNaN">IsNan</a></td>
  <td></td>
  <td></td>
  <td>f(x) = isnan(x)</td>
  <td></td>
 </tr>
 <tr>
  <td>Elementwise Math</td>
  <td>Sign</td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/rel-1.4.0/docs/Operators.md#Sign">Sign</a></td>
  <td></td>
  <td></td>
  <td>f(x) = select x when 0 then 0, when >0 then 1, when <0 then -1</td>
  <td></td>
 </tr>
 <tr>
  <td>Elementwise Logical</td>
  <td>Greater Than</td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#Greater">Greater</a></td>
  <td></td>
  <td>DML_OPERATOR_ELEMENT_WISE_LOGICAL_GREATER_THAN</td>
  <td>f(x, y) = (x &gt; y)</td>
  <td></td>
 </tr>
 <tr>
  <td>Elementwise Logical</td>
  <td>Less Than</td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#Less">Less</a></td>
  <td></td>
  <td>DML_OPERATOR_ELEMENT_WISE_LOGICAL_LESS_THAN</td>
  <td>f(x, y) = (x &lt; y)</td>
  <td></td>
 </tr>
 <tr>
  <td>Elementwise Logical</td>
  <td>Equals</td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#Equal">Equal</a></td>
  <td></td>
  <td>DML_OPERATOR_ELEMENT_WISE_LOGICAL_EQUALS</td>
  <td>f(x, y) = (x == y)</td>
  <td></td>
 </tr>
 <tr>
  <td>Elementwise Logical</td>
  <td>Not</td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#Not">Not</a></td>
  <td></td>
  <td>DML_OPERATOR_ELEMENT_WISE_LOGICAL_NOT</td>
  <td>f(x) = !x</td>
  <td></td>
 </tr>
 <tr>
  <td>Elementwise Logical</td>
  <td>And</td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#And">And</a></td>
  <td></td>
  <td>DML_OPERATOR_ELEMENT_WISE_LOGICAL_AND</td>
  <td>f(x, y) = x &amp;&amp; y</td>
  <td></td>
 </tr>
 <tr>
  <td>Elementwise Logical</td>
  <td>Or</td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#Or">Or</a></td>
  <td></td>
  <td>DML_OPERATOR_ELEMENT_WISE_LOGICAL_OR</td>
  <td>f(x, y) = x || y</td>
  <td></td>
 </tr>
 <tr>
  <td>Elementwise Logical</td>
  <td>Xor</td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#Xor">Xor</a></td>
  <td></td>
  <td>DML_OPERATOR_ELEMENT_WISE_LOGICAL_XOR</td>
  <td>f(x, y) = x xor y</td>
  <td></td>
 </tr>
 <tr>
  <td>Elementwise Math Trigonometric</td>
  <td>Sine</td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#Sin">Sin</a></td>
  <td></td>
  <td>DML_OPERATOR_ELEMENT_WISE_SIN</td>
  <td>f(x) = sin(x)</td>
  <td></td>
 </tr>
 <tr>
  <td>Elementwise Math Trigonometric</td>
  <td>Tangent</td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#Tan">Tan</a></td>
  <td></td>
  <td>DML_OPERATOR_ELEMENT_WISE_TAN</td>
  <td>f(x) = tan(x)</td>
  <td></td>
 </tr>
 <tr>
  <td>Elementwise Math Trigonometric</td>
  <td>Arcsine</td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#Asin">Asin</a></td>
  <td></td>
  <td>DML_OPERATOR_ELEMENT_WISE_ASIN</td>
  <td>f(x) = asin(x)</td>
  <td></td>
 </tr>
 <tr>
  <td>Elementwise Math Trigonometric</td>
  <td>Arccosine</td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#Acos">Acos</a></td>
  <td></td>
  <td>DML_OPERATOR_ELEMENT_WISE_ACOS</td>
  <td>f(x) = acos(x)</td>
  <td></td>
 </tr>
 <tr>
  <td>Elementwise Math Trigonometric</td>
  <td>Cosine</td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#Cos">Cos</a></td>
  <td></td>
  <td>DML_OPERATOR_ELEMENT_WISE_COS</td>
  <td>f(x) = cos(x)</td>
  <td></td>
 </tr>
 <tr>
  <td>Elementwise Math Trigonometric</td>
  <td>Arctangent</td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#Atan">Atan</a></td>
  <td></td>
  <td>DML_OPERATOR_ELEMENT_WISE_ATAN</td>
  <td>f(x) = atan(x)</td>
  <td></td>
 </tr>
 <tr>
  <td>Elementwise Math Trigonometric</td>
  <td>Hyperbolic Arccosine</td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/rel-1.4.0/docs/Operators.md#Acosh">Acosh</a></td>
  <td></td>
  <td></td>
  <td>f(x) = arcosh(x)</td>
  <td></td>
 </tr>
 <tr>
  <td>Elementwise Math Trigonometric</td>
  <td>Hyperbolic Cosine</td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/rel-1.4.0/docs/Operators.md#Cosh">Cosh</a></td>
  <td></td>
  <td></td>
  <td>f(x) = cosh(x)</td>
  <td></td>
 </tr>
 <tr>
  <td>Elementwise Math Trigonometric</td>
  <td>Hyperbolic Sine</td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/rel-1.4.0/docs/Operators.md#Sinh">Sinh</a></td>
  <td></td>
  <td></td>
  <td>f(x) = sinh(x)</td>
  <td></td>
 </tr>
 <tr>
  <td>Elementwise Math Reduction</td>
  <td>Sum</td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#Sum">Sum</a></td>
  <td></td>
  <td>DML_OPERATOR_ELEMENT_WISE_ADD via repeated inputs</td>
  <td>f(X, Y, Z…) = X + Y + Z…</td>
  <td></td>
 </tr>
 <tr>
  <td>Elementwise Math Reduction</td>
  <td>Mean</td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#Mean">Mean</a></td>
  <td></td>
  <td>DML_OPERATOR_ELEMENT_WISE_MEAN via repeated inputs</td>
  <td>f(X, Y, Z…) = (X + Y + Z…) / n</td>
  <td></td>
 </tr>
 <tr>
  <td>Elementwise Math Reduction</td>
  <td></td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#Max">Max</a></td>
  <td></td>
  <td>DML_OPERATOR_ELEMENT_WISE_MAX via repeated inputs</td>
  <td>f(X, Y, Z…) = max(X, Y, Z…)</td>
  <td></td>
 </tr>
 <tr>
  <td>Elementwise Math Reduction</td>
  <td></td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#Min">Min</a></td>
  <td></td>
  <td>DML_OPERATOR_ELEMENT_WISE_MIN via repeated inputs</td>
  <td>f(X, Y, Z…) = min(X, Y, Z…)</td>
  <td></td>
 </tr>
 <tr>
  <td>Elementwise Math Quantization</td>
  <td>Quantize Linear</td>
  <td>com.microsoft QuantizeLinear</td>
  <td></td>
  <td>DML_OPERATOR_ELEMENT_WISE_QUANTIZE_LINEAR</td>
  <td>f(input:float32, scale:float32,
  zero_point:int32)
  output:uint8 = clamp(round(input /
  scale) + zero_point, 0, 255)</td>
  <td></td>
 </tr>
 <tr>
  <td>Elementwise Math Quantization</td>
  <td>Dequantize Linear</td>
  <td>com.microsoft DequantizeLinear</td>
  <td></td>
  <td>DML_OPERATOR_ELEMENT_WISE_DEQUANTIZE_LINEAR</td>
  <td>f(input:uint8, scale:float32,
  zero_point:float32)
  output:float32 = (input:uint8 -
  zero_point) * scale</td>
  <td></td>
 </tr>
 <tr>
  <td>Activation</td>
  <td></td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#Sigmoid">Sigmoid</a></td>
  <td></td>
  <td>DML_OPERATOR_ACTIVATION_SIGMOID</td>
  <td>f(x) = 1 / (1 + exp(-x))</td>
  <td></td>
 </tr>
 <tr>
  <td>Activation</td>
  <td></td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#HardSigmoid">HardSigmoid</a></td>
  <td></td>
  <td>DML_OPERATOR_ACTIVATION_HARD_SIGMOID</td>
  <td>f(x) = max(0, min(alpha * x + beta, 1))
  defaults: alpha = 0.2, beta = 0.5</td>
  <td></td>
 </tr>
 <tr>
  <td>Activation</td>
  <td></td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#Tanh">Tanh</a></td>
  <td></td>
  <td>DML_OPERATOR_ACTIVATION_TANH</td>
  <td>f(x) = (1 - exp(-2 * x))/(1 + exp(-2 *
  x))
  = return 2 /(1 + exp(-2 * x)) - 1</td>
  <td></td>
 </tr>
 <tr>
  <td>Activation</td>
  <td></td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#ScaledTanh">ScaledTanh</a></td>
  <td></td>
  <td>DML_OPERATOR_ACTIVATION_SCALED_TANH</td>
  <td>f(x) = alpha * tanh(beta * x)
  Decomposition: Mul(Tanh(Mul(X, beta)),
  alpha)</td>
  <td></td>
 </tr>
 <tr>
  <td>Activation</td>
  <td></td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#Relu">Relu</a></td>
  <td></td>
  <td>DML_OPERATOR_ACTIVATION_RELU</td>
  <td>f(x) = max(0, x) // or if x &gt;= 0
  then x else 0</td>
  <td></td>
 </tr>
 <tr>
  <td>Activation</td>
  <td></td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#LeakyRelu">LeakyRelu</a></td>
  <td></td>
  <td>DML_OPERATOR_ACTIVATION_LEAKY_RELU</td>
  <td>f(x) = if x &gt;= 0 then x else alpha
  * x</td>
  <td></td>
 </tr>
 <tr>
  <td>Activation</td>
  <td></td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#PRelu">PRelu</a></td>
  <td></td>
  <td>DML_OPERATOR_ACTIVATION_PARAMETERIZED_RELU</td>
  <td>f(x) = if x &gt;= 0 then x else tensor
  slope * x</td>
  <td></td>
 </tr>
 <tr>
  <td>Activation</td>
  <td></td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#ThresholdedRelu">ThresholdedRelu</a></td>
  <td></td>
  <td>DML_OPERATOR_ACTIVATION_THRESHOLDED_RELU</td>
  <td>f(x) = if x &gt; alpha then x else 0</td>
  <td></td>
 </tr>
 <tr>
  <td>Activation</td>
  <td></td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#Elu">Elu</a></td>
  <td></td>
  <td>DML_OPERATOR_ACTIVATION_ELU</td>
  <td>f(x) = if x &gt;= 0 then x else alpha * (expe(x) - 1)</td>
  <td></td>
 </tr>
 <tr>
  <td>Activation</td>
  <td></td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#Selu">Selu</a></td>
  <td></td>
  <td>DML_OPERATOR_ACTIVATION_SCALED_ELU</td>
  <td>f(x) = if x &gt; 0 then gamma * x else
  gamma * (alpha * e^x - alpha)
  Defaults: alpha = 1.6732, gamma =
  1.0507</td>
  <td></td>
 </tr>
 <tr>
  <td>Activation</td>
  <td></td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#Softmax">Softmax</a></td>
  <td></td>
  <td>DML_OPERATOR_ACTIVATION_SOFTMAX</td>
  <td>f(x_i) = expe(x_i) / sum(expe(X))
  Â = exp(x_i - max(X)) / sum(expe(x - max(X))) </td>
  <td></td>
 </tr>
 <tr>
  <td>Activation</td>
  <td></td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#LogSoftmax">LogSoftmax</a></td>
  <td></td>
  <td>DML_OPERATOR_ACTIVATION_LOG_SOFTMAX</td>
  <td>f(x_i) = loge(expe(x_i - max(X)) / sum(expe(X - max(X))))
  Â = (x_i - max(X)) - loge(x - max(X))))</td>
  <td></td>
 </tr>
 <tr>
  <td>Activation</td>
  <td></td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#Hardmax">Hardmax</a></td>
  <td></td>
  <td>DML_OPERATOR_ACTIVATION_HARDMAX</td>
  <td>f(x) = if x_i == max(X) then 1 else 0
  *but only for first element in axis</td>
  <td></td>
 </tr>
 <tr>
  <td>Activation</td>
  <td></td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#Softsign">Softsign</a></td>
  <td></td>
  <td>DML_OPERATOR_ACTIVATION_SOFTSIGN</td>
  <td>f(x) = x / (1 + abs(x))</td>
  <td></td>
 </tr>
 <tr>
  <td>Activation</td>
  <td></td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#Softplus">Softplus</a></td>
  <td></td>
  <td>DML_OPERATOR_ACTIVATION_SOFTPLUS</td>
  <td>f(x) = ln(1 + expe(x))</td>
  <td></td>
 </tr>
 <tr>
  <td>Activation</td>
  <td></td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#ParametricSoftplus">ParametricSoftplus</a></td>
  <td></td>
  <td>DML_OPERATOR_ACTIVATION_PARAMETRIC_SOFTPLUS</td>
  <td>f(x) = alpha * loge(1 + expe(beta * x))
  decomposition: Mul(alpha, Log(Add(1, Exp(Mul(beta, X)).</td>
  <td></td>
 </tr>
 <tr>
  <td>Activation</td>
  <td></td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#Affine">Affine</a></td>
  <td></td>
  <td>DML_OPERATOR_ELEMENT_WISE_IDENTITY
  with scale&amp;bias or DML_OPERATOR_ACTIVATION_LINEAR</td>
  <td>f(x) = alpha * x + beta</td>
  <td></td>
 </tr>
 <tr>
  <td>Activation</td>
  <td>Symmetric signal shift</td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/rel-1.4.0/docs/Operators.md#shrink">Shrink</a></td>
  <td></td>
  <td></td>
  <td>if x < -lambda then y = x + bias; 
if x > lambda then y = x - bias else y = 0;</td>
  <td></td>
 </tr>
 <tr>
  <td>Matrix Multiplication</td>
  <td></td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#Gemm">Gemm</a></td>
  <td></td>
  <td>DML_OPERATOR_MATRIX_GEMM</td>
  <td>Y = matmul(alpha * transpose(A)),
  transpose(B)) + beta * C</td>
  <td></td>
 </tr>
 <tr>
  <td>Matrix Multiplication</td>
  <td></td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#MatMul">MatMul</a></td>
  <td></td>
  <td>DML_OPERATOR_MATRIX_GEMM</td>
  <td>Y = matmul(A, B) // A and B can be 1D vectors, which are treated as [1,W] and [H,1] matrices</td>
  <td></td>
 </tr>
 <tr>
  <td>Matrix Multiplication</td>
  <td></td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#Conv">Conv</a></td>
  <td></td>
  <td>DML_OPERATOR_CONVOLUTION with
  DML_CONVOLUTION_MODE_CROSS_CORRELATION, DML_CONVOLUTION_DIRECTION_FORWARD</td>
  <td>out[j] = x[i]*w[0] + x[i+1]*w[1] + x[i+2]*w[2]
  + ... + x[i+k]*w[k] + b</td>
  <td></td>
 </tr>
 <tr>
  <td>Matrix Multiplication</td>
  <td></td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#ConvTranspose">ConvTranspose</a></td>
  <td></td>
  <td>DML_OPERATOR_CONVOLUTION with DML_CONVOLUTION_MODE_CROSS_CORRELATION, DML_CONVOLUTION_DIRECTION_BACKWARD</td>
  <td></td>
  <td></td>
 </tr>
 <tr>
  <td>Data Reorganization</td>
  <td></td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#Cast">Cast</a></td>
  <td></td>
  <td>DML_OPERATOR_CAST</td>
  <td>f(x) = cast(x)</td>
  <td></td>
 </tr>
 <tr>
  <td>Data Reorganization</td>
  <td></td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#Transpose">Transpose</a></td>
  <td></td>
  <td>Identity with new TENSOR_DESC that
  flips via permuted strides</td>
  <td>Reorder axes, such as X Y -&gt; Y X, or X Y Z -&gt; Z X Y.</td>
  <td></td>
 </tr>
 <tr>
  <td>Data Reorganization</td>
  <td>Broadcast</td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/rel-1.3.0/docs/Operators.md#Expand">Expand</a></td>
  <td></td>
  <td>Identity with TENSOR_DESC using zero
  strides along broadcast dimension</td>
  <td>Expand the input tensor, broadcasting any single size dimensions up to the output dimension counts</td>
  <td></td>
 </tr>
 <tr>
  <td>Data Reorganization</td>
  <td></td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#Tile">Tile</a></td>
  <td></td>
  <td>DML_OPERATOR_TILE</td>
  <td></td>
  <td></td>
 </tr>
 <tr>
  <td>Data Reorganization</td>
  <td></td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#Split">Split</a></td>
  <td></td>
  <td>DML_OPERATOR_SPLIT</td>
  <td>Split input into multiple output tensors</td>
  <td></td>
 </tr>
 <tr>
  <td>Data Reorganization</td>
  <td></td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#Slice">Slice</a></td>
  <td></td>
  <td>DML_OPERATOR_SLICE</td>
  <td>Crop the tensor to the given ranges for each axis.</td>
  <td></td>
 </tr>
 <tr>
  <td>Data Reorganization</td>
  <td></td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/rel-1.4.0/docs/Operators.md#DynamicSlice">DynamicSlice</a></td>
  <td></td>
  <td>DML_OPERATOR_SLICE</td>
  <td></td>
  <td></td>
 </tr>
 <tr>
  <td>Data Reorganization</td>
  <td></td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#Concat">Concat</a></td>
  <td></td>
  <td>DML_OPERATOR_JOIN</td>
  <td>Combine multiple tensors into large output tensor. e.g. {1,2,3} with {4,5} -&gt; {1,2,3,4,5}</td>
  <td></td>
 </tr>
 <tr>
  <td>Data Reorganization</td>
  <td></td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#Gather">Gather</a></td>
  <td></td>
  <td>DML_OPERATOR_GATHER</td>
  <td></td>
  <td></td>
 </tr>
 <tr>
  <td>Data reorganization</td>
  <td>Scatter</td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/rel-1.4.0/docs/Operators.md#Scatter">Scatter</a></td>
  <td></td>
  <td></td>
  <td></td>
 </tr>
 <tr>
  <td>Data Reorganization</td>
  <td></td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/rel-1.4.0/docs/Operators.md#experimental-crop">Crop</a> (ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#Slice">Slice</a> subset)?</td>
  <td></td>
  <td>DML_OPERATOR_SLICE</td>
  <td>Crop the tensor to the given ranges for each axis. Crop is confusing and redundant. Just use Slice. TODO:DELETED</td>
  <td></td>
 </tr>
 <tr>
  <td>Data Reorganization</td>
  <td></td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#Pad">Pad</a></td>
  <td></td>
  <td>DML_OPERATOR_PADDING</td>
  <td>Inflate the input with zeroes on the edges</td>
  <td></td>
 </tr>
 <tr>
  <td>Data Reorganization</td>
  <td></td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#SpaceToDepth">SpaceToDepth</a></td>
  <td></td>
  <td>DML_OPERATOR_SPACE_TO_DEPTH</td>
  <td></td>
  <td></td>
 </tr>
 <tr>
  <td>Data Reorganization</td>
  <td></td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#DepthToSpace">DepthToSpace</a></td>
  <td></td>
  <td>DML_OPERATOR_DEPTH_TO_SPACE</td>
  <td></td>
  <td></td>
 </tr>
 <tr>
  <td>Data Reorganization</td>
  <td></td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#Shape">Shape</a></td>
  <td></td>
  <td>NA, just read the TENSOR_DESC dimensions</td>
  <td></td>
  <td></td>
 </tr>
 <tr>
  <td>Data Reorganization</td>
  <td>Element Count</td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#Size">Size</a></td>
  <td></td>
  <td>NA, just compute the number of TENSOR_DESC elements</td>
  <td></td>
  <td></td>
 </tr>
 <tr>
  <td>Data Reorganization</td>
  <td>Reshape</td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#Reshape">Reshape</a></td>
  <td></td>
  <td>NA, no actual data change, just update the TENSOR_DESC</td>
  <td></td>
  <td></td>
 </tr>
 <tr>
  <td>Data Reorganization</td>
  <td>Reshape 2D</td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#Flatten">Flatten</a></td>
  <td></td>
  <td>NA, no actual data change, just update the TENSOR_DESC</td>
  <td></td>
  <td></td>
 </tr>
 <tr>
  <td>Data Reorganization</td>
  <td></td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#Squeeze">Squeeze</a></td>
  <td></td>
  <td>NA, just rearrange the TENSOR_DESC</td>
  <td></td>
  <td></td>
 </tr>
 <tr>
  <td>Data Reorganization</td>
  <td></td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#Unsqueeze">Unsqueeze</a></td>
  <td></td>
  <td>NA, just rearrange the TENSOR_DESC</td>
  <td></td>
  <td></td>
 </tr>
 <tr>
  <td>Data reorganization mapping</td>
  <td></td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/rel-1.4.0/docs/Operators.md#OneHot">OneHot</a></td>
  <td></td>
  <td></td>
  <td></td>
 </tr>
 <tr>
  <td>Data Reorganization</td>
  <td></td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#TopK">TopK</a></td>
  <td></td>
  <td>DML_OPERATOR_TOP_K</td>
  <td>f(X) = Slice(Sort(X, axis),
  range=0..k, axis)</td>
  <td></td>
 </tr>
 <tr>
  <td>Data reorganization selection</td>
  <td>Select elementwise</td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/rel-1.4.0/docs/Operators.md#Where">Where</a></td>
  <td></td>
  <td></td>
  <td>f(b, x, y) = if b then x else y</td>
  <td>A conditional per-element if statement.</td>
 </tr>
 <tr>
  <td>Data reorganization selection</td>
  <td>Select slices</td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/rel-1.4.0/docs/Operators.md#Compress">Compress</a></td>
  <td></td>
  <td></td>
  <td></td>
  <td>A conditional slice/join. Has utterly nothing to do with data compression, despite the confusing name.</td>
 </tr>
 <tr>
  <td>Pooling</td>
  <td></td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#GlobalAveragePool">GlobalAveragePool</a></td>
  <td></td>
  <td>DML_OPERATOR_AVERAGE_POOLING</td>
  <td>y = (x1 + x + …) / pool_size
  where X[N C H W] -&gt; Y[N C 1 1]</td>
  <td></td>
 </tr>
 <tr>
  <td>Pooling</td>
  <td></td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#AveragePool">AveragePool</a></td>
  <td></td>
  <td>DML_OPERATOR_AVERAGE_POOLING</td>
  <td>y = (x1 + x2 + …) / pool_size</td>
  <td></td>
 </tr>
 <tr>
  <td>Pooling</td>
  <td></td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#GlobalMaxPool">GlobalMaxPool</a></td>
  <td></td>
  <td>DML_OPERATOR_MAX_POOLING with output being 1 element</td>
  <td>y = max(x1 + x2 + … x_pool_size)</td>
  <td></td>
 </tr>
 <tr>
  <td>Pooling</td>
  <td></td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#MaxPool">MaxPool</a></td>
  <td></td>
  <td>DML_OPERATOR_MAX_POOLING</td>
  <td>y = max(x1 + x2 + … x_pool_size)</td>
  <td></td>
 </tr>
 <tr>
  <td>Pooling</td>
  <td></td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/rel-1.4.0/docs/Operators.md#MaxUnpool">MaxUnpool</a></td>
  <td></td>
  <td></td>
  <td></td>
 </tr>
 <tr>
  <td>Pooling</td>
  <td></td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#LpPool">LpPool</a></td>
  <td></td>
  <td>DML_OPERATOR_LP_POOLING</td>
  <td>y = (x1^p + x2^p + ... + xn^p) ^ (1/p) where X -&gt; Y reduced for each kernel</td>
  <td></td>
 </tr>
 <tr>
  <td>Pooling</td>
  <td></td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#GlobalLpPool">GlobalLpPool</a></td>
  <td></td>
  <td>DML_OPERATOR_LP_POOLING with output
  being 1 element</td>
  <td>y = (x1^p + x2^p + ... + xn^p) ^ (1/p) where X[N C H W] -&gt; Y[N C 1 1] e.g. (3^2 + 4^2) ^ (1/2) = 5</td>
  <td></td>
 </tr>
 <tr>
  <td>Pooling</td>
  <td></td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#MaxRoiPool">MaxRoiPool</a></td>
  <td></td>
  <td>DML_OPERATOR_ROI_POOLING (only POOLING_MAX is supported)</td>
  <td></td>
  <td></td>
 </tr>
 <tr>
  <td>Reduction</td>
  <td></td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#ReduceSum">ReduceSum</a></td>
  <td></td>
  <td>DML_OPERATOR_REDUCE with DML_REDUCE_FUNCTION_SUM</td>
  <td>x = (x1 + x2 + ... + xn)</td>
  <td></td>
 </tr>
 <tr>
  <td>Reduction</td>
  <td></td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#ReduceMean">ReduceMean</a></td>
  <td></td>
  <td>DML_OPERATOR_REDUCE with DML_REDUCE_FUNCTION_AVERAGE</td>
  <td>x = (x1 + x2 + ... + xn) / n</td>
  <td></td>
 </tr>
 <tr>
  <td>Reduction</td>
  <td></td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#ReduceProd">ReduceProd</a></td>
  <td></td>
  <td>DML_OPERATOR_REDUCE with DML_REDUCE_FUNCTION_MULTIPLY</td>
  <td>x = (x1 * x2 * ... * xn)</td>
  <td></td>
 </tr>
 <tr>
  <td>Reduction</td>
  <td></td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#ReduceLogSum">ReduceLogSum</a></td>
  <td></td>
  <td>DML_OPERATOR_REDUCE with DML_REDUCE_FUNCTION_LOG_SUM</td>
  <td>x = log(x1 + x2 + ... + xn)
  f(X) = Log(ReduceSum(data, axes))</td>
  <td></td>
 </tr>
 <tr>
  <td>Reduction</td>
  <td></td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#ReduceLogSumExp">ReduceLogSumExp</a></td>
  <td></td>
  <td>DML_OPERATOR_REDUCE with DML_REDUCE_FUNCTION_LOG_SUM_EXP</td>
  <td>
  f(X) = Log(ReduceSum(Exp(data), axes))
  x = log(exp(x1) + exp(x2) + ... + exp(xn))
  </td>
  <td></td>
 </tr>
 <tr>
  <td>Reduction</td>
  <td></td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#ReduceSumSquare">ReduceSumSquare</a></td>
  <td></td>
  <td>DML_OPERATOR_REDUCE with DML_REDUCE_FUNCTION_SUM_SQUARE</td>
  <td>f(X) = ReduceSum(Pow(X, 2), axes)
  x = x1^2 + x2^2 + ... + xn^2</td>
  <td></td>
 </tr>
 <tr>
  <td>Reduction</td>
  <td></td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#ReduceL1">ReduceL1</a></td>
  <td></td>
  <td>DML_OPERATOR_REDUCE with DML_REDUCE_FUNCTION_L1</td>
  <td>f(X) = ReduceSum(Abs(X), axes)
  = Abs(x1) + Abs(x2) + ... + Abs(xn)</td>
  <td></td>
 </tr>
 <tr>
  <td>Reduction</td>
  <td></td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#ReduceL2">ReduceL2</a></td>
  <td></td>
  <td>DML_OPERATOR_REDUCE with DML_REDUCE_FUNCTION_L2</td>
  <td>f(X) = ReduceSum(Sqrt(X), axes)
  x = sqrt(x1^2 + x2^2 + ... + xn^2)</td>
  <td></td>
 </tr>
 <tr>
  <td>Reduction</td>
  <td></td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#ReduceMax">ReduceMax</a></td>
  <td></td>
  <td>DML_OPERATOR_REDUCE with DML_REDUCE_FUNCTION_MAX</td>
  <td>x = max(max(max(x1, x2), x3), ..., xn)</td>
  <td></td>
 </tr>
 <tr>
  <td>Reduction</td>
  <td></td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#ReduceMin">ReduceMin</a></td>
  <td></td>
  <td>DML_OPERATOR_REDUCE with DML_REDUCE_FUNCTION_MIN</td>
  <td>x = min(min(min(x1, x2), x3), ..., xn)</td>
  <td></td>
 </tr>
 <tr>
  <td>Reduction</td>
  <td></td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#ArgMax">ArgMax</a></td>
  <td></td>
  <td>DML_OPERATOR_REDUCE with DML_REDUCE_FUNCTION_ARGMAX</td>
  <td>int32 {i j k ..} = maxindex(X Y Z …)</td>
  <td></td>
 </tr>
 <tr>
  <td>Reduction</td>
  <td></td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#ArgMin">ArgMin</a></td>
  <td></td>
  <td>DML_OPERATOR_REDUCE with DML_REDUCE_FUNCTION_ARGMIN</td>
  <td>int32 {i j k ..} = minindex(X Y Z …)</td>
  <td></td>
 </tr>
 <tr>
  <td>Imaging Operators</td>
  <td></td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#ImageScaler">ImageScaler</a></td>
  <td></td>
  <td>DML_OPERATOR_VALUE_SCALE_2D</td>
  <td></td>
  <td></td>
 </tr>
 <tr>
  <td>Imaging Operators</td>
  <td></td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#Upsample">Upsample</a></td>
  <td></td>
  <td>DML_OPERATOR_UPSAMPLE_2D</td>
  <td></td>
  <td></td>
 </tr>
 <tr>
  <td>Control Flow</td>
  <td>If</td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#If">If</a></td>
  <td></td>
  <td>---</td>
  <td></td>
  <td></td>
 </tr>
 <tr>
  <td>Control Flow</td>
  <td>Loop</td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#Loop">Loop</a></td>
  <td></td>
  <td>---</td>
  <td></td>
  <td></td>
 </tr>
 <tr>
  <td>Control Flow</td>
  <td>Scan</td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#Scan">Scan</a></td>
  <td></td>
  <td>---</td>
  <td></td>
  <td></td>
 </tr>
 <tr>
  <td>Normalization</td>
  <td></td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#InstanceNormalization">InstanceNormalization</a></td>
  <td></td>
  <td>DML_OPERATOR_MEAN_VARIANCE_NORMALIZATION
  with acrossChannels=false, normalizeVariance=true, scale and bias provided</td>
  <td>y = scale * (x - mean) / sqrt(variance + epsilon) + B
  mean and variance are computed per instance per channel.
  mean = (x0 + x1 …) / xn;
  variance = ((x0 - xmean)^2 + (x1 - xmean)^2 …) / xn</td>
  <td></td>
 </tr>
 <tr>
  <td>Normalization</td>
  <td></td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#BatchNormalization">BatchNormalization</a></td>
  <td></td>
  <td>DML_OPERATOR_BATCH_NORMALIZATION</td>
  <td>y = scale * (x - batchMean) / sqrt(batchVariance + epsilon) + bias</td>
  <td></td>
 </tr>
 <tr>
  <td>Normalization</td>
  <td>Local Response Normalization</td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#LRN">LRN</a></td>
  <td></td>
  <td>DML_OPERATOR_LOCAL_RESPONSE_NORMALIZATION</td>
  <td>y = x / (bias + (alpha/size) * sum(xi^2 for every xi in the local region))^beta
  defaults: bias = 1</td>
  <td></td>
 </tr>
 <tr>
  <td>Normalization</td>
  <td></td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#MeanVarianceNormalization">MeanVarianceNormalization</a></td>
  <td></td>
  <td>DML_OPERATOR_MEAN_VARIANCE_NORMALIZATION</td>
  <td>Exponent = Const(2.0)
  Epsilon = Const(1e-9)
  X_RM = ReduceMean(X)
  EX_squared = Pow(X_RM, Exponent)
  X_squared = Pow(X, Exponent)
  E_Xsquared = ReduceMean(X_squared)
  Variance = Sub(E_Xsquared, EX_squared)
  STD = Sqrt(Variance)
  X_variance = Sub(X, X_RM)
  Processed_STD = Add(STD, Epsilon)
  X_MVN = Div(X_variance, Processed_STD)</td>
  <td></td>
 </tr>
 <tr>
  <td>Normalization</td>
  <td></td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#LpNormalization">LpNormalization</a></td>
  <td></td>
  <td>DML_OPERATOR_LP_NORMALIZATION</td>
  <td>?</td>
  <td></td>
 </tr>
 <tr>
  <td>?</td>
  <td>Nonzero Indices List</td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/rel-1.4.0/docs/Operators.md#NonZero">Nonzero</a></td>
  <td></td>
  <td></td>
  <td></td>
  <td></td>
 </tr>
 <tr>
  <td>?</td>
  <td>Term Frequency Inverse Document Frequency Vectorizer</td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/rel-1.4.0/docs/Operators.md#tfidfvectorizer">TfldfVectorizer</a></td>
  <td></td>
  <td></td>
  <td></td>
  <td></td>
 </tr>
 <tr>
  <td>Aggregate DNN operators</td>
  <td>Recurrent Neural Network</td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#RNN">RNN</a></td>
  <td></td>
  <td>DML_OPERATOR_RNN</td>
  <td>Y = Activation(clip(MatMul(X,
  Transpose(W)) + MatMul(Initial_h, Transpose(R)) + b))</td>
  <td></td>
 </tr>
 <tr>
  <td>Aggregate DNN operators</td>
  <td>Gated Recurrent Unit</td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#GRU">GRU</a></td>
  <td></td>
  <td>DML_OPERATOR_GRU</td>
  <td>Z = Activation1(clip( X *
  transpose(W1) + Initial_h1 * transpose(R1) + b1))
  R = Activation1(clip( X * transpose(W2) + Initial_h1 * transpose(R2) + b2))
  C = Initial_h1 .* R
  O = Activation2(clip( X * transpose(W3) + Initial_h1 * transpose(R3) + b3))
  Y = (1-Z) .* O + Z .* Initial_h1
  &nbsp;
  (W = [W1, W2, W3]; b1 = B[0, :] +
  B[3*hidden_size]; b2 = B[1, :] +
  B[4*hidden_size, :]; b3 = B[2, :] +
  B[4*hidden_size, :];)</td>
  <td></td>
 </tr>
 <tr>
  <td>Aggregate DNN operators</td>
  <td>Long Short Term Memory</td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#LSTM">LSTM</a></td>
  <td></td>
  <td>DML_OPERATOR_LSTM</td>
  <td>I = Activation1(clip(X * transpose(W1)
  + Initial_h1 * transpose(R1) + p .* initial_c + b1))
  F = Activation1(clip(X * transpose(W2)
  + Initial_h1 * transpose(R2) +Â  p .* initial_c + b2))
  Z = Activation2(clip(X * transpose(W3)
  + Initial_h1 * transpose(R3) + b3))
  C = Initial_h1 .* F + I .* Z
  O = Activation2(clip( X * tr(W4) +
  Initial_h1 * tr(R4) + p .* initial_c + b4))
  Y = Activation3(C) .* O
  &nbsp;
  (W = [W1, W2, W3, W4]; b1 = B[0, :] +
  B[4*hidden_size]; b2 = B[1, :] +
  B[5*hidden_size, :]; b3 = B[2, :] +
  B[6*hidden_size, :]; b4 = B[3, :] + B[7*hidden_size, :];)</td>
  <td></td>
 </tr>
 <tr>
  <td>Training</td>
  <td></td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/master/docs/Operators.md#Dropout">Dropout</a></td>
  <td></td>
  <td>Identity actually (useful during
  training, not trained execution)</td>
  <td>For training: f(x) = iif(randomvalue0to1() < ratio, x, 0); For forward execution: f(x) = x, or rather f(X) = Identity(X)</td>
  <td>Selected randomly per element.</td>
 </tr>
 <tr>
  <td>Deleted</td>
  <td></td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/rel-1.0/docs/Operators.md#GRUUnit">GRUUnit</a></td>
  <td></td>
  <td>NA, experimental</td>
  <td>??</td>
  <td></td>
 </tr>
 <tr>
  <td>Deleted</td>
  <td></td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/rel-1.0/docs/Operators.md#Scale">Scale</a></td>
  <td></td>
  <td>NA, experimental, but can use Mul with broadcasting</td>
  <td>f(x) = x * scale</td>
  <td>Just use Mul(X, scale)</td>
 </tr>
 <tr>
  <td>Deleted</td>
  <td></td>
  <td>ONNX <a href="https://github.com/onnx/onnx/blob/rel-1.0/docs/Operators.md#ATen">ATen</a></td>
  <td></td>
  <td>NA, experimental</td>
  <td>??</td>
  <td></td>
 </tr>
</table>
