using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace NeuralSharp
{
    internal static class Backbone
    {
        internal static T[,] CreateArray<T>(int width, int height)
        {
            return new T[width, height];
        }

        internal static T[] CreateArray<T>(int size)
        {
            return new T[size];
        }

        internal static void ApplyNeuronsString(float[] input, int inputSkip, float[] output, int outputSkip, int length, Func<float, float> function)
        {
            for (int i = 0; i < length; i++)
            {
                output[outputSkip + i] = function(input[inputSkip + i]);
            }
        }

        internal static void BackpropagateNeuronsString(float[] input, int inputSkip, float[] output, int outputSkip, int length, float[] outputError, int outputErrorSkip, float[] inputError, int inputErrorSkip, Func<float, float, float> functionDerivative, bool learning)
        {
            for (int i = 0; i < length; i++)
            {
                inputError[inputErrorSkip + i] = outputError[outputErrorSkip + i] * functionDerivative(input[inputSkip + i], output[outputSkip + i]);
            }
        }

        internal static void ApplySoftmax(float[] input, int inputSkip, float[] output, int outputSkip, int length)
        {
            float inputMax = float.NegativeInfinity;
            for (int i = 0; i < length; i++)
            {
                inputMax = Math.Max(inputMax, input[i]);
            }
            float expSum = 0.0F;
            for (int i = 0; i < length; i++)
            {
                expSum += (float)Math.Exp(input[inputSkip + i] - inputMax);
            }
            for (int i = 0; i < length; i++)
            {
                output[outputSkip + i] = (float)Math.Exp(input[inputSkip + i] - inputMax) / expSum;
            }
        }

        internal static void BackpropagateSoftmax(float[] input, int inputSkip, float[] output, int outputSkip, int length, float[] outputError, int outputErrorSkip, float[] inputError, int inputErrorSkip, bool learning)
        {
            for (int i = 0; i < length; i++)
            {
                float inputErrorValue = 0.0F;
                for (int j = 0; j < length; j++)
                {
                    float derivative;
                    if (j == i)
                    {
                        derivative = output[outputSkip + i] * (1.0F - output[outputSkip + j]);
                    }
                    else
                    {
                        derivative = -output[outputSkip + i] * output[outputSkip + j];
                    }
                    inputErrorValue += outputError[outputErrorSkip + j] * derivative;
                }
                inputError[inputErrorSkip + i] = inputErrorValue;
            }
        }

        internal static void ApplyConnectionMatrix(float[] input, int inputSkip, int inputLength, float[] output, int outputSkip, int outputLength, float[,] weights)
        {
            for (int i = 0; i < outputLength; i++)
            {
                float outputValue = 0.0F;
                for (int j = 0; j < inputLength; j++)
                {
                    outputValue += input[inputSkip + j] * weights[j, i];
                }
                output[outputSkip + i] = outputValue;
            }
        }

        internal static void UpdateConnectionMatrix(float[,] weights, float[,] gradients, float[,] oldUpdates, int inputLength, int outputLength, float rate, float momentum)
        {
            for (int i = 0; i < inputLength; i++)
            {
                for (int j = 0; j < outputLength; j++)
                {
                    oldUpdates[i, j] = gradients[i, j] * rate + oldUpdates[i, j] * momentum;
                    weights[i, j] += oldUpdates[i, j];
                    gradients[i, j] = 0.0F;
                }
            }
        }

        internal static void BackpropagateConnectionMatrix(float[] input, int inputSkip, int inputLength, float[] output, int outputSkip, int outputLength, float[,] weights, float[] outputError, int outputErrorSkip, float[] inputError, int inputErrorSkip, float[,] gradients, bool learning)
        {
            if (learning)
            {
                for (int i = 0; i < inputLength; i++)
                {
                    float inputErrorValue = 0.0F;
                    for (int j = 0; j < outputLength; j++)
                    {
                        inputErrorValue += outputError[outputErrorSkip + j] * weights[i, j];
                        gradients[i, j] += input[inputSkip + i] * outputError[outputErrorSkip + j];
                    }
                    inputError[inputErrorSkip + i] = inputErrorValue;
                }
            }
            else
            {
                for (int i = 0; i < inputLength; i++)
                {
                    float inputErrorValue = 0.0F;
                    for (int j = 0; j < outputLength; j++)
                    {
                        inputErrorValue += outputError[outputErrorSkip + j] * weights[i, j];
                    }
                    inputError[inputErrorSkip + i] = inputErrorValue;
                }
            }
        }

        internal static void ApplyBiasedConnectionMatrix(float[] input, int inputSkip, int inputLength, float[] output, int outputSkip, int outputLength, float[,] weights, float[] biases)
        {
            for (int i = 0; i < outputLength; i++)
            {
                float outputVal = 0.0F;
                for (int j = 0; j < inputLength; j++)
                {
                    outputVal += input[inputSkip + j] * weights[j, i];
                }
                output[outputSkip + i] = outputVal + biases[i];
            }
        }

        internal static void ImageToArray(float[] image, int imageDepth, int imageWidth, int imageHeight, int depth, int width, int height, float[] array, int skip)
        {
            for (int i = 0; i < depth; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    for (int k = 0; k < height; k++)
                    {
                        array[skip + i * width * height + j * height + k] = image[i * imageWidth * imageHeight + j * imageHeight + k];
                    }
                }
            }
        }

        internal static void UpdateBiasedConnectionMatrix(float[,] weights, float[,] gradients, float[,] oldUpdates, float[] biases, float[] biasGradients, float[] oldBiasUpdates, int inputLength, int outputLength, float rate, float momentum)
        {
            for (int i = 0; i < outputLength; i++)
            {
                for (int j = 0; j < inputLength; j++)
                {
                    weights[j, i] += (oldUpdates[j, i] = gradients[j, i] * rate + momentum * oldUpdates[j, i]);
                    gradients[j, i] = 0.0F;
                }
                biases[i] += (oldBiasUpdates[i] = biasGradients[i] * rate + momentum * oldBiasUpdates[i]);
                biasGradients[i] = 0.0F;
            }
        }

        internal static void BackpropagateBiasedConnectionMatrix(float[] input, int inputSkip, int inputLength, float[] output, int outputSkip, int outputLength, float[,] weights, float[] biases, float[] outputError, int outputErrorSkip, float[] inputError, int inputErrorSkip, float[,] weightsGradients, float[] biasGradients, bool learning)
        {
            for (int i = 0; i < inputLength; i++)
            {
                float inputErrorVal = 0.0F;
                for (int j = 0; j < outputLength; j++)
                {
                    inputErrorVal += outputError[outputErrorSkip + j] * weights[i, j];
                    if(learning)
                    {
                        weightsGradients[i, j] += outputError[outputErrorSkip + j] * input[inputSkip + i];
                    }
                }
                inputError[inputErrorSkip + i] = inputErrorVal;
            }
            if (learning)
            {
                for (int i = 0; i < outputLength; i++)
                {
                    biasGradients[i] += outputError[outputErrorSkip + i];
                }
            }
        }

        internal static void ApplyDropout(float[] input, int inputSkip, float[] output, int outputSkip, int length, bool[] dropped, float dropChance, bool learning)
        {
            if (learning)
            {
                for (int i = 0; i < length; i++)
                {
                    dropped[i] = RandomGenerator.GetDouble() < dropChance;
                    if (dropped[i])
                    {
                        output[outputSkip + i] = 0.0F;
                    }
                    else
                    {
                        output[outputSkip + i] = input[inputSkip + i];
                    }
                }
            }
            else
            {
                for (int i = 0; i < length; i++)
                {
                    output[outputSkip + i] = input[inputSkip + i] * (1.0F - dropChance);
                }
            }
        }

        internal static void BackpropagateDropout(float[] input, int inputSkip, float[] output, int outputSkip, int length, bool[] dropped, float dropChance, bool learning, float[] outputError, int outputErrorSkip, float[] inputError, int inputErrorSkip)
        {
            if (learning)
            {
                for (int i = 0; i < length; i++)
                {
                    if (dropped[i])
                    {
                        inputError[inputErrorSkip + i] = 0.0F;
                    }
                    else
                    {
                        inputError[inputErrorSkip + i] = outputError[outputErrorSkip + i];
                    }
                }
            }
            else
            {
                for (int i = 0; i < length; i++)
                {
                    inputError[inputErrorSkip + i] = outputError[outputErrorSkip + i] * (1.0F - dropChance);
                }
            }
        }

        internal static void RandomizeArray(float[] array, int skip, int length, float variance)
        {
            for (int i = 0; i < length; i++)
            {
                array[skip + i] = RandomGenerator.GetNormalNumber(variance);
            }
        }

        internal static void CopyArray<T>(T[] array, int arraySkip, T[] copy, int copySkip, int length)
        {
            Array.Copy(array, arraySkip, copy, copySkip, length);
        }

        internal static void ArrayToMatrix<T>(T[] array, int arraySkip, T[,] matrix, int width, int height)
        {
            for (int i = 0; i < width; i++)
            {
                for (int j = 0; j < height; j++)
                {
                    matrix[i, j] = array[arraySkip + i * height + j];
                }
            }
        }

        internal static void MatrixToArray<T>(T[,] matrix, int width, int height, T[] array, int arraySkip)
        {
            for (int i = 0; i < width; i++)
            {
                for (int j = 0; j < height; j++)
                {
                    array[arraySkip + i * height + j] = matrix[i, j];
                }
            }
        }

        internal static float GetError(float[] output, int outputSkip, float[] expected, int expectedSkip, float[] error, int errorSkip, int length)
        {
            float retVal = 0.0F;
            for (int i = 0; i < length; i++)
            {
                error[errorSkip + i] = expected[expectedSkip + i] - output[i];
                retVal += error[errorSkip + i] * error[errorSkip + i];
            }
            return (float)Math.Sqrt(retVal);
        }

        internal static void ImageToImage(float[] thisImage, int thisDepth, int thisWidth, int thisHeight, float[] source, int sourceDepth, int sourceWidth, int sourceHeight, int sourceW, int sourceX, int sourceY, int thisW, int thisX, int thisY, int depth, int width, int height)
        {
            for (int i = 0; i < depth; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    for (int k = 0; k < height; k++)
                    {
                        thisImage[(thisW + i) * thisWidth * thisHeight + (thisX + j) * thisHeight + thisY + k] = source[(sourceW + i) * sourceWidth * sourceHeight + (sourceX + j) * sourceHeight + sourceY + k];
                    }
                }
            }
        }
        
        internal static void ArrayToImage(float[] thisImage, int thisDepth, int thisWidth, int thisHeight, float[] array, int skip, int thisW, int thisX, int thisY, int depth, int width, int height)
        {
            for (int i = 0; i < depth; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    for (int k = 0; k < height; k++)
                    {
                        thisImage[(thisW + i) * thisWidth * thisHeight + (thisX + j) * thisHeight + thisY + k] = (float)array[skip + i * width * height + j * height + k];
                    }
                }
            }
        }

        internal static void ApplyMaxPool(float[] input, int inputDepth, int inputWidth, int inputHeight, float[] output, int outputDepth, int outputWidth, int outputHeight, int xScale, int yScale)
        {
            for (int i = 0; i < outputDepth; i++)
            {
                for (int j = 0; j < outputWidth; j++)
                {
                    for (int k = 0; k < outputHeight; k++)
                    {
                        float outputValue = float.NegativeInfinity;
                        for (int l = 0; l < xScale; l++)
                        {
                            for (int m = 0; m < yScale; m++)
                            {
                                outputValue = Math.Max(outputValue, input[i * inputWidth * inputHeight + (j * xScale + l) * inputHeight + k * yScale + m]);
                            }
                        }
                        if (true)
                        {

                        }
                        output[i * outputWidth * outputHeight + j * outputHeight + k] = outputValue;
                    }
                }
            }
        }

        internal static void BackpropagateMaxPool(float[] input, int inputDepth, int inputWidth, int inputHeight, float[] output, int outputDepth, int outputWidth, int outputHeight, int xScale, int yScale, float[] outputError, int outputErrorDepth, int outputErrorWidth, int outputErrorHeight, float[] inputError, int inputErrorDepth, int inputErrorWidth, int inputErrorHeight, bool learning)
        {
            for (int i = 0; i < outputDepth; i++)
            {
                for (int j = 0; j < outputWidth; j++)
                {
                    for (int k = 0; k < outputHeight; k++)
                    {
                        float outputValue = output[i * outputWidth * outputHeight + j * outputHeight + k];
                        for (int l = 0; l < xScale; l++)
                        {
                            for (int m = 0; m < yScale; m++)
                            {
                                if (input[i * inputWidth * inputHeight + (j * xScale + l) * inputHeight + k * yScale + m] == outputValue)
                                {
                                    inputError[i * inputErrorWidth * inputErrorHeight + (j * xScale + l) * inputErrorHeight + k * yScale + m] = outputError[i * outputErrorWidth * outputErrorHeight + j * outputErrorHeight + k];
                                }
                                else
                                {
                                    inputError[i * inputErrorWidth * inputErrorHeight + (j * xScale + l) * inputErrorHeight + k * yScale + m] = 0.0F;
                                }
                            }
                        }
                    }
                }
            }
        }

        internal static void ApplyConvolution(float[] biases, float[] input, int inputDepth, int inputWidth, int inputHeight, float[] output, int outputDepth, int outputWidth, int outputHeight, float[,] kernels, int kernelSide, int stride, int scale, int padding, Func<float, float> function)
        {
            for (int i = 0; i < outputDepth; i++)
            {
                for (int j = 0; j < outputWidth; j++)
                {
                    for (int k = 0; k < outputHeight; k++)
                    {
                        float outputValue = 0.0F;
                        for (int l = ((j - padding) % scale + scale) % scale; l < kernelSide; l += scale)
                        {
                            for (int m = ((k - padding) % scale + scale) % scale; m < kernelSide; m += scale)
                            {
                                int inputX = (j * stride - padding + l) / scale;
                                int inputY = (k * stride - padding + m) / scale;
                                if (0 <= inputX && inputX < inputWidth && 0 <= inputY && inputY < inputHeight)
                                {
                                    for (int n = 0; n < inputDepth; n++)
                                    {
                                        outputValue += kernels[i, n * kernelSide * kernelSide + l * kernelSide + m] * input[n * inputWidth * inputHeight + inputX * inputHeight + inputY];
                                    }
                                }
                            }
                        }
                        if (biases != null)
                        {
                            outputValue += biases[i];
                        }
                        output[i * outputWidth * outputHeight + j * outputHeight + k] = function(outputValue);
                    }
                }
            }
        }

        internal static void BackpropagateConvolution(float[] biases, float[] biasesDeltas, float[] input, int inputDepth, int inputWidth, int inputHeight, float[] output, int outputDepth, int outputWidth, int outputHeight, float[,] kernels, int kernelSide, int stride, int scale, int padding, float[] outputError, int outputErrorDepth, int outputErrorWidth, int outputErrorHeight, float[] inputError, int inputErrorDepht, int inputErrorWidth, int inputErrorHeight, Func<float, float> functionDerivative, float[,] gradients, bool learning)
        {
            for (int i = 0; i < inputDepth; i++)
            {
                for (int j = 0; j < inputWidth; j++)
                {
                    for (int k = 0; k < inputHeight; k++)
                    {
                        float inputErrorValue = 0.0F;
                        for (int l = j % stride; l < kernelSide; l += stride)
                        {
                            for (int m = k % stride; m < kernelSide; m += stride)
                            {
                                int outputX = (j * scale - l + padding) / stride;
                                int outputY = (k * scale - m + padding) / stride;
                                if (0 <= outputX && outputX < outputWidth && 0 <= outputY && outputY < outputHeight)
                                {
                                    for (int n = 0; n < outputDepth; n++)
                                    {
                                        float error = outputError[n * outputErrorWidth * outputErrorHeight + outputX * outputErrorHeight + outputY] * functionDerivative(output[n * outputWidth * outputHeight + outputX * outputHeight + outputY]);
                                        inputErrorValue += kernels[n, i * kernelSide * kernelSide + l * kernelSide + m] * error;
                                        if (learning)
                                        {
                                            gradients[n, i * kernelSide * kernelSide + l * kernelSide + m] += input[i * inputWidth * inputHeight + j * inputHeight + k] * error;
                                        }
                                    }
                                }
                            }
                        }
                        inputError[i * inputErrorWidth * inputErrorHeight + j * inputErrorHeight + k] = inputErrorValue;
                    }
                }
            }
            if (biases != null)
            {
                for (int i = 0; i < outputWidth; i++)
                {
                    for (int j = 0; j < outputHeight; j++)
                    {
                        for (int k = 0; k < outputDepth; k++)
                        {
                            float error = outputError[k * outputErrorWidth * outputErrorHeight + i * outputErrorHeight + j] * functionDerivative(output[k * outputWidth * outputHeight + i * outputHeight + j]);
                            biasesDeltas[k] += error;
                        }
                    }
                }
            }
        }

        /*internal static void ApplyDeConvolution(float[] biases, float[] input, int inputDepth, int inputWidth, int inputHeight, float[] output, int outputDepth, int outputWidth, int outputHeight, float[,] kernels, int kernelSide, int stride, int padding, Func<float, float> function)
        {
            for (int i = 0; i < outputDepth; i++)
            {
                for (int j = 0; j < outputWidth; j++)
                {
                    for (int k = 0; k < outputHeight; k++)
                    {
                        float outputValue = 0.0F;
                        for (int l = ((j - padding) % stride + stride) % stride; l < kernelSide; l += stride)
                        {
                            for (int m = ((k - padding) % stride + stride) % stride; m < kernelSide; m += stride)
                            {
                                int inputX = (j - padding) / stride + l;
                                int inputY = ((k - padding) / stride + m);
                                if (0 <= inputX && inputX < inputWidth && 0 <= inputY && inputY < inputHeight)
                                {
                                    for (int n = 0; n < inputDepth; n++)
                                    {
                                        outputValue += kernels[i, n * kernelSide * kernelSide + l * kernelSide + m] * input[n * inputWidth * inputHeight + inputX * inputHeight + inputY];
                                    }
                                }
                            }
                        }
                        //outputValue += biases[i];
                        output[i * outputWidth * outputHeight + j * outputHeight + k] = function(outputValue);
                    }
                }
            }
        }*/

        /*internal static void BackpropagateDeConvolution(float[] biases, float[] biasesDeltas, float[] input, int inputDepth, int inputWidth, int inputHeight, float[] output, int outputDepth, int outputWidth, int outputHeight, float[,] kernels, int kernelSide, int stride, int padding, float[] outputError, int outputErrorDepth, int outputErrorWidth, int outputErrorHeight, float[] inputError, int inputErrorDepht, int inputErrorWidth, int inputErrorHeight, Func<float, float> functionDerivative, float[,] gradients, bool learning)
        {
            for (int i = 0; i < inputDepth; i++)
            {
                for (int j = 0; j < inputWidth; j++)
                {
                    for (int k = 0; k < inputHeight; k++)
                    {
                        float inputErrorValue = 0.0F;
                        for (int l = 0; l < kernelSide; l++)
                        {
                            for (int m = 0; m < kernelSide; m++)
                            {
                                int outputX = j * stride - l + padding;
                                int outputY = k * stride - m + padding;
                                if (0 <= outputX && outputX < outputWidth && 0 <= outputY && outputY < outputHeight)
                                {
                                    for (int n = 0; n < outputDepth; n++)
                                    {
                                        float error = outputError[n * outputErrorWidth * outputErrorHeight + outputX * outputErrorHeight + outputY] * functionDerivative(output[n * outputWidth * outputHeight + outputX * outputHeight + outputY]);
                                        inputErrorValue += kernels[n, i * kernelSide * kernelSide + l * kernelSide + m] * error;
                                        if (learning)
                                        {
                                            gradients[n, i * kernelSide * kernelSide + l * kernelSide + m] += input[i * inputWidth * inputHeight + j * inputHeight + k] * error;
                                        }
                                    }
                                }
                            }
                        }
                        inputError[i * inputErrorWidth * inputErrorHeight + j * inputErrorHeight + k] = inputErrorValue;
                    }
                }
            }
            for (int i = 0; i < outputWidth; i++)
            {
                for (int j = 0; j < outputHeight; j++)
                {
                    for (int k = 0; k < outputDepth; k++)
                    {
                        float error = outputError[k * outputErrorWidth * outputErrorHeight + i * outputErrorHeight + j] * functionDerivative(output[k * outputWidth * outputHeight + i * outputHeight + j]);
                        biasesDeltas[k] += error;
                    }
                }
            }
        }*/

        internal static void RandomizeMatrix(float[,] matrix, int width, int height, float variance)
        {
            for (int i = 0; i < width; i++)
            {
                for (int j = 0; j < height; j++)
                {
                    matrix[i, j] = (float)RandomGenerator.GetNormalNumber(variance);
                }
            }
        }

        internal static void UpdateConvolution(float[] biasesDeltas, float[] biasesMomentums, float[] biases, float[,] kernels, float[,] gradients, float[,] oldUpdates, int inputDepth, int inputWidth, int inputHeight, int outputDepth, int outputWidth, int outputHeight, int kernelSide, float rate, float momentum)
        {
            for (int i = 0; i < outputDepth; i++)
            {
                for (int j = 0; j < inputDepth * kernelSide * kernelSide; j++)
                {
                    oldUpdates[i, j] = gradients[i, j] * rate + oldUpdates[i, j] * momentum;
                    kernels[i, j] += oldUpdates[i, j];
                    gradients[i, j] = 0.0F;
                }
                biasesMomentums[i] = biasesDeltas[i] * rate + biasesMomentums[i] * momentum;
                biases[i] += biasesMomentums[i];
                biasesDeltas[i] = 0.0F;
            }
        }

        /*internal static void UpdateDeConvolution(float[] biasesDeltas, float[] biasesMomentums, float[] biases, float[,] kernels, float[,] gradients, float[,] oldUpdates, int inputDepth, int inputWidth, int inputHeight, int outputDepth, int outputWidth, int outputHeight, int kernelSide, float rate, float momentum)
        {
            for (int i = 0; i < outputDepth; i++)
            {
                for (int j = 0; j < inputDepth * kernelSide * kernelSide; j++)
                {
                    oldUpdates[i, j] = gradients[i, j] * rate + oldUpdates[i, j] * momentum;
                    kernels[i, j] += oldUpdates[i, j];
                    gradients[i, j] = 0.0F;
                }
                biasesMomentums[i] = biasesDeltas[i] * rate + biasesMomentums[i] * momentum;
                biases[i] += biasesMomentums[i];
                biasesDeltas[i] = 0.0F;
            }
        }*/

        internal static void ApplyImageDropout(float[] input, float[] output, int depth, int width, int height, bool[] dropped, float dropChance, bool learning)
        {
            if (learning)
            {
                for (int i = 0; i < depth * width * height; i++)
                {
                    dropped[i] = RandomGenerator.GetDouble() < dropChance;
                    if (dropped[i])
                    {
                        output[i] = 0.0F;
                    }
                    else
                    {
                        output[i] = input[i];
                    }
                }
            }
            else
            {
                for (int i = 0; i < depth * width * height; i++)
                {
                    output[i] = input[i] * (float)(1.0F - dropChance);
                }
            }
        }

        internal static void BackpropagateImageDropout(float[] input, float[] output, int depth, int width, int height, bool[] dropped, float dropChance, bool learning, float[] outputError, int outputErrorDepth, int outputErrorWidth, int outputErrorHeight, float[] inputError, int inputErrorDepth, int inputErrorWidth, int inputErrorHeight)
        {
            if (learning)
            {
                for (int i = 0; i < depth; i++)
                {
                    for (int j = 0; j < width; j++)
                    {
                        for (int k = 0; k < height; k++)
                        {
                            if (dropped[i * width * height + j * height + k])
                            {
                                inputError[i * inputErrorWidth * inputErrorHeight + j * inputErrorHeight + k] = 0.0F;
                            }
                            else
                            {
                                inputError[i * inputErrorWidth * inputErrorHeight + j * inputErrorHeight + k] = outputError[i * outputErrorWidth * outputErrorHeight + j * outputErrorHeight + k];
                            }
                        }
                    }
                }
            }
            else
            {
                for (int i = 0; i < depth; i++)
                {
                    for (int j = 0; j < width; j++)
                    {
                        for (int k = 0; k < height; k++)
                        {
                            inputError[i * inputErrorWidth * inputErrorHeight + j * inputErrorHeight + k] = outputError[i * outputErrorWidth * outputErrorHeight + j * outputErrorHeight + k] * (float)(1.0F - dropChance);
                        }
                    }
                }
            }
        }
    }
}
