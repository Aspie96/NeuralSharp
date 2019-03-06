#define AALEA_GPU

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
#if ALEA_GPU
using Alea;
using Alea.Parallel;
using Alea.CSharp;
#endif

[assembly: InternalsVisibleTo("UnitTestProject1")]
namespace NeuralSharp
{
    internal static class Backbone
    {
#if !ALEA_GPU
        #region !ALEA_GPU
        internal static T[,] CreateArray<T>(int width, int height)
        {
            return new T[width, height];
        }

        internal static T[] CreateArray<T>(int size)
        {
            return new T[size];
        }

        internal static void ApplyNeuronsString(double[] input, int inputSkip, double[] output, int outputSkip, int length, Func<double, double> function)
        {
            for (int i = 0; i < length; i++)
            {
                output[outputSkip + i] = function(input[inputSkip + i]);
            }
        }

        internal static void BackpropagateNeuronsString(double[] input, int inputSkip, double[] output, int outputSkip, int length, double[] outputError, int outputErrorSkip, double[] inputError, int inputErrorSkip, Func<double, double, double> functionDerivative, bool learning)
        {
            for (int i = 0; i < length; i++)
            {
                inputError[inputErrorSkip + i] = outputError[outputErrorSkip + i] * functionDerivative(input[inputSkip + i], output[outputSkip + i]);
            }
        }

        internal static void ApplySoftmax(double[] input, int inputSkip, double[] output, int outputSkip, int length)
        {
            double inputMax = double.NegativeInfinity;
            for (int i = 0; i < length; i++)
            {
                inputMax = Math.Max(inputMax, input[i]);
            }
            double expSum = 0.0;
            for (int i = 0; i < length; i++)
            {
                expSum += Math.Exp(input[inputSkip + i] - inputMax);
            }
            for (int i = 0; i < length; i++)
            {
                output[outputSkip + i] = Math.Exp(input[inputSkip + i] - inputMax) / expSum;
            }
        }

        internal static void BackpropagateSoftmax(double[] input, int inputSkip, double[] output, int outputSkip, int length, double[] outputError, int outputErrorSkip, double[] inputError, int inputErrorSkip, bool learning)
        {
            for (int i = 0; i < length; i++)
            {
                double inputErrorValue = 0.0;
                for (int j = 0; j < length; j++)
                {
                    double derivative;
                    if (j == i)
                    {
                        derivative = output[outputSkip + i] * (1.0 - output[outputSkip + j]);
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

        internal static void ApplyConnectionMatrix(double[] input, int inputSkip, int inputLength, double[] output, int outputSkip, int outputLength, double[,] weights)
        {
            for (int i = 0; i < outputLength; i++)
            {
                double outputValue = 0.0;
                for (int j = 0; j < inputLength; j++)
                {
                    outputValue += input[inputSkip + j] * weights[j, i];
                }
                output[outputSkip + i] = outputValue;
            }
        }

        internal static void UpdateConnectionMatrix(double[,] weights, double[,] gradients, double[,] oldUpdates, int inputLength, int outputLength, double rate, double momentum)
        {
            for (int i = 0; i < inputLength; i++)
            {
                for (int j = 0; j < outputLength; j++)
                {
                    oldUpdates[i, j] = gradients[i, j] * rate + oldUpdates[i, j] * momentum;
                    weights[i, j] += oldUpdates[i, j];
                    gradients[i, j] = 0.0;
                }
            }
        }

        internal static void BackpropagateConnectionMatrix(double[] input, int inputSkip, int inputLength, double[] output, int outputSkip, int outputLength, double[,] weights, double[] outputError, int outputErrorSkip, double[] inputError, int inputErrorSkip, double[,] gradients, bool learning)
        {
            if (learning)
            {
                for (int i = 0; i < inputLength; i++)
                {
                    double inputErrorValue = 0.0;
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
                    double inputErrorValue = 0.0;
                    for (int j = 0; j < outputLength; j++)
                    {
                        inputErrorValue += outputError[outputErrorSkip + j] * weights[i, j];
                    }
                    inputError[inputErrorSkip + i] = inputErrorValue;
                }
            }
        }

        internal static void ApplyBiasedConnectionMatrix(double[] input, int inputSkip, int inputLength, double[] output, int outputSkip, int outputLength, double[,] weights, double[] biases)
        {
            for (int i = 0; i < outputLength; i++)
            {
                double outputVal = 0.0;
                for (int j = 0; j < inputLength; j++)
                {
                    outputVal += input[inputSkip + j] * weights[j, i];
                }
                output[outputSkip + i] = outputVal + biases[i];
            }
        }

        internal static void ImageToArray(float[] image, int imageDepth, int imageWidth, int imageHeight, int depth, int width, int height, double[] array, int skip)
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

        internal static void UpdateBiasedConnectionMatrix(double[,] weights, double[,] gradients, double[,] oldUpdates, double[] biases, double[] biasGradients, double[] oldBiasUpdates, int inputLength, int outputLength, double rate, double momentum)
        {
            for (int i = 0; i < outputLength; i++)
            {
                for (int j = 0; j < inputLength; j++)
                {
                    weights[j, i] += (oldUpdates[j, i] = gradients[j, i] * rate + momentum * oldUpdates[j, i]);
                    gradients[j, i] = 0.0;
                }
                biases[i] += (oldBiasUpdates[i] = biasGradients[i] * rate + momentum * oldBiasUpdates[i]);
                biasGradients[i] = 0.0;
            }
        }

        internal static void BackpropagateBiasedConnectionMatrix(double[] input, int inputSkip, int inputLength, double[] output, int outputSkip, int outputLength, double[,] weights, double[] biases, double[] outputError, int outputErrorSkip, double[] inputError, int inputErrorSkip, double[,] weightsGradients, double[] biasGradients, bool learning)
        {
            for (int i = 0; i < inputLength; i++)
            {
                double inputErrorVal = 0.0;
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

        internal static void ApplyDropout(double[] input, int inputSkip, double[] output, int outputSkip, int length, bool[] dropped, double dropChance, bool learning)
        {
            if (learning)
            {
                for (int i = 0; i < length; i++)
                {
                    dropped[i] = RandomGenerator.GetDouble() < dropChance;
                    if (dropped[i])
                    {
                        output[outputSkip + i] = 0.0;
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
                    output[outputSkip + i] = input[inputSkip + i] * (1.0 - dropChance);
                }
            }
        }

        internal static void BackpropagateDropout(double[] input, int inputSkip, double[] output, int outputSkip, int length, bool[] dropped, double dropChance, bool learning, double[] outputError, int outputErrorSkip, double[] inputError, int inputErrorSkip)
        {
            if (learning)
            {
                for (int i = 0; i < length; i++)
                {
                    if (dropped[i])
                    {
                        inputError[inputErrorSkip + i] = 0.0;
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
                    inputError[inputErrorSkip + i] = outputError[outputErrorSkip + i] * (1.0 - dropChance);
                }
            }
        }

        internal static void RandomizeArray(double[] array, int skip, int length, double variance)
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

        internal static double GetError(double[] output, int outputSkip, double[] expected, int expectedSkip, double[] error, int errorSkip, int length)
        {
            double retVal = 0.0;
            for (int i = 0; i < length; i++)
            {
                error[errorSkip + i] = expected[expectedSkip + i] - output[i];
                retVal += error[errorSkip + i] * error[errorSkip + i];
            }
            return Math.Sqrt(retVal);
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
        
        internal static void ArrayToImage(float[] thisImage, int thisDepth, int thisWidth, int thisHeight, double[] array, int skip, int thisW, int thisX, int thisY, int depth, int width, int height)
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
                        double outputValue = output[i * outputWidth * outputHeight + j * outputHeight + k];
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
                                int inputX = (j * stride - padding) / scale + l;
                                int inputY = (k * stride - padding) / scale + m;
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
                                int outputX = j * scale - l + padding;
                                int outputY = k * scale - m + padding;
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

        internal static void RandomizeMatrix(float[,] matrix, int width, int height, double variance)
        {
            for (int i = 0; i < width; i++)
            {
                for (int j = 0; j < height; j++)
                {
                    matrix[i, j] = (float)RandomGenerator.GetNormalNumber(variance);
                }
            }
        }

        internal static void RandomizeMatrix(double[,] matrix, int width, int height, double variance)
        {
            for (int i = 0; i < width; i++)
            {
                for (int j = 0; j < height; j++)
                {
                    matrix[i, j] = RandomGenerator.GetNormalNumber(variance);
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

        internal static void ApplyImageDropout(float[] input, float[] output, int depth, int width, int height, bool[] dropped, double dropChance, bool learning)
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
                    output[i] = input[i] * (float)(1.0 - dropChance);
                }
            }
        }

        internal static void BackpropagateImageDropout(float[] input, float[] output, int depth, int width, int height, bool[] dropped, double dropChance, bool learning, float[] outputError, int outputErrorDepth, int outputErrorWidth, int outputErrorHeight, float[] inputError, int inputErrorDepth, int inputErrorWidth, int inputErrorHeight)
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
                            inputError[i * inputErrorWidth * inputErrorHeight + j * inputErrorHeight + k] = outputError[i * outputErrorWidth * outputErrorHeight + j * outputErrorHeight + k] * (float)(1.0 - dropChance);
                        }
                    }
                }
            }
        }
        #endregion
#endif
        
#if ALEA_GPU
        #region ALEA_GPU
        private static readonly Gpu gpu;

        static Backbone()
        {
            Backbone.gpu = Gpu.Default;
        }

        internal static T[,] CreateArray<T>(int width, int height)
        {
            return new T[width, height];
        }

        internal static T[] CreateArray<T>(int size)
        {
            return new T[size];
        }

        internal static void ApplyNeuronsString(double[] input, int inputSkip, double[] output, int outputSkip, int length, Func<double, double> function)
        {
            Backbone.gpu.For(0, length, delegate (int i)
            {
                output[outputSkip + i] = function(input[inputSkip + i]);
            });
        }

        internal static void BackpropagateNeuronsString(double[] input, int inputSkip, double[] output, int outputSkip, int length, double[] outputError, int outputErrorSkip, double[] inputError, int inputErrorSkip, Func<double, double, double> functionDerivative, bool learning)
        {
            Backbone.gpu.For(0, length, delegate (int i)
            {
                inputError[inputErrorSkip + i] = outputError[outputErrorSkip + i] * functionDerivative(input[inputSkip + i], output[outputSkip + i]);
            });
        }

        internal static void ApplySoftmax(double[] input, int inputSkip, double[] output, int outputSkip, int length)
        {
            double inputMax = Backbone.gpu.Aggregate(length, delegate (int i)
            {
                return input[inputSkip + i];
            }, delegate (double a, double b)
            {
                if (a > b)
                {
                    return a;
                }
                return b;
            });
            double expSum = Backbone.gpu.Aggregate(length, delegate (int i)
            {
                return Math.Exp(input[inputSkip + i] - inputMax);
            }, delegate (double a, double b)
            {
                return a + b;
            });
            Backbone.gpu.For(0, length, delegate (int i)
            {
                output[outputSkip + i] = Math.Exp(input[inputSkip + i] - inputMax) / expSum;
            });
        }

        internal static void BackpropagateSoftmax(double[] input, int inputSkip, double[] output, int outputSkip, int length, double[] outputError, int outputErrorSkip, double[] inputError, int inputErrorSkip, bool learning)
        {
            Backbone.gpu.For(0, length, delegate (int i)
            {
                double inputErrorValue = 0.0;
                for (int j = 0; j < length; j++)
                {
                    double derivative;
                    if (i == j)
                    {
                        derivative = output[outputSkip + i] * (1.0 - output[outputSkip + j]);
                    }
                    else
                    {
                        derivative = output[outputSkip + i] * output[outputSkip + j];
                    }
                    inputErrorValue += outputError[outputErrorSkip + i] * derivative;
                }
                inputError[inputErrorSkip + i] = inputErrorValue;
            });
        }

        internal static void ApplyConnectionMatrix(double[] input, int inputSkip, int inputLength, double[] output, int outputSkip, int outputLength, double[,] weights)
        {
            Backbone.gpu.For(0, outputLength, delegate (int i)
            {
                double outputValue = 0.0;
                for (int j = 0; j < inputLength; j++)
                {
                    outputValue += input[inputSkip + j] * weights[j, i];
                }
                output[outputSkip + i] = outputValue;
            });
        }

        internal static void UpdateConnectionMatrix(double[,] weights, double[,] gradients, double[,] oldUpdates, int inputLength, int outputLength, double rate, double momentum)
        {
            Backbone.gpu.For(0, inputLength, delegate (int i)
            {
                for (int j = 0; j < outputLength; j++)
                {
                    double update = gradients[i, j] * rate + oldUpdates[i, j] * momentum;
                    double weight = weights[i, j] + update;
                    weights[i, j] = weight;
                    oldUpdates[i, j] = update;
                    gradients[i, j] = 0.0;
                }
            });
        }

        internal static void BackpropagateConnectionMatrix(double[] input, int inputSkip, int inputLength, double[] output, int outputSkip, int outputLength, double[,] weights, double[] outputError, int outputErrorSkip, double[] inputError, int inputErrorSkip, double[,] gradients, bool learning)
        {
            if (learning)
            {
                Backbone.gpu.For(0, inputLength, delegate (int i)
                {
                    double inputErrorValue = 0.0;
                    double inputValue = input[inputSkip + i];
                    for (int j = 0; j < outputLength; j++)
                    {
                        inputErrorValue += outputError[outputErrorSkip + j] * weights[i, j];
                        /*if (learning)
                        {
                            gradients[i, j] += inputValue * outputError[outputErrorSkip + j];
                        }*/
                    }
                    inputError[inputErrorSkip + i] = inputErrorValue;
                });
            }
            else
            {
                Backbone.gpu.For(0, inputLength, delegate (int i)
                {
                    double inputErrorValue = 0.0;
                    for (int j = 0; j < outputLength; j++)
                    {
                        inputErrorValue += outputError[outputErrorSkip + j] * weights[i, j];
                    }
                    inputError[inputErrorSkip + i] = inputErrorValue;
                });
            }
        }

        internal static void ApplyBiasedConnectionMatrix(double[] input, int inputSkip, int inputLength, double[] output, int outputSkip, int outputLength, double[,] weights, double[] biases)
        {
            Backbone.gpu.For(0, outputLength, delegate (int i)
            {
                double outputVal = 0.0;
                for (int j = 0; j < inputLength; j++)
                {
                    outputVal += input[inputSkip + j] * weights[j, i];
                }
                output[outputSkip + i] = outputVal + biases[i];
            });
        }

        internal static void ImageToArray(float[] image, int imageDepth, int imageWidth, int imageHeight, int depth, int width, int height, double[] array, int skip)
        {
            Backbone.gpu.For(0, depth * width * height, delegate (int i)
            {
                int w = i / (width * height);
                int x = (i % (width * height)) / height;
                int y = (i % (width * height)) % height;
                array[skip + i] = image[w * imageWidth * imageHeight + x * imageHeight + y];
            });
        }

        internal static void UpdateBiasedConnectionMatrix(double[,] weights, double[,] gradients, double[,] oldUpdates, double[] biases, double[] biasGradients, double[] oldBiasUpdates, int inputLength, int outputLength, double rate, double momentum)
        {
            Backbone.gpu.For(0, outputLength, delegate (int i)
            {
                for (int j = 0; j < inputLength; j++)
                {
                    double update = gradients[j, i] * rate + oldUpdates[j, i] * momentum;
                    double val = weights[j, i] + update;
                    weights[j, i] = val;
                    oldUpdates[j, i] = update;
                    gradients[j, i] = 0.0;
                }
                double bUp = biasGradients[i] * rate + oldBiasUpdates[i] * momentum;
                double bi = biases[i] + bUp;
                biases[i] = bi;
                oldBiasUpdates[i] = bUp;
                biasGradients[i] = 0.0;
            });
        }

        internal static void BackpropagateBiasedConnectionMatrix(double[] input, int inputSkip, int inputLength, double[] output, int outputSkip, int outputLength, double[,] weights, double[] biases, double[] outputError, int outputErrorSkip, double[] inputError, int inputErrorSkip, double[,] weightsGradients, double[] biasGradients, bool learning)
        {
            Backbone.gpu.For(0, inputLength, delegate (int i)
            {
                double inputErrorVal = 0.0;
                for (int j = 0; j < outputLength; j++)
                {
                    inputErrorVal += outputError[outputErrorSkip + j] * weights[i, j];
                    if (learning)
                    {
                        double gradient = weightsGradients[i, j];
                        weightsGradients[i, j] = gradient + outputError[outputErrorSkip + j] * input[inputSkip + i];
                    }
                }
                inputError[inputErrorSkip + i] = inputErrorVal;
            });
            if (learning)
            {
                Backbone.gpu.For(0, outputLength, delegate (int i)
                {
                    biasGradients[i] += outputError[outputErrorSkip + i];
                });
            }
        }

        internal static void ApplyDropout(double[] input, int inputSkip, double[] output, int outputSkip, int length, bool[] dropped, double dropChance, bool learning)
        {
            double[] cop = new double[length];
            for (int i = 0; i < length; i++)
            {
                cop[i] = RandomGenerator.GetDouble();
            }
            if (learning)
            {
                gpu.For(0, length, delegate (int i)
                {
                    dropped[i] = cop[i] < dropChance;
                    if (dropped[i])
                    {
                        output[outputSkip + i] = 0.0;
                    }
                    else
                    {
                        output[outputSkip + i] = input[inputSkip + i];
                    }
                });
            }
            else
            {
                gpu.For(0, length, delegate (int i)
                {
                    output[outputSkip + i] = input[inputSkip + i] * (1.0 - dropChance);
                });
            }
        }

        internal static void BackpropagateDropout(double[] input, int inputSkip, double[] output, int outputSkip, int length, bool[] dropped, double dropChance, bool learning, double[] outputError, int outputErrorSkip, double[] inputError, int inputErrorSkip)
        {
            if (learning)
            {
                Backbone.gpu.For(0, length, delegate (int i)
                {
                    if (dropped[i])
                    {
                        inputError[inputErrorSkip + i] = 0.0;
                    }
                    else
                    {
                        inputError[inputErrorSkip + i] = outputError[outputErrorSkip + i];
                    }
                });
            }
            else
            {
                Backbone.gpu.For(0, length, delegate (int i)
                {
                    inputError[inputErrorSkip + i] = inputError[inputErrorSkip + i] * (1.0 - dropChance);
                });
            }
        }

        internal static void RandomizeArray(double[] array, int skip, int length, double variance)
        {
            double[] cop = new double[length];
            for (int i = 0; i < length; i++)
            {
                cop[i] = RandomGenerator.GetNormalNumber(variance);
            }
            Backbone.gpu.For(0, length, delegate (int i)
            {
                array[skip + i] = cop[i];
            });
        }

        internal static void CopyArray<T>(T[] array, int arraySkip, T[] copy, int copySkip, int length)
        {
            Backbone.gpu.For(0, length, delegate (int i)
            {
                copy[arraySkip + i] = array[copySkip + i];
            });
        }

        internal static void ArrayToMatrix<T>(T[] array, int arraySkip, T[,] matrix, int width, int height)
        {
            Backbone.gpu.For(0, width * height, delegate (int i)
            {
                int x = i / height;
                int y = i % height;
                matrix[x, y] = array[arraySkip + i];
            });
        }

        internal static void MatrixToArray<T>(T[,] matrix, int width, int height, T[] array, int arraySkip)
        {
            Backbone.gpu.For(0, width * height, delegate (int i)
            {
                int x = i / height;
                int y = i % height;
                array[arraySkip + i] = matrix[x, y];
            });
        }

        internal static double GetError(double[] output, int outputSkip, double[] expected, int expectedSkip, double[] error, int errorSkip, int length)
        {
            return Math.Sqrt(Backbone.gpu.Aggregate(length, delegate (int i)
            {
                error[errorSkip + i] = expected[expectedSkip + i] - output[i];
                return error[errorSkip + i] * error[errorSkip + i];
            }, delegate (double valA, double valB)
            {
                return valA + valB;
            }));
        }

        internal static void ImageToImage(float[] thisImage, int thisDepth, int thisWidth, int thisHeight, float[] source, int sourceDepth, int sourceWidth, int sourceHeight, int sourceW, int sourceX, int sourceY, int thisW, int thisX, int thisY, int depth, int width, int height)
        {
            Backbone.gpu.For(0, depth * width * height, delegate (int i)
            {
                int w = i / (width * height);
                int x = (i % (width * height)) / height;
                int y = (i % (width * height)) % height;
                thisImage[(thisW + w) * thisWidth * thisHeight + (thisX + x) * thisHeight + thisY + y] = source[(sourceW + w) * sourceWidth * sourceHeight + (sourceX + x) * sourceHeight + sourceY + y];
            });
        }

        internal static void ArrayToImage(float[] thisImage, int thisDepth, int thisWidth, int thisHeight, double[] array, int skip, int thisW, int thisX, int thisY, int depth, int width, int height)
        {
            Backbone.gpu.For(0, depth * width * height, delegate (int i)
            {
                int w = i / (width * height);
                int x = (i % (width * height)) / height;
                int y = (i % (width * height)) % height;
                thisImage[(thisW + w) * thisWidth * thisHeight + (thisX + x) * thisHeight + thisY + y] = (float)array[skip + i];
            });
        }

        internal static void ApplyMaxPool(float[] input, int inputDepth, int inputWidth, int inputHeight, float[] output, int outputDepth, int outputWidth, int outputHeight, int xScale, int yScale)
        {
            Backbone.gpu.For(0, outputDepth * outputWidth * outputHeight, delegate (int i)
            {
                int w = i / (outputWidth * outputHeight);
                int x = (i % (outputWidth * outputHeight)) / outputHeight;
                int y = (i % (outputWidth * outputHeight)) % outputHeight;
                //if (w < outputDepth && x < outputWidth && y < outputHeight)
                {
                    float outputValue = float.NegativeInfinity;
                    for (int j = 0; j < xScale; j++)
                    {
                        for (int k = 0; k < yScale; k++)
                        {
                            float inputValue = input[w * inputWidth * inputHeight + (x * xScale + j) * inputHeight + y * yScale + k];
                            if (inputValue > outputValue)
                            {
                                outputValue = inputValue;
                            }
                        }
                    }
                    output[w * outputWidth * outputHeight + x * outputHeight + y] = outputValue;
                }
            }/*, new LaunchParam(new dim3((outputDepth + 3) / 4, (outputWidth + 3) / 4, (outputHeight + 3) / 4), new dim3(4, 4, 4))*/);
        }

        internal static void BackpropagateMaxPool(float[] input, int inputDepth, int inputWidth, int inputHeight, float[] output, int outputDepth, int outputWidth, int outputHeight, int xScale, int yScale, float[] outputError, int outputErrorDepth, int outputErrorWidth, int outputErrorHeight, float[] inputError, int inputErrorDepth, int inputErrorWidth, int inputErrorHeight, bool learning)
        {
            Backbone.gpu.For(0, outputDepth * outputWidth * outputHeight, delegate (int i)
            {
                int w = i / (outputWidth * outputHeight);
                int x = (i % (outputWidth * outputHeight)) / outputHeight;
                int y = (i % (outputWidth * outputHeight)) % outputHeight;
                //if (w < outputDepth && x < outputWidth && y < outputHeight)
                {
                    double outputValue = output[w * outputWidth * outputHeight + x * outputHeight + y];
                    int inc = 1;
                    for (int j = 0; j < xScale; j++)
                    {
                        for (int k = 0; k < yScale; k++)
                        {
                            if (input[w * inputWidth * inputHeight + (x * xScale + j) * inputHeight + y * yScale + k] == outputValue)
                            {
                                inputError[w * inputErrorWidth * inputErrorHeight + (x * xScale + j) * inputErrorHeight + y * yScale + k] = outputError[w * outputErrorWidth * outputErrorHeight + x * outputErrorHeight + y];
                            }
                            else
                            {
                                inputError[w * inputErrorWidth * inputErrorHeight + (x * xScale + j) * inputErrorHeight + y * yScale + k] = 0.0F;
                            }
                        }
                    }
                }
            }/*, new LaunchParam(new dim3((outputDepth + 3) / 4, (outputWidth + 3) / 4, (outputHeight + 3) / 4), new dim3(4, 4, 4))*/);
            for (int i = 0; i < output.Length; i++)
            {
                if (output[i] > 10.0 || output[i] < -10.0)
                {

                }
            }
        }

        internal static void ApplyConvolution(float[] input, int inputDepth, int inputWidth, int inputHeight, float[] output, int outputDepth, int outputWidth, int outputHeight, float[,] kernels, int kernelSide, int stride, int padding, Func<float, float> function)
        {
            //float[,] innerKernels = Gpu.Default.Allocate<float>(outputDepth, inputDepth * kernelSide * kernelSide);
            ///*Backbone.gpu.For(0, outputDepth * inputDepth * kernelSide * kernelSide, delegate (int i)
            //{
            //    innerKernels[i / (inputDepth * kernelSide * kernelSide), i % (inputDepth * kernelSide * kernelSide)] = kernels[i / (inputDepth * kernelSide * kernelSide)][(i % (inputDepth * kernelSide * kernelSide)) / kernelSide / kernelSide][(i % (inputDepth * kernelSide * kernelSide)) % (kernelSide * kernelSide)][i % (inputDepth * kernelSide * kernelSide)];
            //});*/
            //int kernelSideSquared = kernelSide * kernelSide;
            Backbone.gpu.For(0, outputDepth * outputWidth * outputHeight, delegate (int i)
            {
                int w = i / (outputWidth * outputHeight);
                int x = (i % (outputWidth * outputHeight)) / outputHeight;
                int y = (i % (outputWidth * outputHeight)) % outputHeight;
                //if (w < outputDepth && x < outputWidth && y < outputHeight)
                float outputValue = 0.0F;
                for (int j = 0; j < kernelSide; j++)
                {
                    for (int l = 0; l < kernelSide; l++)
                    {
                        int inputX = (x * stride + j) - padding;
                        int inputY = (y * stride + l) - padding;
                        if (0 <= inputX && inputX < inputWidth && 0 <= inputY && inputY < inputHeight)
                        {
                            for (int n = 0; n < inputDepth; n++)
                            {
                                outputValue += kernels[w, n * kernelSide * kernelSide + j * kernelSide + l] * input[n * inputWidth * inputHeight + inputX * inputHeight + inputY];
                            }
                        }
                    }
                }
                output[w * outputWidth * outputHeight + x * outputHeight + y] = function(outputValue);
            });//, new LaunchParam(new dim3((outputDepth + 7) / 8, (outputWidth + 7) / 8, (outputHeight + 7) / 8), new dim3(8, 8, 8)));
            //for (int i = 0; i < output.Length; i++)
            //{
            //    if (output[i] > 10.0 || output[i] < -10.0)
            //    {

            //    }
            //}
            //for (int w = 0; w < outputDepth; w++)
            //{
            //    for (int x = 0; x < outputWidth; x++)
            //    {
            //        for (int y = 0; y < outputHeight; y++)
            //        {
            //            float outputValue = 0.0F;
            //            for (int j = 0; j < kernelSide; j++)
            //            {
            //                for (int l = 0; l < kernelSide; l++)
            //                {
            //                    int inputX = (x * stride + j) - padding;
            //                    int inputY = (y * stride + l) - padding;
            //                    if (0 <= inputX && inputX < inputWidth && 0 <= inputY && inputY < inputHeight)
            //                    {
            //                        for (int n = 0; n < inputDepth; n++)
            //                        {
            //                            outputValue += kernels[w, n * kernelSide * kernelSide + j * kernelSide + l] * input[n * inputWidth * inputHeight + inputX * inputHeight + inputY];
            //                        }
            //                    }
            //                }
            //            }
            //            output[w * outputWidth * outputHeight + x * outputHeight + y] = function(outputValue);
            //        }
            //    }
            //}
        }

        internal static void BackpropagateConvolution(float[] input, int inputDepth, int inputWidth, int inputHeight, float[] output, int outputDepth, int outputWidth, int outputHeight, float[,] kernels, int kernelSide, int stride, int padding, float[] outputError, int outputErrorDepth, int outputErrorWidth, int outputErrorHeight, float[] inputError, int inputErrorDepht, int inputErrorWidth, int inputErrorHeight, Func<float, float> functionDerivative, float[,] gradients, bool learning)
        {
            ////int kernelSideSquared = kernelSide * kernelSide;
            //int index = 0;
            //foreach (var item in inputError)
            //{
            //    inputError[index] = 1.0F;
            //        ++index;
            //}
            //for (int i = 0; i < inputDepth * inputWidth * inputHeight; i++)
            //{
            //    int w = i / (inputWidth * inputHeight);
            //    int x = (i % (inputWidth * inputHeight)) / inputHeight;
            //    int y = (i % (inputWidth * inputHeight)) % inputHeight;
            //    //if (w < inputDepth && x < inputWidth && y < inputHeight)
            //    {
            //        float inputErrorValue = 0.0F;
            //        for (int j = x % stride; j < kernelSide; j += stride)
            //        {
            //            for (int k = y % stride; k < kernelSide; k += stride)
            //            {
            //                int outputX = x - j + padding;
            //                int outputY = y - k + padding;
            //                if (0 <= outputX && outputX < outputWidth && 0 <= outputY && outputY < outputHeight)
            //                {
            //                    for (int l = 0; l < outputDepth; l++)
            //                    {
            //                        float error = outputError[l * outputErrorWidth * outputErrorHeight + outputX * outputErrorHeight + outputY] * functionDerivative(output[l * outputWidth * outputHeight + outputX * outputHeight + outputY]);
            //                        inputErrorValue += kernels[l, w * kernelSide * kernelSide + j * kernelSide + k] * error;
            //                        if (learning)
            //                        {
            //                            DeviceFunction.AtomicAdd(DeviceFunction.AddressOfArray(gradients) + l * outputDepth + w * kernelSide * kernelSide + j * kernelSide + k, input[w * inputWidth * inputHeight + x * inputHeight + y] * error);
            //                            /*float gradient = gradients[l, w * kernelSide * kernelSide + j * kernelSide + k];
            //                            gradients[l, w * kernelSide * kernelSide + j * kernelSide + k] = gradient + input[w * inputWidth * inputHeight + x * inputHeight + y] * error;*/
            //                        }
            //                    }
            //                }
            //            }
            //        }
            //    }
            //}
            Backbone.gpu.For(0, inputDepth * inputWidth * inputHeight, delegate (int i)
            {
                int w = i / (inputWidth * inputHeight);
                int x = (i % (inputWidth * inputHeight)) / inputHeight;
                int y = (i % (inputWidth * inputHeight)) % inputHeight;
                if (w < inputDepth && x < inputWidth && y < inputHeight)
                {
                    float inputErrorValue = 0.0F;
                    for (int j = x % stride; j < kernelSide; j += stride)
                    {
                        for (int k = y % stride; k < kernelSide; k += stride)
                        {
                            int outputX = x - j + padding;
                            int outputY = y - k + padding;
                            if (0 <= outputX && outputX < outputWidth && 0 <= outputY && outputY < outputHeight)
                            {
                                for (int l = 0; l < outputDepth; l++)
                                {
                                    float error = outputError[l * outputErrorWidth * outputErrorHeight + outputX * outputErrorHeight + outputY] * functionDerivative(output[l * outputWidth * outputHeight + outputX * outputHeight + outputY]);
                                    inputErrorValue += kernels[l, w * kernelSide * kernelSide + j * kernelSide + k] * error;
                                    if (learning)
                                    {
                                        DeviceFunction.AtomicAdd(DeviceFunction.AddressOfArray(gradients) + l * inputDepth * kernelSide * kernelSide + w * kernelSide * kernelSide + j * kernelSide + k, input[w * inputWidth * inputHeight + x * inputHeight + y] * error);
                                        /*float gradient = gradients[l, w * kernelSide * kernelSide + j * kernelSide + k];
                                        gradients[l, w * kernelSide * kernelSide + j * kernelSide + k] = gradient + input[w * inputWidth * inputHeight + x * inputHeight + y] * error;*/
                                    }
                                }
                            }
                        }
                    }
                    inputError[w * inputErrorWidth * inputErrorHeight + x * inputErrorHeight + y] = inputErrorValue;
                }
            }/*, new LaunchParam(new dim3((inputDepth + 3) / 4, (inputWidth + 3) / 4, (inputHeight + 3) / 4), new dim3(4, 4, 4))*/);
            //for (int w = 0; w < inputDepth; w++)
            //{
            //    for (int x = 0; x < inputWidth; x++)
            //    {
            //        for (int y = 0; y < inputHeight; y++)
            //        {
            //            float inputErrorValue = 0.0F;
            //            for (int j = x % stride; j < kernelSide; j += stride)
            //            {
            //                for (int k = y % stride; k < kernelSide; k += stride)
            //                {
            //                    int outputX = x - j + padding;
            //                    int outputY = y - k + padding;
            //                    if (0 <= outputX && outputX < outputWidth && 0 <= outputY && outputY < outputHeight)
            //                    {
            //                        for (int l = 0; l < outputDepth; l++)
            //                        {
            //                            float error = outputError[l * outputErrorWidth * outputErrorHeight + outputX * outputErrorHeight + outputY] * functionDerivative(output[l * outputWidth * outputHeight + outputX * outputHeight + outputY]);
            //                            inputErrorValue += kernels[l, w * kernelSide * kernelSide + j * kernelSide + k] * error;
            //                            if (learning)
            //                            {
            //                                float gradient = gradients[l, w * kernelSide * kernelSide + j * kernelSide + k];
            //                                gradients[l, w * kernelSide * kernelSide + j * kernelSide + k] = gradient + input[w * inputWidth * inputHeight + x * inputHeight + y] * error;
            //                            }
            //                        }
            //                    }
            //                }
            //            }
            //            inputError[w * inputErrorWidth * inputErrorHeight + x * inputErrorHeight + y] = inputErrorValue;
            //        }
            //    }
            //}
        }

        internal static void RandomizeMatrix(float[,] matrix, int width, int height, double variance)
        {
            float[] cop = new float[width * height];
            for (int i = 0; i < width * height; i++)
            {
                cop[i] = (float)RandomGenerator.GetNormalNumber(variance);
            }
            Backbone.gpu.For(0, width * height, delegate (int i)
            {
                int x = i / height;
                int y = i % height;
                matrix[x, y] = cop[i];
            });
        }

        internal static void RandomizeMatrix(double[,] matrix, int width, int height, double variance)
        {
            double[] cop = new double[width * height];
            for (int i = 0; i < width * height; i++)
            {
                cop[i] = RandomGenerator.GetNormalNumber(variance);
            }
            Backbone.gpu.For(0, width * height, delegate (int i)
            {
                int x = i / height;
                int y = i % height;
                matrix[x, y] = cop[i];
            });
        }

        internal static void UpdateConvolution(float[,] kernels, float[,] gradients, float[,] oldUpdates, int inputDepth, int inputWidth, int inputHeight, int outputDepth, int outputWidth, int outputHeight, int kernelSide, float rate, float momentum)
        {
            Backbone.gpu.For(0, outputDepth * inputDepth * kernelSide * kernelSide, delegate (int i)
            {
                int x = i % outputDepth;
                int y = i / outputDepth;
                float update = gradients[x, y] * rate + oldUpdates[x, y] * momentum;
                float kernel = kernels[x, y] + update;
                kernels[x, y] = kernel;
                oldUpdates[x, y] = update;
                gradients[x, y] = 0.0F;
            });
            /*Backbone.gpu.For(0, outputDepth, delegate (int i)
            {
                for (int j = 0; j < inputDepth * kernelSide * kernelSide; j++)
                {
                    float update = gradients[i, j] * rate + oldUpdates[i, j] * momentum;
                    float kernel = kernels[i, j] + update;
                    kernels[i, j] = kernel;
                    oldUpdates[i, j] = update;
                    gradients[i, j] = 0.0F;
                }
            });*/
        }

        internal static void ApplyImageDropout(float[] input, float[] output, int depth, int width, int height, bool[] dropped, double dropChance, bool learning)
        {
            double[] cop = new double[depth * width * height];
            for (int i = 0; i < depth * width * height; i++)
            {
                cop[i] = RandomGenerator.GetDouble();
            }
            if (learning)
            {
                Backbone.gpu.For(0, depth * width * height, delegate (int i)
                {
                    dropped[i] = cop[i] < dropChance;
                    if (dropped[i])
                    {
                        output[i] = 0.0F;
                    }
                    else
                    {
                        output[i] = input[i];
                    }
                });
            }
            else
            {
                Backbone.gpu.For(0, depth * width * height, delegate (int i)
                {
                    int w = i / (width * height);
                    int x = (i % (width * height)) / height;
                    int y = (i % (width * height)) % height;
                    output[w * width * height + x * height + y] = input[w * width * height + x * height + y] * (float)(1.0 - dropChance);
                });
            }
        }

        internal static void BackpropagateImageDropout(float[] input, float[] output, int depth, int width, int height, bool[] dropped, double dropChance, bool learning, float[] outputError, int outputErrorDepth, int outputErrorWidth, int outputErrorHeight, float[] inputError, int inputErrorDepth, int inputErrorWidth, int inputErrorHeight)
        {
            if (learning)
            {
                Backbone.gpu.For(0, depth * width * height, delegate (int i)
                {
                    int w = i / (width * height);
                    int x = (i % (width * height)) / height;
                    int y = (i % (width * height)) % height;
                    if (dropped[i])
                    {
                        inputError[i] = 0.0F;
                    }
                    else
                    {
                        inputError[i] = outputError[i];
                    }
                });
            }
            else
            {
                Backbone.gpu.For(0, depth * width * height, delegate (int i)
                {
                    int w = i / (width * height);
                    int x = (i / (width * height) * (width * height)) / height;
                    int y = (i / (width * height) * (width * height)) % height;
                    inputError[w * width * height + x * height + y] = outputError[w * width * height + x * height + y] * (float)(1.0 - dropChance);
                });
            }
        }
        #endregion
#endif
    }
}
