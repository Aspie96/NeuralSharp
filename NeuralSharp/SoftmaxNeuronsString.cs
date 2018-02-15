/*
    (C) 2018 Valentino Giudice

    This software is provided 'as-is', without any express or implied
    warranty. In no event will the authors be held liable for any damages
    arising from the use of this software.

    Permission is granted to anyone to use this software for any purpose,
    including commercial applications, and to alter it and redistribute it
    freely, subject to the following restrictions:

    1. The origin of this software must not be misrepresented; you must not
       claim that you wrote the original software. If you use this software
       in a product, an acknowledgment in the product documentation would be
       appreciated but is not required.
    2. Altered source versions must be plainly marked as such, and must not be
       misrepresented as being the original software.
    3. This notice may not be removed or altered from any source distribution.
*/

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Serialization;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    /// <summary>Represents a layer used for classificaton.</summary>
    [DataContract]
    public class SoftmaxNeuronsString : NeuronsString
    {
        private double expSum;
        private double inputMax;
        private double inputMaxLog;
        private double[] errorCopy;

        /// <summary>Creates a new instance of the <code>SoftmaxNeuronsString</code> class.</summary>
        /// <param name="length">The lenght of the layer.</param>
        public SoftmaxNeuronsString(int length) : base(length)
        {
            this.inputMax = double.NegativeInfinity;
            this.errorCopy = new double[length];
        }

        /// <summary>Returns the value of the activation function for the given input value. For the softmax function, it depends on the current state of the layer.</summary>
        /// <param name="input">The input to be given to the activation function.</param>
        /// <returns>The output of the activation function. The same as the input.</returns>
        protected override double Activation(double input)
        {
            return Math.Exp(input - this.inputMax) / this.expSum;
        }
        
        /// <summary>Updates the given array applying to it the softmax function.</summary>
        /// <param name="array">The array to be updated.</param>
        public static void Softmax(double[] array)
        {
            SoftmaxNeuronsString.Softmax(array, array);
        }

        /// <summary>Applies the softmax function to the given array.</summary>
        /// <param name="array">The array to be applied the softmax function to.</param>
        /// <param name="output">The array to be written the value of the softmax function into.</param>
        public static void Softmax(double[] array, double[] output)
        {
            double expSum = 0.0;
            for (int i = 0; i < array.Length; i++)
            {
                expSum += output[i] = Math.Exp(array[i]);
            }
            for (int i = 0; i < array.Length; i++)
            {
                output[i] /= expSum;
            }
        }
        
        /// <summary>Computes the derivatives of the softmax function.</summary>
        /// <param name="array">The array to be computed the derivative of.</param>
        /// <param name="output">The matrix to be written the derivatives into.</param>
        public static void SoftmaxDerivative(double[] array, double[,] output)
        {
            for (int i = 0; i < array.Length; i++)
            {
                for (int j = 0; j < array.Length; j++)
                {
                    output[i, j] = -array[i] * array[j];
                }
                output[i, i] = array[i] * (1.0 - array[i]);
            }
        }

        /// <summary>Creates a copy of this instance of the <code>SoftmaxNeuronsString</code> class.</summary>
        /// <returns>The generated instance of the <code>SoftmaxNeuronsString</code> class.</returns>
        public override object Clone()
        {
            return new SoftmaxNeuronsString(this.Length);
        }

        /// <summary>Backpropagates an error trough this layer, updating it accoring to the derivatives of the neurons.</summary>
        /// <param name="error">The eror to be backpropagated.</param>
        /// <param name="skip">The amount of positions to skip in the error array.</param>
        public override void BackPropagate(double[] error, int skip = 0)
        {
            Array.Copy(error, skip, this.errorCopy, 0, this.Length);
            for (int i = 0; i < this.Length; i++)
            {
                error[i + skip] = 0;
                for (int j = 0; j < this.Length; j++)
                {
                    double derivative = 0;
                    if (i == j)
                    {
                        derivative += this.Outputs[j] * (1 - this.Outputs[j]);
                    }
                    else
                    {
                        derivative += -this.Outputs[i] * this.Outputs[j];
                    }
                    error[i + skip] += this.errorCopy[j] * derivative;
                }
            }
        }

        /// <summary>Method to be called after every neuron of the layer has been feed.</summary>
        public override void FeedEnd()
        {
            this.expSum = 0;
            this.inputMaxLog = Math.Log(this.inputMax);
            if (this.inputMax <= 0.5)
            {
                this.inputMaxLog = this.inputMax;
            }
            double sum = 0;
            for (int i = 0; i < this.Length; i++)
            {
                this.expSum += Math.Exp(this.Inputs[i] - this.inputMax);
                sum += Math.Abs(this.Inputs[i]);
            }
            for (int i = 0; i < this.Length; i++)
            {
                this.Outputs[i] = this.Activation(this.Inputs[i]);
            }
            this.inputMax = double.NegativeInfinity;
        }

        //// <summary>Feeds the given input trough the layer.</summary>
        /// <param name="input">The array to be fed.</param>
        /// <param name="skip">The amount of positions to be skipped in the input array.</param>
        public sealed override void Feed(double[] input, int skip = 0)
        {
            for (int i = 0; i < this.Length; i++)
            {
                this.Feed(i, input[i]);
            }
            this.FeedEnd();
        }

        /// <summary>Feeds a value to a neuron of the layer.</summary>
        /// <param name="index">The index of the neuron to be fed.</param>
        /// <param name="input">The value to be fed.</param>
        public override void Feed(int index, double input)
        {
            this.Inputs[index] = input;
            if (input > this.inputMax)
            {
                this.inputMax = input;
            }
        }

        [OnDeserialized]
        private void SetValuesOnDeserialized(StreamingContext context)
        {
            this.inputMax = double.NegativeInfinity;
            this.errorCopy = new double[this.Length];
        }
    }
}
