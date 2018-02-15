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
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    /// <summary>Represents a neuron.</summary>
    public class FreeNeuron : ILayer
    {
        private double input;
        private double output;

        /// <summary>Creates a new instance of the <code>FreeNeuron</code> class.</summary>
        public FreeNeuron() { }

        /// <summary>The latest output of this neuron.</summary>
        public virtual double LastOutput
        {
            get { return this.output; }
        }

        /// <summary>Always <code>1</code>.</summary>
        public int Length
        {
            get { return 1; }
        }

        /// <summary>Returns the value of the activation function of this neuron for the given value.</summary>
        /// <param name="input">The input value to be given to the activation function.</param>
        /// <returns>The value of the activation function for the given input.</returns>
        protected virtual double Activation(double input)
        {
            return 1.0 / (1.0 + Math.Exp(-input));
        }

        /// <summary>Returns the derivative of the activation function for the given input and output.</summary>
        /// <param name="input">The input value of the activation function.</param>
        /// <param name="output">The output value of the activation function.</param>
        /// <returns>The derivative of the activation function.</returns>
        protected virtual double ActivationDerivative(double input, double output)
        {
            return output * (1.0 - output);
        }

        /// <summary>Feeds the given input trough this neuron.</summary>
        /// <param name="input">The input to be fed.</param>
        /// <returns>The output of the neuron.</returns>
        public virtual double Feed(double input)
        {
            return this.output = this.Activation(input);
        }

        /// <summary>Backpropagates the given error, updating it using the derivative of the activation function.</summary>
        /// <param name="error">The error to be backpropagated and updated.</param>
        /// <param name="skip">The amount of positions to be skipped in the error array.</param>
        public void BackPropagate(double[] error, int skip = 0)
        {
            error[skip] = error[skip] * this.ActivationDerivative(this.input, this.LastOutput);
        }

        /// <summary>Feeds the given input trough this neuron.</summary>
        /// <param name="index">Must be <code>0</code>.</param>
        /// <param name="input">The output of the neuron.</param>
        public void Feed(int index, double input)
        {
            this.input = input;
            this.output = this.Activation(input);
        }

        /// <summary>Returns the latest output of this neuron.</summary>
        /// <param name="index">Must be <code>0</code>.</param>
        /// <returns>The latest output.</returns>
        public double GetLastOutput(int index)
        {
            return this.LastOutput;
        }

        /// <summary>Gets the latest output of this neuron.</summary>
        /// <param name="output">The array to be written the latest output into.</param>
        /// <param name="skip">The amount of positions to be skipped in the output array.</param>
        public void GetLastOutput(double[] output, int skip = 0)
        {
            output[skip] = this.LastOutput;
        }

        /// <summary>Returns the latest input of this neuron.</summary>
        /// <param name="index">Must be <code>0</code>.</param>
        /// <returns>The latest input fed to this neuron.</returns>
        public double GetLastInput(int index)
        {
            return this.input;
        }

        /// <summary>Gets the latest input of this neuron.</summary>
        /// <param name="input">The array to be written the latest input into.</param>
        /// <param name="skip">The amount of positions to be skipped in the input array.</param>
        public void GetLastInput(double[] input, int skip = 0)
        {
            input[skip] = this.input;
        }

        /// <summary>Feeds the given input trough this neuron.</summary>
        /// <param name="input">The input to be fed.</param>
        /// <param name="skip">The amount of positions to be skipped in the input array.</param>
        public void Feed(double[] input, int skip = 0)
        {
            this.output = input[skip] = this.Activation(this.input);
        }

        /// <summary>Creates a copy of this instance of the <code>FreeNeuron</code> class.</summary>
        /// <returns>The generated instance of the <code>FreeNeuron</code> class.</returns>
        public virtual object Clone()
        {
            return new FreeNeuron();
        }

        /// <summary>Unused.</summary>
        public void FeedEnd() { }
    }
}
