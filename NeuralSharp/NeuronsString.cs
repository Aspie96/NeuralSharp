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
using System.Runtime.Serialization;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    /// <summary>Represents a layer in a neural network.</summary>
    [DataContract]
    public class NeuronsString : ILayer
    {
        private double[] inputs;
        private double[] outputs;

        /// <summary>Creates a new instance of the <code>NeuronsString</code> class.</summary>
        /// <param name="length">The lenght of the layer.</param>
        public NeuronsString(int length)
        {
            this.inputs = new double[length];
            this.outputs = new double[length];
        }

        /// <summary>The length of the layer.</summary>
        [DataMember]
        public int Length
        {
            get { return this.inputs.Length; }
            private set
            {
                this.inputs = new double[value];
                this.outputs = new double[value];
            }
        }

        /// <summary>The lates inputs fed to the layer.</summary>
        protected double[] Inputs
        {
            get { return this.inputs; }
        }

        /// <summary>The latest outputs from the layer.</summary>
        protected double[] Outputs
        {
            get { return this.outputs; }
        }

        /// <summary>Returns the value of the activation function for the given input value.</summary>
        /// <param name="input">The input to be given to the activation function.</param>
        /// <returns>The output of the activation function.</returns>
        protected virtual double Activation(double input)
        {
            return 1.0 / (1.0 + Math.Exp(-input));
        }

        /// <summary>Returns the derivative of the activation function for the given input and output value.</summary>
        /// <param name="input">The input value.</param>
        /// <param name="output">The output value.</param>
        /// <returns>The derviative of the activation function. Always <code>0</code>.</returns>
        protected virtual double ActivationDerivative(double input, double output)
        {
            return output * (1.0 - output);
        }

        /// <summary>Feeds a value to a neuron of the layer.</summary>
        /// <param name="index">The index of the neuron to be fed.</param>
        /// <param name="input">The value to be fed.</param>
        public virtual void Feed(int index, double input)
        {
            this.Inputs[index] = input;
            this.Outputs[index] = this.Activation(input);
        }

        /// <summary>Backpropagates an error trough this layer, updating it accoring to the derivatives of the neurons.</summary>
        /// <param name="error">The eror to be backpropagated.</param>
        /// <param name="skip">The amount of positions to skip in the error array.</param>
        public virtual void BackPropagate(double[] error, int skip = 0)
        {
            for (int i = 0; i < this.Length; i++)
            {
                error[i + skip] = error[i + skip] * this.ActivationDerivative(this.Inputs[i], this.Outputs[i]);
            }
        }

        /// <summary>Returns the latest output of a neuron of this layer.</summary>
        /// <param name="index">The index of the neuron.</param>
        /// <returns>The latest output of the chosen neuron.</returns>
        public double GetLastOutput(int index)
        {
            return this.Outputs[index];
        }

        /// <summary>Gets the latest outputs of the neurons of this layer.</summary>
        /// <param name="output">The array to be written the output into.</param>
        /// <param name="skip">The amount of positions to skip in the output array.</param>
        public void GetLastOutput(double[] output, int skip = 0)
        {
            Array.Copy(this.Outputs, 0, output, skip, this.Length);
        }

        /// <summary>Returns the latest input of a neuron of this layer.</summary>
        /// <param name="index">The index of the neuron.</param>
        /// <returns>The latest input fed to the chosen neuron.</returns>
        public double GetLastInput(int index)
        {
            return this.Inputs[index];
        }

        /// <summary>Gets the latest inputs fed to this layer.</summary>
        /// <param name="input">The array to be written the input into.</param>
        /// <param name="skip">The amount of positions to be skipped in the input array.</param>
        public void GetLastInput(double[] input, int skip = 0)
        {
            Array.Copy(this.Inputs, 0, input, skip, this.Length);
        }

        /// <summary>Feeds the given input trough the layer.</summary>
        /// <param name="input">The array to be fed.</param>
        /// <param name="skip">The amount of positions to be skipped in the input array.</param>
        public virtual void Feed(double[] input, int skip = 0)
        {
            for (int i = 0; i < this.Length; i++)
            {
                this.Inputs[i] = input[i + skip];
                this.Outputs[i] = this.Activation(this.Inputs[i]);
            }
            this.FeedEnd();
        }

        /// <summary>Creates a copy of this instance of the <code>NeuronsString</code> class.</summary>
        /// <returns>The generated instance of the <code>NeuronsString</code> class.</returns>
        public virtual object Clone()
        {
            return new NeuronsString(this.Length);
        }

        /// <summary>Method to be called after every neuron of the layer has been feed.</summary>
        public virtual void FeedEnd() { }
    }
}
