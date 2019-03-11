/*
    (C) 2019 Valentino Giudice

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

namespace NeuralSharp
{
    /// <summary>Represents a string of neurons.</summary>
    public class NeuronsString : IArraysLayer
    {
        private double[] input;
        private double[] output;
        private int inputSkip;
        private int outputSkip;
        private int length;

        /// <summary>Either creates a siamese of the given <code>NeuronsString</code> instance or clones it.</summary>
        /// <param name="original">The original instance to be creted a siamese of or cloned.</param>
        /// <param name="siamese"><code>true</code> if a siamese is to be created, <code>false</code> if a clone is.</param>
        protected NeuronsString(NeuronsString original, bool siamese)
        {
            this.length = original.Length;
        }

        /// <summary>Creates an instance of the <code>NeuronsString</code> class.</summary>
        /// <param name="length">The lenght of the layer.</param>
        /// <param name="createIO">Whether the input array and the output array of the layer are to be created.</param>
        public NeuronsString(int length, bool createIO = false)
        {
            if (createIO)
            {
                this.input = Backbone.CreateArray<double>(length);
                this.output = Backbone.CreateArray<double>(length);
                this.inputSkip = 0;
                this.outputSkip = 0;
            }
            this.length = length;
        }
        
        /// <summary>The input array.</summary>
        public double[] Input
        {
            get { return this.input; }
        }

        /// <summary>The output array.</summary>
        public double[] Output
        {
            get { return this.output; }
        }

        /// <summary>The index of the first used entry of the input array.</summary>
        public int InputSkip
        {
            get { return this.inputSkip; }
        }

        /// <summary>The index of the first used entry of the output array.</summary>
        public int OutputSkip
        {
            get { return this.outputSkip; }
        }

        /// <summary>The size of the input.</summary>
        public int InputSize
        {
            get { return this.length; }
        }

        /// <summary>The size of the output.</summary>
        public int OutputSize
        {
            get { return this.length; }
        }

        /// <summary>The size of the layer.</summary>
        public int Length
        {
            get { return this.length; }
        }

        /// <summary>The amount of parameters of the layer. Always <code>0</code>.</summary>
        public int Parameters
        {
            get { return 0; }
        }

        /// <summary>The activation function of the layer.</summary>
        /// <param name="input">The input.</param>
        /// <returns>The output.</returns>
        protected virtual double Activation(double input)
        {
            return 1.0 / (1.0 + Math.Exp(-input));
        }

        /// <summary>The derivative of the activation function of the layer.</summary>
        /// <param name="input">The input of the layer.</param>
        /// <param name="output">The output.</param>
        /// <returns>The derivative.</returns>
        protected virtual double ActivationDerivative(double input, double output)
        {
            return output * (1.0 - output);
        }

        /// <summary>Feeds the layer forward.</summary>
        /// <param name="learning">Whether the layer is being used in a training session.</param>
        public virtual void Feed(bool learning = false)
        {
            Backbone.ApplyNeuronsString(this.input, this.inputSkip, this.output, this.outputSkip, this.length, this.Activation);
        }

        /// <summary>Backpropagates the layer.</summary>
        /// <param name="outputErrorArray">The output error of the layer.</param>
        /// <param name="outputErrorSkip">The index of the first entry of the output error to be used.</param>
        /// <param name="inputErrorArray">The array to be written the input error into.</param>
        /// <param name="inputErrorSkip">The index of the first entry of the input layer to be used.</param>
        /// <param name="learning">Whether the layer is being used in a learning session.</param>
        public virtual void BackPropagate(double[] outputErrorArray, int outputErrorSkip, double[] inputErrorArray, int inputErrorSkip, bool learning)
        {
            Backbone.BackpropagateNeuronsString(this.input, this.inputSkip, this.output, this.outputSkip, this.length, outputErrorArray, outputErrorSkip, inputErrorArray, inputErrorSkip, this.ActivationDerivative, learning);
        }

        /// <summary>Backpropagates an error trough the layer.</summary>
        /// <param name="outputError">The error to be backpropagated.</param>
        /// <param name="inputError">The array to be written the input error into.</param>
        /// <param name="learning">Whether the layer is being used in a learning session.</param>
        public void BackPropagate(double[] outputError, double[] inputError, bool learning)
        {
            this.BackPropagate(outputError, 0, inputError, 0, learning);
        }

        /// <summary>Sets the input array and the output array of the layer.</summary>
        /// <param name="inputArray">The input array to be set.</param>
        /// <param name="inputSkip">The index of the first entry of the input array to be used.</param>
        /// <param name="outputArray">The output array to be set.</param>
        /// <param name="outputSkip">The index of the first entry of the output array to be used.</param>
        public void SetInputAndOutput(double[] inputArray, int inputSkip, double[] outputArray, int outputSkip)
        {
            this.input = inputArray;
            this.inputSkip = inputSkip;
            this.output = outputArray;
            this.outputSkip = outputSkip;
        }

        /// <summary>Sets the input array and the output array of the layer.</summary>
        /// <param name="input">The input array to be used.</param>
        /// <param name="output">The output array to be used.</param>
        public void SetInputAndOutput(double[] input, double[] output)
        {
            this.SetInputAndOutput(input, 0, output, 0);
        }

        /// <summary>Sets the input array and creates and sets the output array.</summary>
        /// <param name="inputArray">The input array to be set.</param>
        /// <param name="inputSkip">The index of the first entry of the input array to be used.</param>
        /// <returns>The created output array.</returns>
        public double[] SetInputGetOutput(double[] inputArray, int inputSkip)
        {
            this.input = inputArray;
            this.inputSkip = inputSkip;
            this.outputSkip = 0;
            return this.output = Backbone.CreateArray<double>(this.Length);
        }

        /// <summary>Sets the input array and creates and sets the output array.</summary>
        /// <param name="input">The input array to be set.</param>
        /// <returns>The created output array.</returns>
        public double[] SetInputGetOutput(double[] input)
        {
            return this.SetInputGetOutput(input, 0);
        }

        /// <summary>Updates the weights of the layer.</summary>
        /// <param name="rate">The learning rate to be used.</param>
        /// <param name="momentum">The momentum to be used.</param>
        public void UpdateWeights(double rate, double momentum = 0.0) { }
        
        /// <summary>Creates a siamese of the layer.</summary>
        /// <returns>The siamese.</returns>
        public virtual IUntypedLayer CreateSiamese()
        {
            return new NeuronsString(this, true);
        }

        /// <summary>Creates a clone of the layer.</summary>
        /// <returns>The clone.</returns>
        public virtual IUntypedLayer Clone()
        {
            return new NeuronsString(this, false);
        }
    }
}
