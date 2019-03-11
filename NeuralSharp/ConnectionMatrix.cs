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
using System.Linq;
using System.Runtime.Serialization;
using System.Text;
using System.Threading.Tasks;

namespace NeuralSharp
{
    /// <summary>Represents a connection matrix.</summary>
    public class ConnectionMatrix : IArraysLayer
    {
        private double[] input;
        private double[] output;
        private int inputSkip;
        private int outputSkip;
        private int inputSize;
        private int outputSize;
        private double[,] weights;
        private double[,] gradients;
        private double[,] momentum;

        /// <summary>Either creates a siamese of the given <code>ConnectionMatrix</code> instance or clones it.</summary>
        /// <param name="original">The original instance to be created a siamese of or cloned.</param>
        /// <param name="siamese"><code>true</code> if a siamese is to be created, <code>false</code> if a clone is.</param>
        protected ConnectionMatrix(ConnectionMatrix original, bool siamese)
        {
            this.inputSize = original.InputSize;
            this.outputSize = original.OutputSize;
            if (siamese)
            {
                this.weights = original.Weights;
                this.gradients = original.Gradients;
                this.momentum = original.Momentum;
            }
            else
            {
                this.weights = Backbone.CreateArray<double>(original.InputSize, original.OutputSize);
                Backbone.RandomizeMatrix(this.weights, original.InputSize, original.OutputSize, 2.0 / (original.InputSize + original.OutputSize));
                this.gradients = Backbone.CreateArray<double>(original.InputSize, original.OutputSize);
                this.momentum = Backbone.CreateArray<double>(original.InputSize, original.OutputSize);
            }
        }

        /// <summary>Creates an instance of the <code>ConnectionMatrix</code> class.</summary>
        /// <param name="inputSize">The size of the input of the layer.</param>
        /// <param name="outputSize">The size of the output of the layer.</param>
        /// <param name="createIO">Whether the input array and the output array of the layer are to be created.</param>
        public ConnectionMatrix(int inputSize, int outputSize, bool createIO = false)
        {
            if (createIO)
            {
                this.input = Backbone.CreateArray<double>(inputSize);
                this.output = Backbone.CreateArray<double>(outputSize);
                this.inputSkip = 0;
                this.outputSkip = 0;
            }
            this.inputSize = inputSize;
            this.outputSize = outputSize;
            this.weights = Backbone.CreateArray<double>(inputSize, outputSize);
            Backbone.RandomizeMatrix(this.weights, inputSize, outputSize, 2.0 / (inputSize + outputSize));
            this.gradients = Backbone.CreateArray<double>(inputSize, outputSize);
            this.momentum = Backbone.CreateArray<double>(inputSize, outputSize);
        }
        
        /// <summary>The input array of the layer.</summary>
        public double[] Input
        {
            get { return this.input; }
        }
        
        /// <summary>The output array of the layer.</summary>
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

        /// <summary>The size of the output of the layer.</summary>
        public int InputSize
        {
            get { return this.inputSize; }
        }

        /// <summary>The size of the output of the layer.</summary>
        public int OutputSize
        {
            get { return this.outputSize; }
        }

        /// <summary>The weights of the layer.</summary>
        protected double[,] Weights
        {
            get { return this.weights; }
        }

        /// <summary>The gradients of the layer.</summary>
        protected double[,] Gradients
        {
            get { return this.gradients; }
        }

        /// <summary>The previous updates of the layer.</summary>
        protected double[,] Momentum
        {
            get { return this.momentum; }
        }
        
        /// <summary>The amount of parameters of the layer.</summary>
        public virtual int Parameters
        {
            get { return this.weights.Length; }
        }
        
        /// <summary>Feeds the layer forward.</summary>
        /// <param name="learning">Whether the layer is being used in a training session. Unused.</param>
        public virtual void Feed(bool learning = false)
        {
            Backbone.ApplyConnectionMatrix(this.Input, this.InputSkip, this.InputSize, this.Output, this.OutputSkip, this.OutputSize, this.Weights);
        }
        
        /// <summary>Backpropagates the given error trough the layer.</summary>
        /// <param name="outputErrorArray">The output error to be backpropagated.</param>
        /// <param name="outputErrorSkip">The index of the first entry of the ouptut error array to be used.</param>
        /// <param name="inputErrorArray">The array to be written the input entry into.</param>
        /// <param name="inputErrorSkip">The index of the first entry of the input error array to be used.</param>
        /// <param name="learning">Whether the array is being used in a training session.</param>
        public virtual void BackPropagate(double[] outputErrorArray, int outputErrorSkip, double[] inputErrorArray, int inputErrorSkip, bool learning)
        {
            Backbone.BackpropagateConnectionMatrix(this.Input, this.InputSkip, this.InputSize, this.Output, this.OutputSkip, this.OutputSize, this.Weights, outputErrorArray, outputErrorSkip, inputErrorArray, inputErrorSkip, this.Gradients, learning);
        }

        /// <summary>Backpropagates the given error trough the layer.</summary>
        /// <param name="outputError">The output error to be backpropagated.</param>
        /// <param name="inputError">The array to be written the input error into.</param>
        /// <param name="learning">Whether the array is being used in a training session.</param>
        public void BackPropagate(double[] outputError, double[] inputError, bool learning)
        {
            this.BackPropagate(outputError, 0, inputError, 0, learning);
        }

        /// <summary>Sets the input array and the output array of this layer.</summary>
        /// <param name="inputArray">The input array to be set.</param>
        /// <param name="inputSkip">The index of the first entry of the input array to be used.</param>
        /// <param name="outputArray">The output array to be set.</param>
        /// <param name="outputSkip">The index of the first entry of the output array to be used.</param>
        public void SetInputAndOutput(double[] inputArray, int inputSkip, double[] outputArray, int outputSkip)
        {
            this.input = inputArray;
            this.output = outputArray;
            this.inputSkip = inputSkip;
            this.outputSkip = outputSkip;
        }

        /// <summary>Sets the input array and the ouptut array of this layer.</summary>
        /// <param name="input">The input array to be set.</param>
        /// <param name="output">The output array to be set.</param>
        public void SetInputAndOutput(double[] input, double[] output)
        {
            this.SetInputAndOutput(input, 0, output, 0);
        }

        /// <summary>Sets the input array of this layer and creates and sets an output array.</summary>
        /// <param name="inputArray">The input array to be set.</param>
        /// <param name="inputSkip">The index of the first entry of the input array to be used.</param>
        /// <returns>The created output array.</returns>
        public double[] SetInputGetOutput(double[] inputArray, int inputSkip)
        {
            this.input = inputArray;
            this.inputSkip = inputSkip;
            this.outputSkip = 0;
            return this.output = Backbone.CreateArray<double>(this.OutputSize);
        }

        /// <summary>Sets the input array of this layer and creates and sets an output array.</summary>
        /// <param name="input">The input array to be set.</param>
        /// <returns>The created outptu array.</returns>
        public double[] SetInputGetOutput(double[] input)
        {
            return this.SetInputGetOutput(input, 0);
        }
        
        /// <summary>Updates the weights of the layer.</summary>
        /// <param name="rate">The learning rate to be used.</param>
        /// <param name="momentum">The momentum to be used.</param>
        public virtual void UpdateWeights(double rate, double momentum = 0.0)
        {
            Backbone.UpdateConnectionMatrix(this.Weights, this.Gradients, this.Momentum, this.InputSize, this.OutputSize, rate, momentum);
        }
        
        /// <summary>Creates a siamese of the layer.</summary>
        /// <returns>The created instance of the <code>ConnectionMatrix</code> class.</returns>
        public virtual IUntypedLayer CreateSiamese()
        {
            return new ConnectionMatrix(this, true);
        }

        /// <summary>Creates a clone of the layer.</summary>
        /// <returns>The created instance of the <code>ConnectionMatrix</code> class.</returns>
        public virtual IUntypedLayer Clone()
        {
            return new ConnectionMatrix(this, false);
        }
    }
}
