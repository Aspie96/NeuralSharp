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
        private float[] input;
        private float[] output;
        private int inputSkip;
        private int outputSkip;
        private int inputSize;
        private int outputSize;
        private float[,] weights;
        private float[,] gradients;
        private float[,] momentum;
        private object siameseID;

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
                this.siameseID = original.SiameseID;
            }
            else
            {
                this.weights = Backbone.CreateArray<float>(original.InputSize, original.OutputSize);
                Backbone.RandomizeMatrix(this.weights, original.InputSize, original.OutputSize, 2.0F / (original.InputSize + original.OutputSize));
                this.gradients = Backbone.CreateArray<float>(original.InputSize, original.OutputSize);
                this.momentum = Backbone.CreateArray<float>(original.InputSize, original.OutputSize);
                this.siameseID = new object();
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
                this.input = Backbone.CreateArray<float>(inputSize);
                this.output = Backbone.CreateArray<float>(outputSize);
                this.inputSkip = 0;
                this.outputSkip = 0;
            }
            this.inputSize = inputSize;
            this.outputSize = outputSize;
            this.weights = Backbone.CreateArray<float>(inputSize, outputSize);
            Backbone.RandomizeMatrix(this.weights, inputSize, outputSize, 2.0F / (inputSize + outputSize));
            this.gradients = Backbone.CreateArray<float>(inputSize, outputSize);
            this.momentum = Backbone.CreateArray<float>(inputSize, outputSize);
            this.siameseID = new object();
        }
        
        /// <summary>The input array of the layer.</summary>
        public float[] Input
        {
            get { return this.input; }
        }
        
        /// <summary>The output array of the layer.</summary>
        public float[] Output
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
        protected float[,] Weights
        {
            get { return this.weights; }
        }

        /// <summary>The gradients of the layer.</summary>
        protected float[,] Gradients
        {
            get { return this.gradients; }
        }

        /// <summary>The previous updates of the layer.</summary>
        protected float[,] Momentum
        {
            get { return this.momentum; }
        }
        
        /// <summary>The amount of parameters of the layer.</summary>
        public virtual int Parameters
        {
            get { return this.weights.Length; }
        }

        /// <summary>The siamese identifier of the layer.</summary>
        public object SiameseID
        {
            get { return this.siameseID; }
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
        public virtual void BackPropagate(float[] outputErrorArray, int outputErrorSkip, float[] inputErrorArray, int inputErrorSkip, bool learning)
        {
            Backbone.BackpropagateConnectionMatrix(this.Input, this.InputSkip, this.InputSize, this.Output, this.OutputSkip, this.OutputSize, this.Weights, outputErrorArray, outputErrorSkip, inputErrorArray, inputErrorSkip, this.Gradients, learning);
        }

        /// <summary>Backpropagates the given error trough the layer.</summary>
        /// <param name="outputError">The output error to be backpropagated.</param>
        /// <param name="inputError">The array to be written the input error into.</param>
        /// <param name="learning">Whether the array is being used in a training session.</param>
        public void BackPropagate(float[] outputError, float[] inputError, bool learning)
        {
            this.BackPropagate(outputError, 0, inputError, 0, learning);
        }

        /// <summary>Sets the input array and the output array of this layer.</summary>
        /// <param name="inputArray">The input array to be set.</param>
        /// <param name="inputSkip">The index of the first entry of the input array to be used.</param>
        /// <param name="outputArray">The output array to be set.</param>
        /// <param name="outputSkip">The index of the first entry of the output array to be used.</param>
        public void SetInputAndOutput(float[] inputArray, int inputSkip, float[] outputArray, int outputSkip)
        {
            this.input = inputArray;
            this.output = outputArray;
            this.inputSkip = inputSkip;
            this.outputSkip = outputSkip;
        }

        /// <summary>Sets the input array and the ouptut array of this layer.</summary>
        /// <param name="input">The input array to be set.</param>
        /// <param name="output">The output array to be set.</param>
        public void SetInputAndOutput(float[] input, float[] output)
        {
            this.SetInputAndOutput(input, 0, output, 0);
        }

        /// <summary>Sets the input array of this layer and creates and sets an output array.</summary>
        /// <param name="inputArray">The input array to be set.</param>
        /// <param name="inputSkip">The index of the first entry of the input array to be used.</param>
        /// <returns>The created output array.</returns>
        public float[] SetInputGetOutput(float[] inputArray, int inputSkip)
        {
            this.input = inputArray;
            this.inputSkip = inputSkip;
            this.outputSkip = 0;
            return this.output = Backbone.CreateArray<float>(this.OutputSize);
        }

        /// <summary>Sets the input array of this layer and creates and sets an output array.</summary>
        /// <param name="input">The input array to be set.</param>
        /// <returns>The created outptu array.</returns>
        public float[] SetInputGetOutput(float[] input)
        {
            return this.SetInputGetOutput(input, 0);
        }
        
        /// <summary>Updates the weights of the layer.</summary>
        /// <param name="rate">The learning rate to be used.</param>
        /// <param name="momentum">The momentum to be used.</param>
        public virtual void UpdateWeights(float rate, float momentum = 0.0F)
        {
            Backbone.UpdateConnectionMatrix(this.Weights, this.Gradients, this.Momentum, this.InputSize, this.OutputSize, rate, momentum);
        }
        
        /// <summary>Creates a siamese of the layer.</summary>
        /// <returns>The created instance of the <code>ConnectionMatrix</code> class.</returns>
        public virtual ILayer<float[], float[]> CreateSiamese()
        {
            return new ConnectionMatrix(this, true);
        }

        /// <summary>Creates a clone of the layer.</summary>
        /// <returns>The created instance of the <code>ConnectionMatrix</code> class.</returns>
        public virtual ILayer<float[], float[]> Clone()
        {
            return new ConnectionMatrix(this, false);
        }

        /// <summary>Counts the amount of parameters of the layer.</summary>
        /// <param name="siameseIDs">The siamese identifiers to be excluded. The siamese identifiers of the layer will be added to the list.</param>
        /// <returns>The amount of parameters of the layer.</returns>
        public int CountParameters(List<object> siameseIDs)
        {
            if (siameseIDs.Contains(this.siameseID))
            {
                return 0;
            }
            siameseIDs.Add(this.siameseID);
            return this.weights.Length;
        }
    }
}
