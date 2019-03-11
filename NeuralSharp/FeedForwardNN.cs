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
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Runtime.Serialization;
using System.Text;
using System.Threading.Tasks;

namespace NeuralSharp
{
    /// <summary>Represents a feed forward neural network.</summary>
    public class FeedForwardNN : Sequential<double[], double[]>, IArraysLayer
    {
        private double[] error1;
        private double[] error2;
        private IArraysLayer firstLayer;
        private IArraysLayer lastLayer;
        private bool layersConnected;

        /// <summary>Either creates a siamese of the given <code>FeedForwardNN</code> instance or clones it.</summary>
        /// <param name="original">The original instance to be created a siamese of or cloned.</param>
        /// <param name="siamese"><code>true</code> if a siamese is to be created, <code>false</code> if a clone is.</param>
        protected FeedForwardNN(FeedForwardNN original, bool siamese) : base(original, siamese)
        {
            int maxLength = Math.Max(original.Layers.Max(delegate (IUntypedLayer layer)
            {
                return ((IArraysLayer)layer).InputSize;
            }), original.lastLayer.OutputSize);
            this.error1 = Backbone.CreateArray<double>(maxLength);
            this.error2 = Backbone.CreateArray<double>(maxLength);
            this.layersConnected = false;
        }

        /// <summary>Creates an instance of the <code>FeedForwardNN</code> class.</summary>
        /// <param name="layers">The layers to be included in the network.</param>
        public FeedForwardNN(ICollection<IArraysLayer> layers) : base(layers.ToArray<IUntypedLayer>())
        {
            this.firstLayer = layers.First();
            this.lastLayer = layers.Last();
            int maxLength = Math.Max(layers.Max(delegate(IArraysLayer layer)
            {
                return layer.InputSize;
            }), this.lastLayer.OutputSize);
            this.error1 = Backbone.CreateArray<double>(maxLength);
            this.error2 = Backbone.CreateArray<double>(maxLength);
            this.layersConnected = false;
        }
        
        /// <summary>The index of the first used entry of the input array.</summary>
        public int InputSkip
        {
            get { return this.firstLayer.InputSkip; }
        }

        /// <summary>The index of the first used entry of the output array.</summary>
        public int OutputSkip
        {
            get { return this.lastLayer.OutputSkip; }
        }

        /// <summary>The lenght of the input of the network.</summary>
        public int InputSize
        {
            get { return this.firstLayer.InputSize; }
        }
        
        /// <summary>The length of the output of the network.</summary>
        public int OutputSize
        {
            get { return this.lastLayer.OutputSize; }
        }
        
        /// <summary>Creates an array which can be used as output error.</summary>
        /// <returns>The created array.</returns>
        protected override double[] NewError()
        {
            return Backbone.CreateArray<double>(this.OutputSize);
        }
        
        /// <summary>Feeds the network forward.</summary>
        /// <param name="input">The array to be copied the input from.</param>
        /// <param name="output">The array to be copied the output into.</param>
        /// <param name="learning">Whether the network is being used in a training session.</param>
        public override void Feed(double[] input, double[] output, bool learning = false)
        {
            this.Feed(input, 0, output, 0, learning);
        }

        /// <summary>Feeds the network forward.</summary>
        /// <param name="inputArray">The array to be copied the input from.</param>
        /// <param name="inputSkip">The index of the first entry of the given input array to be used.</param>
        /// <param name="outputArray">The array to be copied the output into.</param>
        /// <param name="outputSkip">The index of the first entry of the given output array to be used.</param>
        /// <param name="learning">Whether the network is being used in a training session.</param>
        public void Feed(double[] inputArray, int inputSkip, double[] outputArray, int outputSkip, bool learning = false)
        {
            Array.Copy(inputArray, inputSkip, this.Input, this.InputSkip, this.InputSize);
            this.Feed(learning);
            Array.Copy(this.Output, this.OutputSkip, outputArray, outputSkip, this.OutputSize);
        }
        
        /// <summary>Backpropagates the given error trough the network.</summary>
        /// <param name="outputErrorArray">The output error to be backpropagated.</param>
        /// <param name="outputErrorSkip">The index of the first entry of the output error array to be used.</param>
        /// <param name="learning">Whether the network is being used in a training session.</param>
        public void BackPropagate(double[] outputErrorArray, int outputErrorSkip, bool learning = true)
        {
            Backbone.CopyArray(outputErrorArray, outputErrorSkip, this.error2, 0, this.OutputSize);
            for (int i = this.Layers.Count - 1; i >= 0; i -= 1)
            {
                ((IArraysLayer)this.Layers[i]).BackPropagate(this.error2, this.error1, learning);
                double[] aux = this.error1;
                this.error1 = this.error2;
                this.error2 = aux;
            }
        }

        /// <summary>Backpropagates the given error trough the network.</summary>
        /// <param name="outputError">The output error to be backpropagated.</param>
        /// <param name="learning">Whether the network is being used in a training session.</param>
        public override void BackPropagate(double[] outputError, bool learning = true)
        {
            this.BackPropagate(outputError, 0, learning);
        }

        /// <summary>Backpropagates the given error trough the network.</summary>
        /// <param name="outputErrorArray">The output error to be backpropagated.</param>
        /// <param name="outputErrorSkip">The index of the first entry of the output error to be used.</param>
        /// <param name="inputErrorArray">The array to be written the input error into.</param>
        /// <param name="inputErrorSkip">The index of the first entry of the output error array to be used.</param>
        /// <param name="learning">Whether the network is being used in a training session.</param>
        public void BackPropagate(double[] outputErrorArray, int outputErrorSkip, double[] inputErrorArray, int inputErrorSkip, bool learning)
        {
            this.BackPropagate(outputErrorArray, outputErrorSkip, learning);
            Backbone.CopyArray(this.error2, 0, inputErrorArray, inputErrorSkip, this.InputSize);
        }

        /// <summary>Backpropagates the given error trough the network.</summary>
        /// <param name="outputError">The output error to be backpropagated.</param>
        /// <param name="inputError">The array to be written the input error into.</param>
        /// <param name="learning">Whether the network is being used in a training session.</param>
        public override void BackPropagate(double[] outputError, double[] inputError, bool learning)
        {
            this.BackPropagate(outputError, 0, inputError, 0, learning);
        }
        
        /// <summary>Updates the weights of the network.</summary>
        /// <param name="rate">The learning rate to be used.</param>
        /// <param name="momentum">The momentum to be used.</param>
        public override void UpdateWeights(double rate, double momentum = 0.0)
        {
            foreach (IArraysLayer layer in this.Layers)
            {
                layer.UpdateWeights(rate, momentum);
            }
        }

        /// <summary>Sets the input array and the output array of the network, connecting its inner layers.</summary>
        /// <param name="inputArray">The input array to be set.</param>
        /// <param name="inputSkip">The index of the first entry of the input array to be used.</param>
        /// <param name="outputArray">The output array to be set.</param>
        /// <param name="outputSkip">The index of the first entry of the output array to be used.</param>
        public void SetInputAndOutput(double[] inputArray, int inputSkip, double[] outputArray, int outputSkip)
        {
            if (this.layersConnected)
            {
                this.firstLayer.SetInputAndOutput(inputArray, inputSkip, this.firstLayer.Output, this.firstLayer.OutputSkip);
                this.lastLayer.SetInputAndOutput(this.lastLayer.Input, this.lastLayer.InputSkip, outputArray, outputSkip);
            }
            else
            {
                if (this.Layers.Count > 1)
                {
                    double[] array = inputArray;
                    array = this.firstLayer.SetInputGetOutput(array, inputSkip);
                    for (int i = 1; i < this.Layers.Count - 1; i++)
                    {
                        array = ((IArraysLayer)(this.Layers[i])).SetInputGetOutput(array);
                    }
                    this.lastLayer.SetInputAndOutput(array, 0, outputArray, outputSkip);
                }
                else
                {
                    this.firstLayer.SetInputAndOutput(inputArray, inputSkip, outputArray, outputSkip);
                }
                this.layersConnected = true;
            }
        }

        /// <summary>Sets the input array of the network and creates the output array.</summary>
        /// <param name="inputArray">The input array to be set.</param>
        /// <param name="inputSkip">The index of the first entry of the output array to be used.</param>
        /// <returns>The created output array.</returns>
        public double[] SetInputGetOutput(double[] inputArray, int inputSkip)
        {
            double[] retVal = Backbone.CreateArray<double>(this.OutputSize);
            this.SetInputAndOutput(inputArray, InputSkip, retVal, 0);
            return retVal;
        }
        
        /// <summary>Gets the output error of the network, given its actual and expected output.</summary>
        /// <param name="outputArray">The actual output of the network.</param>
        /// <param name="outputSkip">The index of the first entry of the given output array to be used.</param>
        /// <param name="expectedArray">The expected output of the network.</param>
        /// <param name="expectedSkip">The index of the first entry of the given output array to be used.</param>
        /// <param name="errorArray">The array to be written the output error into.</param>
        /// <param name="errorSkip">The index of the first entry of the error array to be used.</param>
        /// <returns>The output error of the network.</returns>
        public virtual double GetError(double[] outputArray, int outputSkip, double[] expectedArray, int expectedSkip, double[] errorArray, int errorSkip)
        {
            double retVal = 0;
            for (int i = 0; i < this.OutputSize; i++)
            {
                errorArray[errorSkip + i] = expectedArray[expectedSkip + i] - outputArray[outputSkip + i];
                retVal += errorArray[errorSkip + i] * errorArray[errorSkip + i];
            }
            return Math.Sqrt(retVal);
        }

        /// <summary>Gets the output error of the network, given its actual and expected output.</summary>
        /// <param name="output">The actual output of the network.</param>
        /// <param name="expectedOuptut">The expected output of the network.</param>
        /// <param name="error">The array to be written the output error into.</param>
        /// <returns>The output error of the network.</returns>
        public override double GetError(double[] output, double[] expectedOuptut, double[] error)
        {
            return this.GetError(output, 0, expectedOuptut, 0, error, 0);
        }

        /// <summary>Feeds the network forward and gets its error, given the expected output.</summary>
        /// <param name="inputArray">The array to be copied the input from.</param>
        /// <param name="inputSkip">The index of the first entry of the given input array to be used.</param>
        /// <param name="expectedArray">The expected output of the network.</param>
        /// <param name="expectedSkip">The index of the first entry of the expected array to be used.</param>
        /// <param name="errorArray">The array to be written the output error into.</param>
        /// <param name="errorSkip">The index of the first entry of the output array to be used.</param>
        /// <param name="learning">Whether the network is being used in a training session.</param>
        /// <returns>The output error of the network.</returns>
        public double FeedAndGetError(double[] inputArray, int inputSkip, double[] expectedArray, int expectedSkip, double[] errorArray, int errorSkip, bool learning)
        {
            Array.Copy(inputArray, inputSkip, this.Input, this.InputSkip, this.InputSize);
            this.Feed(learning);
            return this.GetError(this.Output, this.OutputSkip, expectedArray, expectedSkip, errorArray, errorSkip);
        }

        /// <summary>Feeds the network forward and gets its error, given the expected output.</summary>
        /// <param name="input">The array to be copied the input from.</param>
        /// <param name="expectedOutput">The expected output of the network.</param>
        /// <param name="error">The array to be written the output error into.</param>
        /// <param name="learning">Whether the network is being used in a training session.</param>
        /// <returns>The output error of the network.</returns>
        public override double FeedAndGetError(double[] input, double[] expectedOutput, double[] error, bool learning)
        {
            return this.FeedAndGetError(input, 0, expectedOutput, 0, error, 0, learning);
        }

        /// <summary>Sets the input array and the output array of the network.</summary>
        /// <param name="input">The input array to be set.</param>
        /// <param name="output">The output array to be set.</param>
        public override void SetInputAndOutput(double[] input, double[] output)
        {
            if (this.layersConnected)
            {
                ((IArraysLayer)this.Layers.First()).SetInputAndOutput(input, ((IArraysLayer)this.Layers.First()).Output);
                ((IArraysLayer)this.Layers.Last()).SetInputAndOutput(((IArraysLayer)this.Layers.Last()).Input, output);
            }
            else
            {
                double[] array = input;
                for (int i = 0; i < this.Layers.Count - 1; i++)
                {
                    array = ((IArraysLayer)this.Layers[i]).SetInputGetOutput(array);
                }
                ((IArraysLayer)this.Layers.Last()).SetInputAndOutput(array, output);
            }
        }

        /// <summary>Sets the input array of the network and creates and sets the output array.</summary>
        /// <param name="input">The input array to be set.</param>
        /// <returns>The created output array.</returns>
        public override double[] SetInputGetOutput(double[] input)
        {
            double[] retVal = new double[this.OutputSize];
            this.SetInputAndOutput(input, retVal);
            return retVal;
        }

        /// <summary>Creates a siamese of the network.</summary>
        /// <returns>The created siamese.</returns>
        public override IUntypedLayer CreateSiamese()
        {
            return new FeedForwardNN(this, true);
        }

        /// <summary>Creates a clone of the network.</summary>
        /// <returns>The created clone.</returns>
        public override IUntypedLayer Clone()
        {
            return new FeedForwardNN(this, false);
        }

        /// <summary>Adds a top layer to the network.</summary>
        /// <param name="layer">The layer to be added.</param>
        protected override void AddTopLayer(ILayer<double[], double[]> layer)
        {
            this.Layers.Insert(0, layer);
        }

        /// <summary>Adds a bottom layer to the network.</summary>
        /// <param name="layer">The layer to be added.</param>
        protected override void AddBottomLayer(ILayer<double[], double[]> layer)
        {
            this.Layers.Add(layer);
        }

        /// <summary>Removes a top layer.</summary>
        protected override void RemoveTopLayer()
        {
            this.Layers.RemoveAt(0);
        }

        /// <summary>Removes a bottom layer.</summary>
        protected override void RemoveBottomLayer()
        {
            throw new NotImplementedException();
        }
    }
}
