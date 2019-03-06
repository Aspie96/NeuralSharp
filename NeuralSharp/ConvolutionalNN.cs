﻿/*
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
using System.IO;
using System.Linq;
using System.Runtime.Serialization;
using System.Text;
using System.Threading.Tasks;

namespace NeuralSharp
{
    /// <summary>Represents a convolutional neural network.</summary>
    [DataContract]
    public class ConvolutionalNN : ForwardLearner<Image, double[]>, IImageArrayLayer
    {
        [DataMember]
        private PurelyConvolutionalNN firstPart;
        [DataMember]
        private ImageToArray i2a;
        [DataMember]
        private FeedForwardNN fnn;
        private bool layersConnected;
        private double[] errorArray;
        private Image errorImage;
        
        /// <summary>Either creates a siamese of the given <code>ConvolutionalNN</code> instance or clones it.</summary>
        /// <param name="original">The original instance to be created a siamese of or cloned.</param>
        /// <param name="siamese"><code>true</code> if a siamese is to be created, <code>false</code> otherwise.</param>
        protected ConvolutionalNN(ConvolutionalNN original, bool siamese)
        {
            this.errorArray = Backbone.CreateArray<double>(firstPart.OutputDepth * firstPart.OutputWidth * firstPart.OutputHeight);
            this.errorImage = new Image(this.firstPart.OutputDepth, this.firstPart.OutputWidth, this.firstPart.OutputHeight);
            this.layersConnected = false;
            if (siamese)
            {
                this.firstPart = (PurelyConvolutionalNN)original.CreateSiamese();
                this.i2a = (ImageToArray)original.i2a.CreateSiamese();
                this.fnn = (FeedForwardNN)original.fnn.CreateSiamese();
            }
            else
            {
                this.firstPart = (PurelyConvolutionalNN)original.firstPart.Clone();
                this.i2a = (ImageToArray)original.i2a.Clone();
                this.fnn = (FeedForwardNN)original.fnn.Clone();
            }
        }

        /// <summary>Creates an instance of the <code>ConvolutionalNN</code> class.</summary>
        /// <param name="firstPart">The first part of the convolutional neural network.</param>
        /// <param name="fnn">The second part of the convolutional neural network.</param>
        /// <param name="createIO">Whether the input image and the output array of the network are to be created.</param>
        public ConvolutionalNN(PurelyConvolutionalNN firstPart, FeedForwardNN fnn, bool createIO = true)
        {
            this.firstPart = firstPart;
            this.i2a = new ImageToArray(firstPart.OutputDepth, firstPart.OutputWidth, firstPart.OutputHeight, false);
            this.fnn = fnn;
            this.errorArray = Backbone.CreateArray<double>(firstPart.OutputDepth * firstPart.OutputWidth * firstPart.OutputHeight);
            this.errorImage = new Image(this.firstPart.OutputDepth, this.firstPart.OutputWidth, this.firstPart.OutputHeight);
            if (createIO)
            {
                this.SetInputGetOutput(new Image(firstPart.InputDepth, firstPart.InputWidth, firstPart.InputHeight));
            }
        }
        
        /// <summary>The input image of the network.</summary>
        public Image Input
        {
            get { return this.firstPart.Input; }
        }

        /// <summary>The output array of the network.</summary>
        public double[] Output
        {
            get { return this.fnn.Output; }
        }

        /// <summary>The depth of the input of the network.</summary>
        public int InputDepth
        {
            get { return this.firstPart.InputDepth; }
        }

        /// <summary>The width of the input of the network.</summary>
        public int InputWidth
        {
            get { return this.firstPart.InputWidth; }
        }

        /// <summary>The height of the input of the network.</summary>
        public int InputHeight
        {
            get { return this.firstPart.InputHeight; }
        }

        /// <summary>The index of the first used entry of the output array.</summary>
        public int OutputSkip
        {
            get { return this.fnn.OutputSkip; }
        }

        /// <summary>The lenght of the output of the network.</summary>
        public int OutputSize
        {
            get { return this.fnn.OutputSize; }
        }
        
        /// <summary>The amount of parameters of the network.</summary>
        public override int Parameters
        {
            get { return this.firstPart.Parameters + this.fnn.Parameters; }
        }

        [OnDeserialized]
        private void SetValuesOnDeserialized(StreamingContext context)
        {
            this.layersConnected = false;
            this.errorArray = Backbone.CreateArray<double>(this.firstPart.OutputDepth * this.firstPart.OutputWidth * this.firstPart.OutputHeight);
            this.errorImage = new Image(this.firstPart.OutputDepth, this.firstPart.OutputWidth, this.firstPart.OutputHeight);
        }

        /// <summary>Creates a new array which can be used as output error.</summary>
        /// <returns>The created array.</returns>
        protected override double[] NewError()
        {
            return Backbone.CreateArray<double>(this.OutputSize);
        }

        /// <summary>Backpropagates the given error trough the network.</summary>
        /// <param name="outputErrorArray">The output array to be backpropagated.</param>
        /// <param name="outputErrorSkip">The index of the first entry of the output array to be used.</param>
        /// <param name="learning">Whether the network is being used in a training session.</param>
        public void BackPropagate(double[] outputErrorArray, int outputErrorSkip, bool learning)
        {
            this.fnn.BackPropagate(outputErrorArray, outputErrorSkip, this.errorArray, 0, learning);
            this.i2a.BackPropagate(this.errorArray, this.errorImage, learning);
            this.firstPart.BackPropagate(this.errorImage, learning);
        }

        /// <summary>Backpropagates the given error trough the network.</summary>
        /// <param name="outputError">The output error to be backpropagated.</param>
        /// <param name="learning">Whether the network is being used in a training session.</param>
        public override void BackPropagate(double[] outputError, bool learning = true)
        {
            this.BackPropagate(outputError, 0, learning);
        }

        /// <summary>Backpropagates the given error trought the network.</summary>
        /// <param name="outputErrorArray">The output error to be backpropagated.</param>
        /// <param name="outputErrorSkip">The index of the first entry of the output error array to be used.</param>
        /// <param name="inputError">The image to be written the input error into.</param>
        /// <param name="learning">Whether the network is being used in a training session.</param>
        public void BackPropagate(double[] outputErrorArray, int outputErrorSkip, Image inputError, bool learning)
        {
            this.fnn.BackPropagate(outputErrorArray, outputErrorSkip, this.errorArray, 0, learning);
            this.i2a.BackPropagate(this.errorArray, this.errorImage, learning);
            this.firstPart.BackPropagate(this.errorImage, inputError, learning);
        }

        /// <summary>Backpropagates the given error trough the network.</summary>
        /// <param name="outputError">The output error to be backpropgated.</param>
        /// <param name="inputError">The image to be written the input error into.</param>
        /// <param name="learning">Whether the network is being used in a training session.</param>
        public override void BackPropagate(double[] outputError, Image inputError, bool learning)
        {
            this.BackPropagate(outputError, 0, inputError, learning);
        }

        /// <summary>Feeds the network forward.</summary>
        /// <param name="learning">Whether the network is being used in a training session.</param>
        public void Feed(bool learning = false)
        {
            this.firstPart.Feed(learning);
            this.i2a.Feed(learning);
            this.fnn.Feed(learning);
        }

        /// <summary>Feeds the network forward.</summary>
        /// <param name="input">The image to be copied the input from.</param>
        /// <param name="outputArray">The array to be copied the output into.</param>
        /// <param name="outputSkip">The index of the first entry of the given output array to be used.</param>
        /// <param name="learning">Whether the network is being used in a training session.</param>
        public void Feed(Image input, double[] outputArray, int outputSkip, bool learning = false)
        {
            this.Input.FromImage(input);
            this.Feed(learning);
            Array.Copy(this.Output, this.OutputSkip, outputArray, outputSkip, this.OutputSize);
        }

        /// <summary>Feeds the network forward.</summary>
        /// <param name="input">The image to be copied the input from.</param>
        /// <param name="outputArray">The array to be copied the output into.</param>
        /// <param name="learning">Whether the network is being used in a training session.</param>
        public override void Feed(Image input, double[] outputArray, bool learning = false)
        {
            this.Feed(input, outputArray, 0, learning);
        }

        /// <summary>Sets the input image and the output array of the network, connecting its inner layers if needed.</summary>
        /// <param name="input">The input image to be set.</param>
        /// <param name="outputArray">The output array to be set.</param>
        /// <param name="outputSkip">The index of the first entry of the output array to be used.</param>
        public void SetInputAndOutput(Image input, double[] outputArray, int outputSkip)
        {
            if (this.layersConnected)
            {
                this.firstPart.SetInputAndOutput(input, this.firstPart.Output);
                this.fnn.SetInputAndOutput(this.fnn.Input, this.fnn.InputSkip, outputArray, outputSkip);
            }
            else
            {
                Image image = this.firstPart.SetInputGetOutput(input);
                double[] array = this.i2a.SetInputGetOutput(image);
                this.fnn.SetInputAndOutput(array, 0, outputArray, outputSkip);
                this.layersConnected = true;
            }
        }

        /// <summary>Sets the input image and the output array of the network, connecting its inner layers if needed.</summary>
        /// <param name="input">The input image to be set.</param>
        /// <param name="output">The output array to be set.</param>
        public void SetInputAndOutput(Image input, double[] output)
        {
            this.SetInputAndOutput(input, output, 0);
        }

        /// <summary>Sets the input image of the network and creates and sets an output array.</summary>
        /// <param name="input">The input image to be set.</param>
        /// <returns>The created output array.</returns>
        public double[] SetInputGetOutput(Image input)
        {
            double[] retVal = Backbone.CreateArray<double>(this.OutputSize);
            this.SetInputAndOutput(input, retVal);
            return retVal;
        }

        /// <summary>Updates the weights of the network.</summary>
        /// <param name="rate">The learning rate to be used.</param>
        /// <param name="momentum">The momentum to be used.</param>
        public override void UpdateWeights(double rate, double momentum = 0.0)
        {
            this.firstPart.UpdateWeights(rate, momentum);
            this.fnn.UpdateWeights(rate, momentum);
        }

        /// <summary>Gets the error of the network, given its actual and expected output.</summary>
        /// <param name="outputArray">The actual output of the network.</param>
        /// <param name="outputSkip">The index of the first entry of the given output array to be used.</param>
        /// <param name="expectedArray">The expected output of the network.</param>
        /// <param name="expectedSkip">The index of the first entry of the expected array to be used.</param>
        /// <param name="errorArray">The array to be written the output error into.</param>
        /// <param name="errorSkip">The index of the first entry of the output error array to be used.</param>
        /// <returns>The error of the network.</returns>
        public virtual double GetError(double[] outputArray, int outputSkip, double[] expectedArray, int expectedSkip, double[] errorArray, int errorSkip)
        {
            return Backbone.GetError(outputArray, outputSkip, expectedArray, expectedSkip, errorArray, errorSkip, this.OutputSize);
        }

        /// <summary>Gets the error of the network, given its actual and expected output.</summary>
        /// <param name="output">The actual output of the network.</param>
        /// <param name="expectedOuptut">The expected output of the network.</param>
        /// <param name="error">The array to be written the output error into.</param>
        /// <returns>The error of the network.</returns>
        public override double GetError(double[] output, double[] expectedOuptut, double[] error)
        {
            return this.GetError(output, 0, expectedOuptut, 0, error, 0);
        }

        /// <summary>Feeds the network forward and, given the expected output, gets its error.</summary>
        /// <param name="input">The image to be copied the input from.</param>
        /// <param name="expectedArray">The expected output of the network.</param>
        /// <param name="expectedSkip">The index of the first entry of the expected array to be used.</param>
        /// <param name="errorArray">The array to be written the output error into.</param>
        /// <param name="errorSkip">The index of the first entry of the output error array to be written into.</param>
        /// <param name="learning">Whether the network is being used in a learning session.</param>
        /// <returns>The error of the network.</returns>
        public double FeedAndGetError(Image input, double[] expectedArray, int expectedSkip, double[] errorArray, int errorSkip, bool learning)
        {
            this.Input.FromImage(input);
            this.Feed(learning);
            return this.GetError(this.Output, this.OutputSkip, expectedArray, expectedSkip, errorArray, errorSkip);
        }

        /// <summary>Feeds the network forward and, given the expected output, gets its error.</summary>
        /// <param name="input">The image to be copied the input from.</param>
        /// <param name="expectedOutput">The expected output of the network.</param>
        /// <param name="error">The array to be written the output error into.</param>
        /// <param name="learning">Whether the network is being used in a learning session.</param>
        /// <returns>The error of the network.</returns>
        public override double FeedAndGetError(Image input, double[] expectedOutput, double[] error, bool learning)
        {
            return this.FeedAndGetError(input, expectedOutput, 0, error, 0, learning);
        }

        /// <summary>Exports the convolutional neural network to a stream.</summary>
        /// <param name="stream">The stream to be exported the convolutional neural network into.</param>
        public override void Save(Stream stream)
        {
            new DataContractSerializer(typeof(ConvolutionalNN)).WriteObject(stream, this);
        }
        
        /// <summary>Creates a siamese of the network.</summary>
        /// <returns>The created instance of the <code>ConvolutionalNN</code> class.</returns>
        public virtual IUntypedLayer CreateSiamese()
        {
            return new ConvolutionalNN(this, true);
        }

        /// <summary>Creates a clone of the network.</summary>
        /// <returns>The created instance of the <code>ConvolutionalNN</code> class.</returns>
        public virtual IUntypedLayer Clone()
        {
            return new ConvolutionalNN(this, false);
        }
    }
}
