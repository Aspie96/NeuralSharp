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
using System.IO;
using System.Linq;
using System.Runtime.Serialization;
using System.Text;
using System.Threading.Tasks;

namespace NeuralSharp
{
    /// <summary>Represents a deconvolutional neural network.</summary>
    [DataContract]
    public class DeConvolutionalNN : ForwardLearner<double[], Image>, IArrayImageLayer
    {
        [DataMember]
        private FeedForwardNN firstPart;
        [DataMember]
        private ArrayToImage a2i;
        [DataMember]
        private PurelyConvolutionalNN cnn;
        private bool layersConnected;
        private double[] errorArray;
        private Image errorImage;

        /// <summary>Either creates a siamese or clones the given <code>DeConvolutionalNN</code> instance.</summary>
        /// <param name="original">The original instance to be created a siamese of or cloned.</param>
        /// <param name="siamese"><code>true</code> if a siamese is to be created, <code>false</code> if a clone is.</param>
        protected DeConvolutionalNN(DeConvolutionalNN original, bool siamese)
        {
            this.errorArray = Backbone.CreateArray<double>(original.cnn.InputDepth * original.cnn.InputWidth * original.cnn.InputHeight);
            this.errorImage = new Image(original.cnn.InputDepth, original.cnn.InputWidth, original.cnn.InputHeight);
            this.layersConnected = false;
            if (siamese)
            {
                this.firstPart = (FeedForwardNN)original.firstPart.CreateSiamese();
                this.a2i = (ArrayToImage)original.a2i.CreateSiamese();
                this.cnn = (PurelyConvolutionalNN)original.cnn.CreateSiamese();
            }
            else
            {
                this.firstPart = (FeedForwardNN)original.firstPart.Clone();
                this.a2i = (ArrayToImage)original.a2i.Clone();
                this.cnn = (PurelyConvolutionalNN)original.cnn.Clone();
            }
        }

        /// <summary>Creates an instance of the <code>DeConvolutionalNN</code> class.</summary>
        /// <param name="firstPart">First part of the network.</param>
        /// <param name="cnn">Second part of the network.</param>
        /// <param name="createIO">Whether the input array and the output image of the netwok are to be created.</param>
        public DeConvolutionalNN(FeedForwardNN firstPart, PurelyConvolutionalNN cnn, bool createIO = true)
        {
            this.firstPart = firstPart;
            this.a2i = new ArrayToImage(cnn.InputDepth, cnn.InputWidth, cnn.InputHeight, false);
            this.cnn = cnn;
            this.errorArray = Backbone.CreateArray<double>(cnn.InputDepth * cnn.InputWidth * cnn.InputHeight);
            this.errorImage = new Image(cnn.InputDepth, cnn.InputWidth, cnn.InputHeight);
            this.layersConnected = false;
            if (createIO)
            {
                this.SetInputGetOutput(Backbone.CreateArray<double>(firstPart.InputSize));
            }
        }
        
        /// <summary>The input array of the network.</summary>
        public double[] Input
        {
            get { return this.firstPart.Input; }
        }

        /// <summary>The output image of the network.</summary>
        public Image Output
        {
            get { return this.cnn.Output; }
        }

        /// <summary>The depth of the output of the network.</summary>
        public int OutputDepth
        {
            get { return this.cnn.OutputDepth; }
        }

        /// <summary>The width of the output of the network.</summary>
        public int OutputWidth
        {
            get { return this.cnn.OutputWidth; }
        }

        /// <summary>The height of the output of the network.</summary>
        public int OutputHeight
        {
            get { return this.cnn.OutputHeight; }
        }

        /// <summary>The index of the first used position of the input array.</summary>
        public int InputSkip
        {
            get { return this.firstPart.InputSkip; }
        }

        /// <summary>The lenght of the input of the network.</summary>
        public int InputSize
        {
            get { return this.firstPart.InputSize; }
        }
        
        /// <summary>The amount of parameters of the network.</summary>
        public override int Parameters
        {
            get { return this.firstPart.Parameters + this.cnn.Parameters; }
        }

        [OnDeserialized]
        private void SetValuesOnDeserialized(StreamingContext context)
        {
            this.layersConnected = false;
            this.errorArray = Backbone.CreateArray<double>(this.firstPart.OutputSize);
            this.errorImage = new Image(this.cnn.InputDepth, this.cnn.InputWidth, this.cnn.InputHeight);
        }

        /// <summary>Creates an image which can be used as output error.</summary>
        /// <returns>The created image.</returns>
        protected override Image NewError()
        {
            return new Image(this.OutputDepth, this.OutputWidth, this.OutputHeight);
        }

        public override void BackPropagate(Image outputError, bool learning)
        {
            this.cnn.BackPropagate(outputError, this.errorImage, learning);
            this.a2i.BackPropagate(this.errorImage, this.errorArray, learning);
            this.firstPart.BackPropagate(this.errorArray, learning);
        }

        /// <summary>Backpropagates an error trough the network.</summary>
        /// <param name="outputError">The output error to be backpropagated.</param>
        /// <param name="inputErrorArray">The array to be written the input error into.</param>
        /// <param name="inputErrorSkip">The index of the first used entry of the input error array.</param>
        /// <param name="learning">Whether the network is being used in a training session.</param>
        public void BackPropagate(Image outputError, double[] inputErrorArray, int inputErrorSkip, bool learning)
        {
            this.cnn.BackPropagate(outputError, this.errorImage, learning);
            this.a2i.BackPropagate(this.errorImage, this.errorArray, learning);
            this.firstPart.BackPropagate(this.errorArray, 0, inputErrorArray, inputErrorSkip, learning);
        }

        /// <summary>Backpropagates an error trough the network.</summary>
        /// <param name="outputError">The output error to be backpropagated.</param>
        /// <param name="inputError">The array to be written the input error into.</param>
        /// <param name="learning">Whether the network is being used in a training session.</param>
        public override void BackPropagate(Image outputError, double[] inputError, bool learning)
        {
            this.BackPropagate(outputError, inputError, 0, learning);
        }

        /// <summary>Feeds the network forward.</summary>
        /// <param name="learning">Whether the network is being used in a training session.</param>
        public void Feed(bool learning = false)
        {
            this.firstPart.Feed(learning);
            this.a2i.Feed(learning);
            this.cnn.Feed(learning);
        }

        /// <summary>Feeds the network forward.</summary>
        /// <param name="inputArray">The array to be copied the input from.</param>
        /// <param name="inputSkip">The index of the first entry of the given input array to be used.</param>
        /// <param name="output">The image to be copied the output into.</param>
        /// <param name="learning">Whether the network is being used in a training session.</param>
        public void Feed(double[] inputArray, int inputSkip, Image output, bool learning = false)
        {
            Backbone.CopyArray(inputArray, inputSkip, this.Input, this.InputSkip, this.InputSize);
            this.Feed(learning);
            output.FromImage(this.Output);
        }

        /// <summary>Feeds the network forward.</summary>
        /// <param name="input">The array to be copied the input from.</param>
        /// <param name="output">The image to be copied the output into.</param>
        /// <param name="learning">Whether the network is being used in a training session.</param>
        public override void Feed(double[] input, Image output, bool learning = false)
        {
            this.Feed(input, 0, output, learning);
        }

        /// <summary>Sets the input array and the output image of the network, connecting its inner layers.</summary>
        /// <param name="inputArray">The input array to be set.</param>
        /// <param name="inputSkip">The index of the first entry of the input array to be used.</param>
        /// <param name="output">The output image to be set.</param>
        public void SetInputAndOutput(double[] inputArray, int inputSkip, Image output)
        {
            if (this.layersConnected)
            {
                this.firstPart.SetInputAndOutput(inputArray, InputSkip, this.firstPart.Output, this.firstPart.OutputSkip);
                this.cnn.SetInputAndOutput(this.cnn.Input, output);
            }
            else
            {
                double[] array = this.firstPart.SetInputGetOutput(inputArray, inputSkip);
                Image image = this.a2i.SetInputGetOutput(array);
                this.cnn.SetInputAndOutput(image, output);
                this.layersConnected = true;
            }
        }

        /// <summary>Sets the input array and the output image of the network, connecting its inner layers.</summary>
        /// <param name="input">The input array to be set.</param>
        /// <param name="output">The output image to be set.</param>
        public void SetInputAndOutput(double[] input, Image output)
        {
            this.SetInputAndOutput(input, 0, output);
        }

        /// <summary>Sets the input array of the network and creates and sets the output image.</summary>
        /// <param name="input">The input array to be set.</param>
        /// <returns>The created output image.</returns>
        public Image SetInputGetOutput(double[] input)
        {
            Image retVal = new Image(this.OutputDepth, this.OutputWidth, this.OutputHeight);
            this.SetInputAndOutput(input, retVal);
            return retVal;
        }

        /// <summary>Updates the weights of the network.</summary>
        /// <param name="rate">The learning rate to be used.</param>
        /// <param name="momentum">The momentum to be used.</param>
        public override void UpdateWeights(double rate, double momentum = 0.0)
        {
            this.firstPart.UpdateWeights(rate, momentum);
            this.cnn.UpdateWeights(rate, momentum);
        }
        
        /// <summary>Gets the error of the network, given its actual output and expected output.</summary>
        /// <param name="output">Actual output of the network.</param>
        /// <param name="expectedOuptut">Expected output of the network.</param>
        /// <param name="error">Image to be written the output error into.</param>
        /// <returns>The output error of the network.</returns>
        public override double GetError(Image output, Image expectedOuptut, Image error)
        {
            double err = 0.0;
            for (int i = 0; i < this.OutputDepth; i++)
            {
                for (int j = 0; j < this.OutputWidth; j++)
                {
                    for (int k = 0; k < this.OutputHeight; k++)
                    {
                        double e = expectedOuptut.GetValue(i, j, k) - output.GetValue(i, j, k);
                        error.SetValue(i, j, k, (float)e);
                        err += e * e;
                    }
                }
            }
            return err;
        }

        /// <summary>Feeds the network forward and gets its error, given its expected output.</summary>
        /// <param name="inputArray">The array to be copied the input from.</param>
        /// <param name="inputSkip">The index of the first used entry of the input array.</param>
        /// <param name="expectedOutput">The expected output of the network.</param>
        /// <param name="error">The image to be written the output array into.</param>
        /// <param name="learning">Whether the network is being used in a training session.</param>
        /// <returns>The output error of the network.</returns>
        public double FeedAndGetError(double[] inputArray, int inputSkip, Image expectedOutput, Image error, bool learning)
        {
            Backbone.CopyArray(inputArray, inputSkip, this.Input, this.InputSkip, this.InputSize);
            this.Feed(learning);
            return this.GetError(this.Output, expectedOutput, error);
        }

        /// <summary>Feeds the network forward and gets its error, given its expected output.</summary>
        /// <param name="input">The array to be copied the input from.</param>
        /// <param name="expectedOutput">The expected output of the network.</param>
        /// <param name="error">The image to be written the output array into.</param>
        /// <param name="learning">Whether the network is being used in a training session.</param>
        /// <returns>The output array of the network.</returns>
        public override double FeedAndGetError(double[] input, Image expectedOutput, Image error, bool learning)
        {
            return this.FeedAndGetError(input, 0, expectedOutput, error, learning);
        }

        /// <summary>Exports the deconvolutional neural network to a stream.</summary>
        /// <param name="stream">The stream to be exported the deconvolutional neural network into.</param>
        public override void Save(Stream stream)
        {
            new DataContractSerializer(typeof(DeConvolutionalNN)).WriteObject(stream, this);
        }
        
        /// <summary>Creates a siamese of the network.</summary>
        /// <returns>The created instance of the <code>DeConvolutionalNN</code> class.</returns>
        public virtual IUntypedLayer CreateSiamese()
        {
            return new DeConvolutionalNN(this, true);
        }

        /// <summary>Creates a clone of the network.</summary>
        /// <returns>The created instance of the <code>DeConvolutionalNN</code> class.</returns>
        public virtual IUntypedLayer Clone()
        {
            return new DeConvolutionalNN(this, false);
        }
    }
}
