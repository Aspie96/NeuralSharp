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
using System.IO;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Json;

namespace NeuralNetwork.Convolutional
{
    /// <summary>Represents a convolutional neural network.</summary>
    [DataContract]
    [KnownType(typeof(Convolution))]
    [KnownType(typeof(MaxPooling))]
    [KnownType(typeof(FeedForwardNN))]
    public class ConvolutionalNN : ForwardLearner<Image>, IConvolutionalNN
    {
        [DataMember]
        private IFeedForwardNN fullyConnected;
        [DataMember]
        PurelyConvolutionalNN cnn;
        private double[] auxArray;

        /// <summary>Empty constructor. It does not initialize the fields.</summary>
        protected ConvolutionalNN() { }

        /// <summary>Creates a new instance of the <code>ConvolutionalNN</code> class.</summary>
        /// <param name="depth">The depth of the input image of the convolutional neural network.</param>
        /// <param name="width">The widht of the input image of the convolutional neural network.</param>
        /// <param name="height">The height of the input image of the convolutional neural network.</param>
        /// <param name="layersInfo">Information about the layers in the convolutional neural network, before the fully connected ones.</param>
        /// <param name="fullyConnected">The fully connected layers of the convolutional neural network.</param>
        public ConvolutionalNN(int depth, int width, int height, IEnumerable<ITransofrmationInfo> layersInfo, IFeedForwardNN fullyConnected)
        {
            this.cnn = new PurelyConvolutionalNN(depth, width, height, layersInfo);
            this.auxArray = new double[this.cnn.Outputs];
            this.fullyConnected = fullyConnected;
        }

        /// <summary>Creates an instance of the <code>ConvolutionalNN</code> class.</summary>
        /// <param name="depth">The depth of the input image of the convolutional neural network.</param>
        /// <param name="width">The widht of the input image of the convolutional neural network.</param>
        /// <param name="height">The height of the input image of the convolutional neural network.</param>
        /// <param name="layersInfo">Information about the layers in the convolutional neural network, before the fully connected ones.</param>
        /// <param name="outputs">The amount of outputs of this convolutional neural network.</param>
        /// <param name="classification"><code>true</code> if the convolutional neural network is to be used for classification purposes, <code>false</code> otherwise.</param>
        public ConvolutionalNN(int depth, int width, int height, IEnumerable<ITransofrmationInfo> layersInfo, int outputs, bool classification = false)
        {
            this.cnn = new PurelyConvolutionalNN(depth, width, height, layersInfo);
            this.auxArray = new double[this.cnn.Outputs];
            this.fullyConnected = new FeedForwardNN(this.cnn.Outputs, outputs, 2, classification, new int[1] { (this.cnn.Outputs + outputs) * 2 / 3 });
        }

        /// <summary>Creates a new instance of the <code>ConvolutionalNN</code> class.</summary>
        /// <param name="depth">The depth of the input image of the convolutional neural network.</param>
        /// <param name="width">The widht of the input image of the convolutional neural network.</param>
        /// <param name="height">The height of the input image of the convolutional neural network.</param>
        /// <param name="innerDepth">The depth of the layer in the convolutional neural network right before the fully connected ones.</param>
        /// <param name="outputs">The amount of outputs of the convolutional neural network.</param>
        /// <param name="classification"><code>true</code> if this neural network is to be used for classification purposes, <code>false</code> otherwise.</param>
        public ConvolutionalNN(int depth, int width, int height, int innerDepth, int outputs, bool classification = false)
        {
            this.cnn = new PurelyConvolutionalNN(depth, width, height, innerDepth);
            this.auxArray = new double[this.cnn.Outputs];
            this.fullyConnected = new FeedForwardNN(this.cnn.Outputs, outputs, 2, classification, new int[1] { this.cnn.Outputs * 2 / 3 + outputs });
        }

        /// <summary>Creates a new instance of the <code>ConvolutionalNN</code> class.</summary>
        /// <param name="depth">The depth of the input image of the convolutional neural network.</param>
        /// <param name="width">The widht of the input image of the convolutional neural network.</param>
        /// <param name="height">The height of the input image of the convolutional neural network.</param>
        /// <param name="outputs">The amount of outputs of the convolutional neural network.</param>
        /// <param name="classification"><code>true</code> if this neural network is to be used for classification purposes, <code>false</code> otherwise.</param>
        public ConvolutionalNN(int depth, int width, int height, int outputs, bool classification = false) : this(depth, width, height, (int)Math.Round(3 * Math.Log(depth * width * height * outputs * outputs, 2)), outputs, classification) { }

        /// <summary>The amount of outputs for this convolutional neural network.</summary>
        public override int Outputs
        {
            get { return this.fullyConnected.Outputs; }
        }

        /// <summary>The amount of parameters in this convolutional neural network.</summary>
        public int Params
        {
            get{ return this.fullyConnected.Params + this.cnn.Params; }
        }

        /// <summary>The amount of layer in this convolutional neural network before the fully connected ones.</summary>
        public int ConvLayers
        {
            get { return this.cnn.Layers; }
        }

        /// <summary>The depth of the input image of this convolutional neural network.</summary>
        public int InputDepth
        {
            get { return this.cnn.InputDepth; }
        }

        /// <summary>The width of the input image of this convolutional neural network.</summary>
        public int InputWidth
        {
            get { return this.cnn.InputWidth; }
        }

        /// <summary>The width of the output image of this convolutional neural network.</summary>
        public int InputHeight
        {
            get { return this.cnn.InputHeight; }
        }

        /// <summary>Backpropagates the given error trough this convolutional neural network, updating its weights.</summary>
        /// <param name="error">The error array to be backpropagated. It must refer to the lates feeding process.</param>
        /// <param name="rate">The learning rate at which he weights are to be updated.</param>
        public override void BackPropagate(double[] error, double rate)
        {
            this.fullyConnected.BackPropagate(error, this.auxArray, rate);
            this.cnn.BackPropagate(this.auxArray, rate);
        }

        /// <summary>Backpropagates the given error trough this convolutional neural network, updating its weights.</summary>
        /// <param name="error">The error array to be backpropagated. It must refer to the latest feeding process.</param>
        /// <param name="inputError">The image to be written the error of the input image into.</param>
        /// <param name="rate">The rate at which the weights are to be updated.</param>
        public override void BackPropagate(double[] error, Image inputError, double rate)
        {
            this.fullyConnected.BackPropagate(error, this.auxArray, rate);
            this.cnn.BackPropagate(this.auxArray, inputError, rate);
        }

        /// <summary>Feeds an image trough this convolutional neural network.</summary>
        /// <param name="input">The image to be fed trough the network.</param>
        /// <param name="output">The array to be written the output into.</param>
        public override void Feed(Image input, double[] output)
        {
            this.cnn.Feed(input, this.auxArray);
            this.fullyConnected.Feed(this.auxArray, output);
        }

        /// <summary>Feeds an image trough this convolutional neural network and gets the images generated during the process.</summary>
        /// <param name="input">The image to be fed trough the network.</param>
        /// <param name="output">The array to be written the output into.</param>
        /// <param name="steps">The array of images to be written each image generated by this convolutional neural network during the feeding process into.</param>
        public void Feed(Image input, double[] output, Image[] steps)
        {
            this.cnn.Feed(input, this.auxArray, steps);
            this.fullyConnected.Feed(this.auxArray, output);
        }

        /// <summary>Feeds an image trough this convolutional neural network and exports the images generated during the process.</summary>
        /// <param name="input">The image to be fed trough the network.</param>
        /// <param name="outputPath">The path of the folder to be exported the images into.</param>
        /// <param name="output">The array to be written the output into.</param>
        public void Visualize(Image input, string outputPath, double[] output = null)
        {
            this.cnn.Visualize(input, outputPath, output);
        }

        /// <summary>Exports the images generated by this neural network during the latest feeding process.</summary>
        /// <param name="outputPath">The path of the folder to be exported the images into.</param>
        public void Visualize(string outputPath)
        {
            this.cnn.Visualize(outputPath);
        }

        /// <summary>Feeds an image trough the network and gets an error array for the generated outputs.</summary>
        /// <param name="input">The input image to be fed.</param>
        /// <param name="expected">The expected outputs.</param>
        /// <param name="error">The array to be written the error into.</param>
        /// <returns>A value represending how big the error is.</returns>
        public override double GetError(Image input, double[] expected, double[] error)
        {
            this.cnn.Feed(input, this.auxArray);
            return this.fullyConnected.GetError(this.auxArray, expected, error);
        }
        
        [OnDeserialized]
        private void SetValuesOnDeserialized(StreamingContext context)
        {
            this.auxArray = new double[this.cnn.Outputs];
        }

        /// <summary>Exports this instance of the <code>ConvolutionalNN</code> class to a stream.</summary>
        /// <param name="stream">The stream to be exported into.</param>
        public override void Save(Stream stream)
        {
            new DataContractJsonSerializer(typeof(ConvolutionalNN)).WriteObject(stream, this);
        }

        /// <summary>Imports an instance of the <code>ConvolutionalNN</code> class from a stream.</summary>
        /// <param name="stream">The stream to be imported from.</param>
        /// <returns>The imported instance of the <code>ConvolutionalNN</code> class.</returns>
        public static ConvolutionalNN Load(Stream stream)
        {
            return (ConvolutionalNN)(new DataContractJsonSerializer(typeof(ConvolutionalNN)).ReadObject(stream));
        }

        /// <summary>Imports an instance of the <code>ConvolutionalNN</code> class from a file.</summary>
        /// <param name="fileName">The of the file to be imported from.</param>
        /// <returns>The imported instance of the <code>ConvolutionalNN</code> class.</returns>
        public static ConvolutionalNN Load(string fileName)
        {
            FileStream fs = new FileStream(fileName, FileMode.Open);
            ConvolutionalNN retVal = ConvolutionalNN.Load(fs);
            fs.Close();
            return retVal;
        }

        /// <summary>Copies this instance of <code>ConvolutionalNN</code> into another.</summary>
        /// <param name="cnn">The instance to be copied into.</param>
        protected virtual void CloneTo(ConvolutionalNN cnn)
        {
            cnn.cnn = (PurelyConvolutionalNN)this.cnn.Clone();
            cnn.fullyConnected = (IFeedForwardNN)this.fullyConnected.Clone();
            cnn.auxArray = new double[cnn.cnn.Outputs];
        }

        /// <summary>Creates a copy of this instance of the <code>ConvolutionalNN</code> class.</summary>
        /// <returns>The generated instance of the <code>ConvolutionalNN</code> class.</returns>
        public override object Clone()
        {
            ConvolutionalNN retVal = new ConvolutionalNN();
            this.CloneTo(retVal);
            return retVal;
        }
    }
}
