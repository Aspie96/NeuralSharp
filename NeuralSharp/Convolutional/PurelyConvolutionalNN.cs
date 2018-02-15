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
using System.IO;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Json;

namespace NeuralNetwork.Convolutional
{
    /// <summary>Represents a convolutional neural network with no fully connected layers.</summary>
    [DataContract]
    [KnownType(typeof(Convolution))]
    [KnownType(typeof(MaxPooling))]
    public class PurelyConvolutionalNN : ForwardLearner<Image>, IConvolutionalNN
    {
        private Image first;
        private Image last;
        [DataMember]
        private IImageTransformation[] transformations;
        private Image error1;
        private Image error2;
        [DataMember]
        private int outputs;
        [DataMember]
        private int inputDepth;
        [DataMember]
        private int inputWidth;
        [DataMember]
        private int inputHeight;

        /// <summary>Empty constructor. It does not initialize the fields.</summary>
        private PurelyConvolutionalNN() { }

        private void InitializeConv(int depth, int width, int height, IEnumerable<ITransofrmationInfo> layersInfo)
        {
            this.inputDepth = depth;
            this.inputWidth = width;
            this.inputHeight = height;
            this.first = new Image(depth, width, height);
            this.transformations = new IImageTransformation[layersInfo.Count()];
            Image prev = this.first;
            int maxDepth = depth;
            int maxWidth = width;
            int maxHeight = height;
            for (int i = 0; i < this.transformations.Length; i++)
            {
                ITransofrmationInfo info = layersInfo.ElementAt(i);
                info.SizeAfter(prev.Depth, prev.Width, prev.Height, out int outDepth, out int outWidth, out int outHeight);
                maxDepth = Math.Max(maxDepth, outDepth);
                maxWidth = Math.Max(maxWidth, outWidth);
                maxHeight = Math.Max(maxHeight, outHeight);
                Image output = new Image(outDepth, outWidth, outHeight);
                this.transformations[i] = info.GetTransformation(prev, output);
                prev = output;
            }
            this.last = prev;
            this.error1 = new Image(maxDepth, maxWidth, maxHeight);
            this.error2 = new Image(maxDepth, maxWidth, maxHeight);
        }

        /// <summary>Creates a new instance of the <code>PurelyConvolutionalNN</code> class.</summary>
        /// <param name="depth">The depth of the input image of the network.</param>
        /// <param name="width">The width of the input image of the network.</param>
        /// <param name="height">The height of the input image of the network.</param>
        /// <param name="layersInfo">Information about the layers of the network.</param>
        public PurelyConvolutionalNN(int depth, int width, int height, IEnumerable<ITransofrmationInfo> layersInfo)
        {
            this.InitializeConv(depth, width, height, layersInfo);
            this.outputs = this.last.Raw.Length;
        }
        
        /// <summary>Creates a new instance of the <code>PurelyConvolutionalNN</code> class.</summary>
        /// <param name="depth">The depth of the input image of the network.</param>
        /// <param name="width">The width of the input image of the network.</param>
        /// <param name="height">The height of the input image of the network.</param>
        /// <param name="innerDepth">The depth of the output image of the network.</param>
        public PurelyConvolutionalNN(int depth, int width, int height, int innerDepth)
        {
            List<ITransofrmationInfo> layersInfo = new List<ITransofrmationInfo>();
            int auxDepth = depth;
            int auxWidth = width;
            int auxHeight = height;
            int step = 0;
            int convLayers = 0;
            while (auxWidth > 1 && auxHeight > 1)
            {
                ITransofrmationInfo newElement;
                if (step == 0 || step == 1)
                {
                    if (auxWidth <= 3 || auxHeight <= 3)
                    {
                        newElement = new ConvLayerInfo(2, 1, 1, false);
                    }
                    else if (width % 2 == 1)
                    {
                        newElement = new ConvLayerInfo(4, 1, 1, false);
                    }
                    else
                    {
                        newElement = new ConvLayerInfo(3, 1, 1, false);
                    }
                    convLayers++;
                }
                else
                {
                    newElement = new PoolLayerInfo(2, 2);
                }
                newElement.SizeAfter(auxDepth, auxWidth, auxHeight, out auxDepth, out auxWidth, out auxHeight);
                layersInfo.Add(newElement);
                step = (step + 1) % 3;
            }
            double alpha = Math.Pow((double)innerDepth / depth, 1.0 / convLayers);
            int index = 1;
            for (int i = 0; i < layersInfo.Count; i++)
            {
                if (layersInfo[i] is ConvLayerInfo)
                {
                    layersInfo[i] = new ConvLayerInfo(((ConvLayerInfo)layersInfo[i]).KernelSide, (int)Math.Round(Math.Pow(alpha, index) * depth), 1, false);
                    index++;
                }
            }
            this.InitializeConv(depth, width, height, layersInfo);
            int inputs = last.Raw.Length;
            this.outputs = this.last.Raw.Length;
        }
        
        /// <summary>The amount of outputs of this network.</summary>
        public override int Outputs
        {
            get { return this.outputs; }
        }

        /// <summary>The amount of weights of the network.</summary>
        public int Params
        {
            get
            {
                int retVal = 0;
                for (int i = 0; i < this.transformations.Length; i++)
                {
                    retVal += this.transformations[i].Params;
                }
                return retVal;
            }
        }

        /// <summary>The amount of layers in this network.</summary>
        public int Layers
        {
            get { return this.transformations.Length; }
        }

        /// <summary>The depth of the input image of this network.</summary>
        public int InputDepth
        {
            get { return this.inputDepth; }
        }

        /// <summary>The width of the input image of this network.</summary>
        public int InputWidth
        {
            get { return this.inputWidth; }
        }

        /// <summary>The height of the input image of this network.</summary>
        public int InputHeight
        {
            get { return this.inputHeight; }
        }

        /// <summary>The depth of the output image of this network.</summary>
        public int OutputDepth
        {
            get { return this.last.Depth; }
        }

        /// <summary>The width of the output image of this network.</summary>
        public int OutputWidth
        {
            get { return this.last.Width; }
        }

        /// <summary>The height of the output image of this network.</summary>
        public int OutputHeight
        {
            get { return this.last.Height; }
        }

        private void BackPropagateFunc(double rate)
        {
            for (int i = this.transformations.Length - 1; i >= 0; i--)
            {
                this.transformations[i].BackPropagate(this.error2, this.error1, rate);
                Image aux = this.error1;
                this.error1 = this.error2;
                this.error2 = aux;
            }
        }

        /// <summary>Backpropagates the given error trough the network, updating its weights.</summary>
        /// <param name="error">The error array to be backpropagated. It must refer to the latest feeding process.</param>
        /// <param name="rate">The learning rate at which the weights are to be updated.</param>
        public override void BackPropagate(double[] error, double rate)
        {
            this.error2.FromArray(error, this.OutputDepth, this.OutputWidth, this.OutputHeight);
            this.BackPropagateFunc(rate);
        }

        /// <summary>Backpropagates the given error trough the network, updating its weights.</summary>
        /// <param name="error">The error to be backpropagated. It must refer to the latest feeding process.</param>
        /// <param name="rate">The learning rate at which the weights are to be updated.</param>
        public void BackPropagate(Image error, double rate)
        {
            this.error2.FromImage(error, this.OutputDepth, this.OutputWidth, this.OutputHeight);
            this.BackPropagateFunc(rate);
        }

        /// <summary>Backpropagates the given error trough the network, updating the weights.</summary>
        /// <param name="error">The error array to be backpropagated. It must refer to the latest feeding process.</param>
        /// <param name="inputError">The image to be written the error of the input image into.</param>
        /// <param name="rate">The learning rate at which the weights are to be updated.</param>
        public override void BackPropagate(double[] error, Image inputError, double rate)
        {
            this.BackPropagate(error, rate);
            inputError.FromImage(this.error2, this.InputDepth, this.InputWidth, this.InputHeight);
        }

        /// <summary>Backpropagates the given error trough the network updating its weights.</summary>
        /// <param name="error">The error to be backpropagated.</param>
        /// <param name="inputError">The image to be written the input error into.</param>
        /// <param name="rate">The learning rate at which the weights are to be updated.</param>
        public void BackPropagate(Image error, Image inputError, double rate)
        {
            this.BackPropagate(error, rate);
            inputError.FromImage(this.error2, this.InputDepth, this.inputWidth, this.InputHeight);
        }

        /// <summary>Feeds the given image trough the network.</summary>
        /// <param name="input">The image to be fed.</param>
        /// <param name="output">The array to be written the output into.</param>
        public override void Feed(Image input, double[] output)
        {
            this.first.FromImage(input, this.InputDepth, this.InputWidth, this.InputHeight);
            for (int i = 0; i < this.transformations.Length; i++)
            {
                this.transformations[i].Feed();
            }
            this.last.ToArray(output, this.OutputDepth, this.OutputWidth, this.OutputHeight);
        }

        private void FeedFunc(Image input)
        {
            this.first.FromImage(input, this.InputDepth, this.InputWidth, this.InputHeight);
            for (int i = 0; i < this.transformations.Length; i++)
            {
                this.transformations[i].Feed();
            }
        }

        /// <summary>Feeds an image trough the network.</summary>
        /// <param name="input">The image to be fed.</param>
        /// <param name="output">The image to be written the output into.</param>
        public void Feed(Image input, Image output)
        {
            this.FeedFunc(input);
            output.FromImage(this.last, this.OutputDepth, this.OutputWidth, this.OutputHeight);
        }

        /// <summary>Feeds an image trough the network and creates a copy of each image generated by the network during the feeding process.</summary>
        /// <param name="input">The image to be fed.</param>
        /// <param name="output">The array to be written the output into.</param>
        /// <param name="steps">The array to be written the generated images into.</param>
        public void Feed(Image input, double[] output, Image[] steps)
        {
            this.FeedFunc(input);
            this.last.ToArray(output, this.OutputDepth, this.OutputWidth, this.OutputHeight);
        }

        /// <summary>Feeds an image trough the network and exports each image generated by the network during the feeding process.</summary>
        /// <param name="input">The image to be fed.</param>
        /// <param name="outputPath">The path of the folder to be exported the files into.</param>
        /// <param name="output">The array to be written the output into.</param>
        public void Visualize(Image input, string outputPath, double[] output = null)
        {
            Image[] steps = new Image[this.Layers];
            this.Feed(input, output ?? new double[this.Outputs], steps);
            Image.NormalizeAll(steps);
            Directory.CreateDirectory(outputPath);
            input.Export(outputPath + "\\original");
            Image.SaveAllLayers(outputPath, steps, true);
        }

        /// <summary>Exports the images generated trough the latest feeding process.</summary>
        /// <param name="outputPath">The path of the folder to be exported the files into.</param>
        public void Visualize(string outputPath)
        {
            Image[] steps = new Image[this.Layers];
            for (int i = 0; i < this.transformations.Length; i++)
            {
                steps[i] = new Image(this.transformations[i].Output.Depth, this.transformations[i].Output.Width, this.transformations[i].Output.Height);
                steps[i].FromImage(this.transformations[i].Output);
            }
            Image.NormalizeAll(steps);
            Image.SaveAllLayers(outputPath, steps, true);
        }

        /// <summary>Feeds an image trough this network and gets an error array for the generated outputs.</summary>
        /// <param name="input">The image to be fed.</param>
        /// <param name="expected">The expected output image.</param>
        /// <param name="error">The image to be written the error into.</param>
        /// <returns>A value representing how big the error is.</returns>
        public double GetError(Image input, Image expected, Image error)
        {
            double retVal = 0;
            this.FeedFunc(input);
            for (int i = 0; i < this.OutputDepth; i++)
            {
                for (int j = 0; j < this.OutputWidth; j++)
                {
                    for (int k = 0; k < this.OutputHeight; k++)
                    {
                        error.Raw[i, j, k] = expected.Raw[i, j, k] - last.Raw[i, j, k];
                        retVal += error.Raw[i, j, k] * error.Raw[i, j, k];
                    }
                }
            }
            retVal = Math.Sqrt(retVal);
            return retVal;
        }
        
        [OnDeserialized]
        private void SetValuesOnDeserialized(StreamingContext context)
        {
            Image prev = this.first = new Image(this.InputDepth, this.InputWidth, this.InputHeight);
            int maxDepth = this.InputDepth;
            int maxWidth = this.InputWidth;
            int maxHeight = this.InputHeight;
            for (int i = 0; i < this.transformations.Length; i++)
            {
                this.transformations[i].Info.SizeAfter(prev.Depth, prev.Width, prev.Height, out int outDepth, out int outWidth, out int outHeight);
                maxDepth = Math.Max(maxDepth, outDepth);
                maxWidth = Math.Max(maxWidth, outWidth);
                maxHeight = Math.Max(maxHeight, outHeight);
                Image output = new Image(outDepth, outWidth, outHeight);
                this.transformations[i].SetLayers(prev, output);
                prev = output;
            }
            this.last = prev;
            this.error1 = new Image(maxDepth, maxWidth, maxHeight);
            this.error2 = new Image(maxDepth, maxWidth, maxHeight);
        }

        /// <summary>Exports this instance of the <code>PurelyConvolutionalNN</code> to a file.</summary>
        /// <param name="stream">The stream to be exported into.</param>
        public override void Save(Stream stream)
        {
            new DataContractJsonSerializer(typeof(PurelyConvolutionalNN)).WriteObject(stream, this);
        }
        
        /// <summary>Imports an instance of the <code>PurelyConvolutionalNN</code> from a stream.</summary>
        /// <param name="stream">The stream to be imported from.</param>
        /// <returns>The imported instance.</returns>
        public static PurelyConvolutionalNN Load(Stream stream)
        {
            return (PurelyConvolutionalNN)(new DataContractJsonSerializer(typeof(PurelyConvolutionalNN)).ReadObject(stream));
        }

        /// <summary>Loads an instance of the <code>PurelyConvolutionalNN</code> from a file.</summary>
        /// <param name="fileName">The name of the file to be read.</param>
        /// <returns>The read instance.</returns>
        public static PurelyConvolutionalNN Load(string fileName)
        {
            FileStream fs = new FileStream(fileName, FileMode.Open);
            PurelyConvolutionalNN retVal = PurelyConvolutionalNN.Load(fs);
            fs.Close();
            return retVal;
        }

        /// <summary>Copies this instance of <code>PurelyConvolutionalNN</code> into anothers.</summary>
        /// <param name="cnn">Instance to be copied into.</param>
        protected virtual void CloneTo(PurelyConvolutionalNN cnn)
        {
            cnn.first = new Image(this.InputDepth, this.InputWidth, this.InputHeight);
            cnn.transformations = new IImageTransformation[this.transformations.Length];
            Image prev = cnn.first;
            for (int i = 0; i < this.transformations.Length; i++)
            {
                cnn.transformations[i] = this.transformations[i].Clone(prev, prev = new Image(this.transformations[i].Output.Depth, this.transformations[i].Output.Width, this.transformations[i].Output.Height));
            }
            cnn.last = prev;
            cnn.error1 = new Image(this.error1.Depth, this.error1.Width, this.error1.Height);
            cnn.error2 = new Image(this.error1.Depth, this.error1.Width, this.error1.Height);
            cnn.outputs = this.Outputs;
            cnn.inputDepth = this.InputDepth;
            cnn.inputWidth = this.InputWidth;
            cnn.inputHeight = this.InputHeight;
        }

        /// <summary>Creates a copy of this instance of the <code>PurelyConvolutionalNN</code> class.</summary>
        /// <returns>The generated instance of the <code>PurelyConvolutionalNN</code> class.</returns>
        public override object Clone()
        {
            PurelyConvolutionalNN retVal = new PurelyConvolutionalNN();
            this.CloneTo(retVal);
            return retVal;
        }
    }
}
