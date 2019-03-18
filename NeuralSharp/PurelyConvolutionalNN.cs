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
    /// <summary>Represents a neural network which works on images.</summary>
    public class PurelyConvolutionalNN : Sequential<Image, IImagesLayer, IError<Image>>, IImagesLayer
    {
        private Image error1;
        private Image error2;
        private bool layersConnected;
        
        /// <summary>Either creates a siamese of the given <code>PurelyConvolutionalNN</code> class or clones it.</summary>
        /// <param name="original">The original instance to be created a siamese of or cloned.</param>
        /// <param name="siamese"><code>true</code> if a siamese is to be created, <code>false</code> if a clone is.</param>
        protected PurelyConvolutionalNN(PurelyConvolutionalNN original, bool siamese) : base(original, siamese)
        {
            int maxDepth = 0;
            int maxWidth = 0;
            int maxHeight = 0;
            foreach (IImagesLayer layer in original.Layers)
            {
                maxDepth = Math.Max(maxDepth, layer.OutputDepth);
                maxWidth = Math.Max(maxWidth, layer.OutputWidth);
                maxHeight = Math.Max(maxHeight, layer.OutputHeight);
            }
            this.error1 = new Image(maxDepth, maxWidth, maxHeight);
            this.error2 = new Image(maxDepth, maxWidth, maxHeight);
            this.layersConnected = false;
        }

        /// <summary>Creates an instance of the <code>PurelyConvolutionalNN</code> class.</summary>
        /// <param name="layers">The layers of the network.</param>
        /// <param name="createIO">Whether the input image and the output image of the network are to be created.</param>
        public PurelyConvolutionalNN(ICollection<IImagesLayer> layers, bool createIO = true) : base(layers.ToArray())
        {
            int maxDepth = 0;
            int maxWidth = 0;
            int maxHeight = 0;
            foreach (IImagesLayer layer in layers)
            {
                maxDepth = Math.Max(maxDepth, layer.OutputDepth);
                maxWidth = Math.Max(maxWidth, layer.OutputWidth);
                maxHeight = Math.Max(maxHeight, layer.OutputHeight);
            }
            this.error1 = new Image(maxDepth, maxWidth, maxHeight);
            this.error2 = new Image(maxDepth, maxWidth, maxHeight);
            this.layersConnected = false;
            if (createIO)
            {
                this.SetInputGetOutput(new Image(layers.ElementAt(0).InputDepth, layers.ElementAt(0).InputWidth, layers.ElementAt(0).InputHeight));
            }
        }

        /// <summary>The depth of the input.</summary>
        public int InputDepth
        {
            get { return this.FirstLayer.InputDepth; }
        }

        /// <summary>The width of the input.</summary>
        public int InputWidth
        {
            get { return this.FirstLayer.InputWidth; }
        }

        /// <summary>The height of the input.</summary>
        public int InputHeight
        {
            get { return this.FirstLayer.InputHeight; }
        }

        /// <summary>The depth of the output.</summary>
        public int OutputDepth
        {
            get { return this.LastLayer.OutputDepth; }
        }

        /// <summary>The width of the output.</summary>
        public int OutputWidth
        {
            get { return this.LastLayer.OutputWidth; }
        }

        /// <summary>The height of the output.</summary>
        public int OutputHeight
        {
            get { return this.LastLayer.OutputHeight; }
        }

        /// <summary>Whether the layers are connected.</summary>
        public bool LayersConnected
        {
            get { return this.layersConnected; }
        }

        /// <summary>Create an object which can be used as output error.</summary>
        /// <returns>The created image.</returns>
        protected override Image NewError()
        {
            return new Image(this.OutputDepth, this.OutputWidth, this.OutputHeight);
        }
        
        /// <summary>Feeds the layer forward.</summary>
        /// <param name="input">The image to be copied the input from.</param>
        /// <param name="output">The image to be copied the output into.</param>
        /// <param name="learning">Whether the layer is being used in a training session.</param>
        public override void Feed(Image input, Image output, bool learning = false)
        {
            this.Input.FromImage(input);
            this.Feed(learning);
            output.FromImage(this.Output);
        }

        /// <summary>Backpropagates the given error trough the network.</summary>
        /// <param name="outputError">The error to be backpropagated.</param>
        /// <param name="learning">Whether the layer is being used in a training session.</param>
        public override void BackPropagate(Image outputError, bool learning = true)
        {
            this.error2.FromImage(outputError);
            for (int i = this.Layers.Count - 1; i >= 0; i--)
            {
                this.Layers.ElementAt(i).BackPropagate(this.error2, this.error1, learning);
                Image aux = this.error1;
                this.error1 = this.error2;
                this.error2 = aux;
            }
        }

        /// <summary>Backpropagates the given error trough the network.</summary>
        /// <param name="outputError">The output error to be backpropagated.</param>
        /// <param name="inputError">The image to be written the input into.</param>
        /// <param name="learning">Whether the layer is being used in a training session.</param>
        public override void BackPropagate(Image outputError, Image inputError, bool learning)
        {
            this.BackPropagate(outputError, learning);
            inputError.FromImage(this.error2);
        }

        /// <summary>Gets the error of the network.</summary>
        /// <param name="output">The actual output.</param>
        /// <param name="expectedOutput">The expected output.</param>
        /// <param name="error">The image to be written the output error into.</param>
        /// <param name="errorFunction">The error function to be used.</param>
        /// <returns>The output error of the network.</returns>
        public override double GetError(Image output, Image expectedOutput, Image error, IError<Image> errorFunction)
        {
            return errorFunction.GetError(output, expectedOutput, error);
        }

        /// <summary>Feeds the network forward and gets its error.</summary>
        /// <param name="input">The image to be copied the input from.</param>
        /// <param name="expectedOutput">The expected output of the network.</param>
        /// <param name="error">The image to be written the error into.</param>
        /// <param name="errorFunction">The error function to be used.</param>
        /// <param name="learning">Whether the network is being used in a training session.</param>
        /// <returns>The error of the network.</returns>
        public override double FeedAndGetError(Image input, Image expectedOutput, Image error, IError<Image> errorFunction, bool learning)
        {
            this.Input.FromImage(input);
            this.Feed(learning);
            return this.GetError(this.Output, expectedOutput, error, errorFunction);
        }

        /// <summary>Creates a siamese of the network.</summary>
        /// <returns>The created instance of the <code>PurelyConvolutionalNN</code> class.</returns>
        public override ILayer<Image, Image> CreateSiamese()
        {
            return new PurelyConvolutionalNN(this, true);
        }

        /// <summary>Creates a clone of the network.</summary>
        /// <returns>The created instance of the <code>PurelyConvolutionalNN</code> class.</returns>
        public override ILayer<Image, Image> Clone()
        {
            return new PurelyConvolutionalNN(this, false);
        }

        /// <summary>Sets the input image and the output image of the network.</summary>
        /// <param name="input">The input to be set.</param>
        /// <param name="output">The output image to be set.</param>
        public override void SetInputAndOutput(Image input, Image output)
        {
            Image image = input;
            if (this.LayersConnected)
            {
                this.FirstLayer.SetInputAndOutput(input, this.FirstLayer.Output);
            }
            else
            {
                for (int i = 0; i < this.Layers.Count - 1; i++)
                {
                    image = this.Layers.ElementAt(i).SetInputGetOutput(image);
                }
            }
            this.LastLayer.SetInputAndOutput(image, output);
        }

        /// <summary>Sets the input image of the network and creates and sets an output image.</summary>
        /// <param name="input">The input image to be set.</param>
        /// <returns>The created output image.</returns>
        public override Image SetInputGetOutput(Image input)
        {
            Image retVal = new Image(this.OutputDepth, this.OutputWidth, this.OutputHeight);
            this.SetInputAndOutput(input, retVal);
            return retVal;
        }

        /// <summary>Adds a top layer.</summary>
        /// <param name="layer">The layer to be added.</param>
        protected override void AddTopLayer(IImagesLayer layer)
        {
            base.AddTopLayer(layer);
            this.layersConnected = false;
        }

        /// <summary>Adds a bottom layer.</summary>
        /// <param name="layer">The layer to be added.</param>
        protected override void AddBottomLayer(IImagesLayer layer)
        {
            base.AddBottomLayer(layer);
            this.layersConnected = false;
        }
    }
}
