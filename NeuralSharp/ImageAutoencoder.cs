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
using System.Text;
using System.Threading.Tasks;

namespace NeuralSharp
{
    /// <summary>Represents an autoencoder.</summary>
    public class ImageAutoencoder : Autoencoder<Image, IError<Image>>
    {
        private int depth;
        private int width;
        private int height;

        /// <summary>Either creates a siamese of the given <code>ImageAutoencoder</code> class or clones it.</summary>
        /// <param name="original">The original instance to be created a siamese of or cloned.</param>
        /// <param name="siamese"><code>true</code> if a siamese is to be created, <code>false</code> if a clone is.</param>
        protected ImageAutoencoder(ImageAutoencoder original, bool siamese) : base(original, siamese)
        {
            this.depth = original.depth;
            this.width = original.width;
            this.height = original.height;
        }

        /// <summary>Creates an instance of the <code>ImageAutoencoder</code> class.</summary>
        /// <param name="encoder">The first part of the autoencoder.</param>
        /// <param name="decoder">The second part of the autoencoder.</param>
        /// <param name="codeSize">The size of the output of the encoder and of the input of the decoder.</param>
        public ImageAutoencoder(ConvolutionalNN encoder, DeConvolutionalNN decoder, int codeSize, bool createIO = true) : base(encoder, decoder, codeSize)
        {
            this.depth = encoder.InputDepth;
            this.width = encoder.InputWidth;
            this.height = encoder.InputHeight;
            if (createIO)
            {
                this.SetInputGetOutput(new Image(encoder.InputDepth, encoder.InputWidth, encoder.InputHeight));
            }
        }

        /// <summary>The depth of the input and the output of the layer.</summary>
        public int Depth
        {
            get { return this.depth; }
        }

        /// <summary>The width of the input and the output of the layer.</summary>
        public int Width
        {
            get { return this.width; }
        }

        /// <summary>The height of the input and the output of the layer.</summary>
        public int Height
        {
            get { return this.height; }
        }

        /// <summary>Backpropagates the given error trough the autoencoder.</summary>
        /// <param name="outputError">The output error to be backpropagated.</param>
        /// <param name="learning">Whether the autoencoder is being used in a training session.</param>
        public override void BackPropagate(Image outputError, bool learning = true)
        {
            this.Decoder.BackPropagate(outputError, this.Error, learning);
            ((ConvolutionalNN)this.Encoder).BackPropagate(this.Error, learning);
        }
        
        /// <summary>Creates a clone of the autoencoder.</summary>
        /// <returns>The created <code>ImageAutoencoder</code> instance.</returns>
        public override ILayer<Image, Image> Clone()
        {
            return new ImageAutoencoder(this, false);
        }

        /// <summary>Creates a siamese of the autoencoder.</summary>
        /// <returns>The created <code>ImageAutoencoder</code> instance.</returns>
        public override ILayer<Image, Image> CreateSiamese()
        {
            return new ImageAutoencoder(this, true);
        }

        /// <summary>Feeds the autoencoder forward.</summary>
        /// <param name="input">The image to be copied the input from.</param>
        /// <param name="output">The image to be copied the output into.</param>
        /// <param name="learning">Whether the autoencoder is being used in a training session.</param>
        public override void Feed(Image input, Image output, bool learning = false)
        {
            this.Input.FromImage(input);
            this.Feed(learning);
            output.FromImage(this.Output);
        }

        /// <summary>Feeds the autoencoder forward and gets its error, given its expected output.</summary>
        /// <param name="input">The image to be copied the input from.</param>
        /// <param name="expectedOutput">The expected output of the autoencoder.</param>
        /// <param name="error">The image to be copied the error into.</param>
        /// <param name="errorFunction">The error function to be used.</param>
        /// <param name="learning">Whether the autoencoder is being used in a training session.</param>
        /// <returns>The error of the autoencoder.</returns>
        public override float FeedAndGetError(Image input, Image expectedOutput, Image error, IError<Image> errorFunction, bool learning)
        {
            this.Input.FromImage(input);
            this.Feed(learning);
            return this.GetError(this.Output, expectedOutput, error, errorFunction);
        }

        /// <summary>Gets the error of the autoencoder, given its actual output and its expected output.</summary>
        /// <param name="output">The actual output of the autoencoder.</param>
        /// <param name="expectedOutput">The expected output of the autoencoder.</param>
        /// <param name="error">The image to be written the error into.</param>
        /// <param name="errorFunction">The error function to be used.</param>
        /// <returns>The error of the autoencoder.</returns>
        public override float GetError(Image output, Image expectedOutput, Image error, IError<Image> errorFunction)
        {
            return errorFunction.GetError(output, expectedOutput, error);
        }

        /// <summary>Creates an image which can be used as output error.</summary>
        /// <returns>The created image.</returns>
        protected override Image NewError()
        {
            return new Image(this.Depth, this.Width, this.Height);
        }
    }
}
