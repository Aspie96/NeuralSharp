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
    /// <summary>Represents an image dropout layer.</summary>
    public class ImageDropoutLayer : IImagesLayer
    {
        private Image input;
        private Image output;
        private int depth;
        private int width;
        private int height;
        private bool[] dropped;
        private double dropChance;
        private object siameseID;

        /// <summary>Either creates a siamese of the given <code>ImageDropoutLayer</code> instance or clones it.</summary>
        /// <param name="original">The original instance to be created a siamese of or cloned.</param>
        /// <param name="siamese"><code>true</code> if a siamese is to be created, <code>false</code> if a clone is.</param>
        protected ImageDropoutLayer(ImageDropoutLayer original, bool siamese)
        {
            this.depth = original.Depth;
            this.width = original.width;
            this.height = original.Height;
            this.dropped = Backbone.CreateArray<bool>(depth * width * height);
            this.dropChance = original.DropChance;
            if (siamese)
            {
                this.siameseID = original.SiameseID;
            }
            else
            {
                this.siameseID = new object();
            }
        }

        /// <summary>Creates an instance of the <code>ImageDropoutLayer</code> class.</summary>
        /// <param name="depth">The depth of the layer.</param>
        /// <param name="width">The width of the layer.</param>
        /// <param name="height">The height of the layer.</param>
        /// <param name="dropChance">The dropout chance of the layer.</param>
        /// <param name="createIO">Whether the input image and the output image of the layer are to be created.</param>
        public ImageDropoutLayer(int depth, int width, int height, double dropChance, bool createIO = false)
        {
            this.depth = depth;
            this.width = width;
            this.height = height;
            if (createIO)
            {
                this.input = new Image(depth, width, height);
                this.output = new Image(depth, width, height);
            }
            this.dropped = Backbone.CreateArray<bool>(depth * width * height);
            this.dropChance = dropChance;
            this.siameseID = new object();
        }

        /// <summary>The input image of the layer.</summary>
        public Image Input
        {
            get { return this.input; }
        }

        /// <summary>The output image of the layer.</summary>
        public Image Output
        {
            get { return this.output; }
        }

        /// <summary>The depth of the input of the layer.</summary>
        public int InputDepth
        {
            get { return this.depth; }
        }

        /// <summary>The width of the input of the layer.</summary>
        public int InputWidth
        {
            get { return this.width; }
        }

        /// <summary>The height of the input of the layer.</summary>
        public int InputHeight
        {
            get { return this.height; }
        }

        /// <summary>The depth of the input of the layer.</summary>
        public int OutputDepth
        {
            get { return this.depth; }
        }

        /// <summary>The width of the input of the layer.</summary>
        public int OutputWidth
        {
            get { return this.width; }
        }

        /// <summary>The height of the input of the layer.</summary>
        public int OutputHeight
        {
            get { return this.height; }
        }
        
        /// <summary>The depth of the layer.</summary>
        public int Depth
        {
            get { return this.depth; }
        }

        /// <summary>The width of the layer.</summary>
        public int Width
        {
            get { return this.width; }
        }

        /// <summary>The height of the layer.</summary>
        public int Height
        {
            get { return this.height; }
        }

        /// <summary>The dropout chance of the layer.</summary>
        public double DropChance
        {
            get { return this.dropChance; }
        }
        
        /// <summary>The amount of parameters of the layer.</summary>
        public int Parameters
        {
            get { return 0; }
        }

        /// <summary>The siamese identifier of the layer.</summary>
        public object SiameseID
        {
            get { return this.siameseID; }
        }

        /// <summary>Feeds the layer forward.</summary>
        /// <param name="learning">Whether the layer is being used in a training session.</param>
        public void Feed(bool learning = false)
        {
            Backbone.ApplyImageDropout(this.Input.Raw, this.Output.Raw, this.Depth, this.Width, this.Height, this.dropped, this.DropChance, learning);
        }

        /// <summary>Backpropagates the given error trough the network.</summary>
        /// <param name="outputError">The output error to be backpropagated.</param>
        /// <param name="inputError">The image to be written the input error into.</param>
        /// <param name="learning">Whether the layer is being used in a training session.</param>
        public void BackPropagate(Image outputError, Image inputError, bool learning)
        {
            Backbone.BackpropagateImageDropout(this.Input.Raw, this.Output.Raw, this.Depth, this.Width, this.Height, this.dropped, this.DropChance, learning, outputError.Raw, outputError.Depth, outputError.Width, outputError.Height, inputError.Raw, inputError.Depth, inputError.Width, inputError.Height);
        }

        /// <summary>Updates the weights of the layer. Does nothing.</summary>
        /// <param name="rate">The lerning reate to be used.</param>
        /// <param name="momentum">The momentum to be used.</param>
        public void UpdateWeights(double rate, double momentum = 0.0) { }
        
        /// <summary>Sets the input image and the output image of the layer.</summary>
        /// <param name="input">The input image to be set.</param>
        /// <param name="output">The output image to be set.</param>
        public void SetInputAndOutput(Image input, Image output)
        {
            this.input = input;
            this.output = output;
        }

        /// <summary>Sets the input image of the layer and creates and sets the output image.</summary>
        /// <param name="input">The input image to be set.</param>
        /// <returns>The created output image.</returns>
        public Image SetInputGetOutput(Image input)
        {
            this.input = input;
            return this.output = new Image(this.OutputDepth, this.OutputWidth, this.OutputHeight);
        }

        /// <summary>Creates a siamese of the layer.</summary>
        /// <returns>The created instance of the <code>ImageDropoutLayer</code> class.</returns>
        public virtual ILayer<Image, Image> CreateSiamese()
        {
            return new ImageDropoutLayer(this, true);
        }

        /// <summary>Creates a clone.</summary>
        /// <returns>The created instance of the <code>ImageDropoutLayer</code> class.</returns>
        public virtual ILayer<Image, Image> Clone()
        {
            return new ImageDropoutLayer(this, false);
        }

        /// <summary>Counts the amount of parameters of the layer.</summary>
        /// <param name="siameseIDs">The siamese identifiers to be excluded.</param>
        /// <returns>The amount of parameters of the layer.</returns>
        public int CountParameters(List<object> siameseIDs)
        {
            if (!siameseIDs.Contains(this.SiameseID))
            {
                siameseIDs.Add(this.SiameseID);
            }
            return 0;
        }
    }
}
