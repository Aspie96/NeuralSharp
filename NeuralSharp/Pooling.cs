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
    /// <summary>Represents a pooling layer.</summary>
    public abstract class Pooling : IImagesLayer
    {
        private Image input;
        private Image output;
        private int inputDepth;
        private int inputWidth;
        private int inputHeight;
        private int outputDepth;
        private int outputWidth;
        private int outputHeight;
        private int xScale;
        private int yScale;
        
        /// <summary>Either creates a siamese of the given <code>Pooling</code> instance or clones it.</summary>
        /// <param name="original">The original instance to be created a siamese of or cloned.</param>
        /// <param name="siamese"><code>true</code> if a siamese is to be crated, <code>false</code> if a clone is.</param>
        protected Pooling(Pooling original, bool siamese)
        {
            this.inputDepth = original.InputDepth;
            this.inputWidth = original.InputWidth;
            this.inputHeight = original.inputHeight;
            this.outputDepth = original.OutputDepth;
            this.outputWidth = original.OutputWidth;
            this.outputHeight = original.OutputHeight;
            this.xScale = original.XScale;
            this.yScale = original.YScale;
        }

        /// <summary>Creates an instance of the <code>Pooling</code> class.</summary>
        /// <param name="inputDepth">The depth of the input.</param>
        /// <param name="inputWidth">The width of the input.</param>
        /// <param name="inputHeight">The height of the input.</param>
        /// <param name="xScale">The X scale factor.</param>
        /// <param name="yScale">The Y scale factor.</param>
        /// <param name="createIO">Whether the input image and the output image are to be created.</param>
        public Pooling(int inputDepth, int inputWidth, int inputHeight, int xScale, int yScale, bool createIO = false)
        {
            this.inputDepth = inputDepth;
            this.inputWidth = inputWidth;
            this.inputHeight = inputHeight;
            this.outputDepth = inputDepth;
            this.outputWidth = inputWidth / xScale;
            this.outputHeight = inputHeight / yScale;
            if (createIO)
            {
                this.input = new Image(inputDepth, inputWidth, inputHeight);
                this.output = new Image(outputDepth, outputWidth, outputHeight);
            }
            this.xScale = xScale;
            this.yScale = yScale;
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

        /// <summary>The input depth of the layer.</summary>
        public int InputDepth
        {
            get { return this.inputDepth; }
        }

        /// <summary>The input width of the layer.</summary>
        public int InputWidth
        {
            get { return this.inputWidth; }
        }

        /// <summary>The input height of the layer.</summary>
        public int InputHeight
        {
            get { return this.inputHeight; }
        }

        /// <summary>The output depth of the layer.</summary>
        public int OutputDepth
        {
            get { return this.outputDepth; }
        }

        /// <summary>The output width of the layer.</summary>
        public int OutputWidth
        {
            get { return this.outputWidth; }
        }

        /// <summary>The output height of the layer.</summary>
        public int OutputHeight
        {
            get { return this.outputHeight; }
        }
        
        /// <summary>The X scale factor of the layer.</summary>
        public int XScale
        {
            get { return this.xScale; }
        }
        
        /// <summary>The Y scale factor of the layer.</summary>
        public int YScale
        {
            get { return this.yScale; }
        }
        
        /// <summary>The amount of parameters of the layer.</summary>
        public int Parameters
        {
            get { return 0; }
        }

        /// <summary>Feeds the layer forward.</summary>
        /// <param name="learning">Whether the layer is being used for training purposes.</param>
        public abstract void Feed(bool learning = false);

        /// <summary>Backpropagates an error trough the layer.</summary>
        /// <param name="outputError">The output error to be backpropagated.</param>
        /// <param name="inputError">The image to be written the input error into.</param>
        /// <param name="learning">Whether the layer is being used in a training session.</param>
        public abstract void BackPropagate(Image outputError, Image inputError, bool learning);
        
        /// <summary>Creates a siamese of the layer.</summary>
        /// <returns>The created siamese.</returns>
        public abstract IUntypedLayer CreateSiamese();

        /// <summary>Creates a siamese of the layer.</summary>
        /// <returns>The created instance of the <code>Pooling</code> class.</returns>
        public abstract IUntypedLayer Clone();
        
        /// <summary>Updates the weights of the layer.</summary>
        /// <param name="rate">The learning rate to be used.</param>
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
    }
}
