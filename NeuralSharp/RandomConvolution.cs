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
    /// <summary>Represents a random convolution.</summary>
    public class RandomConvolution : IImagesLayer
    {
        private Image input;
        private Image output;
        private int inputDepth;
        private int inputWidth;
        private int inputHeight;
        private int outputDepth;
        private int outputWidth;
        private int outputHeight;
        private float[,] kernelFilters;
        private int kernelSide;
        private int stride;
        private int padding;

        /// <summary>Either creates a siamese of the given <code>RandomConvolution</code> instance or clones it.</summary>
        /// <param name="original">The original instance to be created a siamese of or cloned.</param>
        /// <param name="siamese"><code>true</code> if a siamese is to be created, <code>false</code> if a clone is.</param>
        protected RandomConvolution(RandomConvolution original, bool siamese)
        {
            this.inputDepth = original.InputDepth;
            this.inputWidth = original.InputWidth;
            this.inputHeight = original.InputHeight;
            this.outputDepth = original.OutputDepth;
            this.outputWidth = original.OutputWidth;
            this.outputHeight = original.OutputHeight;
            this.padding = original.padding;
            this.kernelFilters = Backbone.CreateArray<float>(original.OutputDepth, original.InputDepth * original.KernelSide * original.KernelSide);
            this.kernelSide = original.KernelSide;
            this.stride = original.Stride;
        }

        /// <summary>Creates an instance of the <code>RandomConvolution</code> class.</summary>
        /// <param name="inputDepth">The input depth.</param>
        /// <param name="inputWidth">The input width.</param>
        /// <param name="inputHeight">The input height.</param>
        /// <param name="depth">The output depth.</param>
        /// <param name="kernelSide">The kernel side.</param>
        /// <param name="stride">The stride.</param>
        /// <param name="padding">Whether padding is to be used.</param>
        /// <param name="createIO">Whether the input image and the output image of the layer are to be created.</param>
        public RandomConvolution(int inputDepth, int inputWidth, int inputHeight, int depth, int kernelSide, int stride, bool padding, bool createIO = false)
        {
            this.inputDepth = inputDepth;
            this.inputWidth = inputWidth;
            this.inputHeight = inputHeight;
            this.outputDepth = depth;
            if (padding)
            {
                this.outputWidth = inputWidth;
                this.outputHeight = inputHeight;
                this.padding = (kernelSide - 1) / 2;
            }
            else
            {
                this.outputWidth = inputWidth - kernelSide + 1;
                this.outputHeight = inputHeight - kernelSide + 1;
                this.padding = 0;
            }
            if (createIO)
            {
                this.input = new Image(inputDepth, inputWidth, inputHeight);
                this.output = new Image(outputWidth, outputWidth, outputHeight);
            }
            this.kernelFilters = Backbone.CreateArray<float>(depth, inputDepth * kernelSide * kernelSide);
            this.kernelSide = kernelSide;
            this.stride = stride;
        }
        
        private float[] SerKernelFilters
        {
            get
            {
                float[] retVal = new float[this.OutputDepth * this.InputDepth * this.KernelSide * this.KernelSide];
                Backbone.MatrixToArray(this.kernelFilters, this.OutputDepth, this.InputDepth * this.KernelSide * this.KernelSide, retVal, 0);
                return retVal;
            }

            set
            {
                this.kernelFilters = Backbone.CreateArray<float>(this.OutputDepth, this.InputDepth * this.KernelSide * this.KernelSide);
                Backbone.ArrayToMatrix(value, 0, this.kernelFilters, this.OutputDepth, this.InputDepth * this.KernelSide * this.KernelSide);
            }
        }

        /// <summary>The input image.</summary>
        public Image Input
        {
            get { return this.input; }
        }

        /// <summary>The output image.</summary>
        public Image Output
        {
            get { return this.output; }
        }

        /// <summary>The depht of the input of the layer.</summary>
        public int InputDepth
        {
            get { return this.inputDepth; }
        }

        /// <summary>The width of the input of the layer.</summary>
        public int InputWidth
        {
            get { return this.inputWidth; }
        }

        /// <summary>The height of the input of the layer.</summary>
        public int InputHeight
        {
            get { return this.inputHeight; }
        }

        /// <summary>The depth of the output of the layer.</summary>
        public int OutputDepth
        {
            get { return this.outputDepth; }
        }

        /// <summary>The width of the output of the layer.</summary>
        public int OutputWidth
        {
            get { return this.outputWidth; }
        }

        /// <summary>The height of the output of the layer.</summary>
        public int OutputHeight
        {
            get { return this.outputHeight; }
        }

        /// <summary>The kernel side.</summary>
        public int KernelSide
        {
            get { return this.kernelSide; }
        }

        /// <summary>The stride.</summary>
        public int Stride
        {
            get { return this.stride; }
        }

        /// <summary>The parameters.</summary>
        public int Parameters
        {
            get { return this.kernelFilters.Length * this.KernelSide * this.KernelSide; }
        }
        
        /// <summary>The activation function.</summary>
        /// <param name="input">The input.</param>
        /// <returns>The output.</returns>
        protected float ActivationFunction(float input)
        {
            return (input > 0.0F ? input : 0.0F);
        }

        /// <summary>The activation function derivative.</summary>
        /// <param name="output">The output of the activation function.</param>
        /// <returns>The derivative.</returns>
        protected float ActivationDerivative(float output)
        {
            return (output > 0.0F ? 1.0F : 0.0F);
        }

        /// <summary>Feeds the layer forward.</summary>
        /// <param name="learning">Whether the layer is being used during a training session.</param>
        public void Feed(bool learning = false)
        {
            Backbone.RandomizeMatrix(this.kernelFilters, this.OutputDepth, inputDepth * kernelSide * kernelSide, 2.0 / (inputDepth * kernelSide * kernelSide));
            Backbone.ApplyConvolution(null, this.input.Raw, this.InputDepth, this.InputWidth, this.InputHeight, this.output.Raw, this.OutputDepth, this.OutputWidth, this.OutputHeight, this.kernelFilters, this.kernelSide, this.Stride, 1, this.padding, this.ActivationFunction);
        }

        /// <summary>Backpropagates the error trough the layer.</summary>
        /// <param name="outputError">The output error to be backpropagated.</param>
        /// <param name="inputError">The image to be written the input error into.</param>
        /// <param name="learning">Whether the layer is being used in a training session.</param>
        public void BackPropagate(Image outputError, Image inputError, bool learning)
        {
            Backbone.BackpropagateConvolution(null, null, this.input.Raw, this.InputDepth, this.InputWidth, this.InputHeight, this.output.Raw, this.OutputDepth, this.OutputWidth, this.OutputHeight, this.kernelFilters, this.kernelSide, this.Stride, 1, this.padding, outputError.Raw, outputError.Depth, outputError.Width, outputError.Height, inputError.Raw, inputError.Depth, inputError.Width, inputError.Height, this.ActivationDerivative, null, false);
        }

        /// <summary>Updates the weights of the layer. Does nothing.</summary>
        /// <param name="rate">The learning rate to be used.</param>
        /// <param name="momentum">The momentum to be used.</param>
        public void UpdateWeights(double rate, double momentum = 0.0) { }

        /// <summary>Sets the input image and the output image of the layer.</summary>
        /// <param name="input">The input image.</param>
        /// <param name="output">The output image to be set.</param>
        public void SetInputAndOutput(Image input, Image output)
        {
            this.input = input;
            this.output = output;
        }

        /// <summary>Sets the input image of the layer and creates and sets the output image.</summary>
        /// <param name="input">The input image.</param>
        /// <returns>The created output image.</returns>
        public Image SetInputGetOutput(Image input)
        {
            this.input = input;
            return this.output = new Image(this.OutputDepth, this.OutputWidth, this.OutputHeight);
        }
        
        /// <summary>Creates a siamese of the layer.</summary>
        /// <returns>The created siamese.</returns>
        public virtual IUntypedLayer CreateSiamese()
        {
            return new RandomConvolution(this, true);
        }

        /// <summary>Creates a clone of the layer.</summary>
        /// <returns>The created clone.</returns>
        public virtual IUntypedLayer Clone()
        {
            return new RandomConvolution(this, false);
        }
    }
}
