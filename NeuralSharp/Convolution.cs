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
    /// <summary>Represents a convolution operation.</summary>
    public class Convolution : IImagesLayer
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
        private float[] biases;
        private float[] biasesDeltas;
        private float[,] kernelGradients;
        private float[,] kernelMomentums;
        private float[] biasesMomentums;
        private int kernelSide;
        private int stride;
        private int padding;

        /// <summary>Either creates a siamese of the given <code>Convolution</code> instance or clones it.</summary>
        /// <param name="original">The original instance to be created a siamese of or cloned.</param>
        /// <param name="siamese"><code>true</code> if a siamese is to be created, <code>false</code> otherwise.</param>
        protected Convolution(Convolution original, bool siamese)
        {
            this.inputDepth = original.InputDepth;
            this.inputWidth = original.InputWidth;
            this.inputHeight = original.inputHeight;
            this.outputDepth = original.OutputDepth;
            this.outputWidth = original.OutputWidth;
            this.outputHeight = original.OutputHeight;
            this.padding = original.Parameters;
            this.kernelSide = original.kernelSide;
            this.stride = original.stride;
            if (siamese)
            {
                this.biases = original.biases;
                this.biasesDeltas = original.biasesDeltas;
                this.biasesMomentums = original.biasesMomentums;
                this.kernelFilters = original.kernelFilters;
                this.kernelGradients = original.kernelGradients;
                this.kernelMomentums = original.kernelMomentums;
            }
            else
            {
                this.biases = Backbone.CreateArray<float>(original.OutputDepth);
                this.biasesDeltas = Backbone.CreateArray<float>(original.OutputDepth);
                this.biasesMomentums = Backbone.CreateArray<float>(original.OutputDepth);
                this.kernelFilters = Backbone.CreateArray<float>(original.OutputDepth, inputDepth * kernelSide * kernelSide);
                Backbone.RandomizeMatrix(this.kernelFilters, original.OutputDepth, inputDepth * kernelSide * kernelSide, 2.0 / (inputDepth * kernelSide * kernelSide + outputDepth));
                this.kernelGradients = Backbone.CreateArray<float>(original.OutputDepth, inputDepth * kernelSide * kernelSide);
                this.kernelMomentums = Backbone.CreateArray<float>(original.OutputDepth, inputDepth * kernelSide * kernelSide);
            }
        }
        
        /// <summary>Creates an instance of the <code>Convolution</code> class.</summary>
        /// <param name="inputDepth">The depth of the input of the layer.</param>
        /// <param name="inputWidth">The width of the input of the layer.</param>
        /// <param name="inputHeight">The height of the input of the layer.</param>
        /// <param name="depth">The depth of the output of the layer.</param>
        /// <param name="kernelSide">The side of the kernel to be used.</param>
        /// <param name="stride">The stride to be used.</param>
        /// <param name="padding"><code>true</code> if padding is to be used, <code>false</code> otherwise.</param>
        /// <param name="createIO">Whether the input image and the output image of the layer are to be created.</param>
        public Convolution(int inputDepth, int inputWidth, int inputHeight, int depth, int kernelSide, int stride, bool padding, bool createIO = false)
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
                this.output = new Image(outputDepth, outputWidth, outputHeight);
            }
            this.biases = Backbone.CreateArray<float>(depth);
            this.biasesDeltas = Backbone.CreateArray<float>(depth);
            this.biasesMomentums = Backbone.CreateArray<float>(depth);
            this.kernelFilters = Backbone.CreateArray<float>(depth, inputDepth * kernelSide * kernelSide);
            Backbone.RandomizeMatrix(this.kernelFilters, depth, inputDepth * kernelSide * kernelSide, 2.0 / (inputDepth * kernelSide * kernelSide + outputDepth));
            this.kernelGradients = Backbone.CreateArray<float>(depth, inputDepth * kernelSide * kernelSide);
            this.kernelMomentums = Backbone.CreateArray<float>(depth, inputDepth * kernelSide * kernelSide);
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

        /// <summary>The input image of the layer.</summary>
        public Image Input
        {
            get{ return this.input; }
        }
        
        /// <summary>The output image of the layer.</summary>
        public Image Output
        {
            get { return this.output; }
        }

        /// <summary>The depth of the input of the layer.</summary>
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
        
        /// <summary>The side of the used kernel.</summary>
        public int KernelSide
        {
            get { return this.kernelSide; }
        }

        /// <summary>The used stride.</summary>
        public int Stride
        {
            get { return this.stride; }
        }
        
        /// <summary>The amount of parameters of the layer.</summary>
        public int Parameters
        {
            get { return this.kernelFilters.Length * this.KernelSide * this.KernelSide; }
        }
        
        /// <summary>The activation function used by the layer.</summary>
        /// <param name="input">The input of the activation function.</param>
        /// <returns>The output of the activation function.</returns>
        protected float ActivationFunction(float input)
        {
            return (input > 0.0F ? input : input * 0.0F);
        }

        /// <summary>The derivative of the activation function relative to the output.</summary>
        /// <param name="output">The output of the activation function.</param>
        /// <returns>The derivative of the activation function.</returns>
        protected float ActivationDerivative(float output)
        {
            return (output > 0.0F ? 1.0F : 0.0F);
        }

        /// <summary>Feeds the layer forward.</summary>
        /// <param name="learning">Whether the layer is being used in a training session. Unused.</param>
        public void Feed(bool learning = false)
        {
            Backbone.ApplyConvolution(this.biases, this.input.Raw, this.InputDepth, this.InputWidth, this.InputHeight, this.output.Raw, this.OutputDepth, this.OutputWidth, this.OutputHeight, this.kernelFilters, this.kernelSide, this.Stride, 1, this.padding, this.ActivationFunction);
        }

        /// <summary>Backpropagates the given error trough the layer.</summary>
        /// <param name="outputError">The output error to be backpropagated.</param>
        /// <param name="inputError">The image to be written the output error into.</param>
        /// <param name="learning">Whether the layer is being used in a training session.</param>
        public void BackPropagate(Image outputError, Image inputError, bool learning)
        {
            Backbone.BackpropagateConvolution(this.biases, this.biasesDeltas, this.input.Raw, this.InputDepth, this.InputWidth, this.InputHeight, this.output.Raw, this.OutputDepth, this.OutputWidth, this.OutputHeight, this.kernelFilters, this.kernelSide, this.Stride, 1, this.padding, outputError.Raw, outputError.Depth, outputError.Width, outputError.Height, inputError.Raw, inputError.Depth, inputError.Width, inputError.Height, this.ActivationDerivative, this.kernelGradients, learning);
        }

        /// <summary>Updates the weights of the layer.</summary>
        /// <param name="rate">The learning rate to be used.</param>
        /// <param name="momentum">The momentum to be used.</param>
        public void UpdateWeights(double rate, double momentum = 0.0)
        {
            Backbone.UpdateConvolution(this.biasesDeltas, this.biasesMomentums, this.biases, this.kernelFilters, this.kernelGradients, this.kernelMomentums, this.InputDepth, this.InputWidth, this.InputHeight, this.OutputDepth, this.OutputWidth, this.OutputHeight, this.KernelSide, (float)rate, (float)momentum);
        }

        /// <summary>Sets the input image and the output image of the layer.</summary>
        /// <param name="input">The input image to be set.</param>
        /// <param name="output">The output image to be set.</param>
        public void SetInputAndOutput(Image input, Image output)
        {
            this.input = input;
            this.output = output;
        }

        /// <summary>Sets the input image of the layer and creates and sets an output image.</summary>
        /// <param name="input">The input image to be set.</param>
        /// <returns>The created output image.</returns>
        public Image SetInputGetOutput(Image input)
        {
            this.input = input;
            return this.output = new Image(this.OutputDepth, this.OutputWidth, this.OutputHeight);
        }
        
        /// <summary>Creates a siamese of the layer.</summary>
        /// <returns>The created instance of the <code>Convolution</code> class.</returns>
        public virtual IUntypedLayer CreateSiamese()
        {
            return new Convolution(this, true);
        }

        /// <summary>Creates a clone of the layer.</summary>
        /// <returns>The created instance of the <code>Convolution</code> class.</returns>
        public virtual IUntypedLayer Clone()
        {
            return new Convolution(this, false);
        }
    }
}
