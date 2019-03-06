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

        public Image Input
        {
            get { return this.input; }
        }

        public Image Output
        {
            get { return this.output; }
        }

        public int InputDepth
        {
            get { return this.inputDepth; }
        }

        public int InputWidth
        {
            get { return this.inputWidth; }
        }

        public int InputHeight
        {
            get { return this.inputHeight; }
        }

        public int OutputDepth
        {
            get { return this.outputDepth; }
        }

        public int OutputWidth
        {
            get { return this.outputWidth; }
        }

        public int OutputHeight
        {
            get { return this.outputHeight; }
        }

        public int KernelSide
        {
            get { return this.kernelSide; }
        }

        public int Stride
        {
            get { return this.stride; }
        }

        public int Parameters
        {
            get { return this.kernelFilters.Length * this.KernelSide * this.KernelSide; }
        }
        
        protected float ActivationFunction(float input)
        {
            return (input > 0.0F ? input : 0.0F);
        }

        protected float ActivationDerivative(float output)
        {
            return (output > 0.0F ? 1.0F : 0.0F);
        }

        public void Feed(bool learning = false)
        {
            Backbone.RandomizeMatrix(this.kernelFilters, this.OutputDepth, inputDepth * kernelSide * kernelSide, 2.0 / (inputDepth * kernelSide * kernelSide));
            Backbone.ApplyConvolution(null, this.input.Raw, this.InputDepth, this.InputWidth, this.InputHeight, this.output.Raw, this.OutputDepth, this.OutputWidth, this.OutputHeight, this.kernelFilters, this.kernelSide, this.Stride, 1, this.padding, this.ActivationFunction);
        }

        public void BackPropagate(Image outputError, Image inputError, bool learning)
        {
            Backbone.BackpropagateConvolution(null, null, this.input.Raw, this.InputDepth, this.InputWidth, this.InputHeight, this.output.Raw, this.OutputDepth, this.OutputWidth, this.OutputHeight, this.kernelFilters, this.kernelSide, this.Stride, 1, this.padding, outputError.Raw, outputError.Depth, outputError.Width, outputError.Height, inputError.Raw, inputError.Depth, inputError.Width, inputError.Height, this.ActivationDerivative, null, false);
        }

        public void UpdateWeights(double rate, double momentum = 0.0) { }

        public void SetInputAndOutput(Image input, Image output)
        {
            this.input = input;
            this.output = output;
        }

        public Image SetInputGetOutput(Image input)
        {
            this.input = input;
            return this.output = new Image(this.OutputDepth, this.OutputWidth, this.OutputHeight);
        }
        
        public virtual IUntypedLayer CreateSiamese()
        {
            return new RandomConvolution(this, true);
        }

        public virtual IUntypedLayer Clone()
        {
            return new RandomConvolution(this, false);
        }
    }
}
