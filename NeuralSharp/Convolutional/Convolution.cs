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
using System.Runtime.Serialization;

namespace NeuralNetwork.Convolutional
{
    /// <summary>Represents a convolutional layer in a convolutional neural network.</summary>
    [DataContract]
    public class Convolution : IImageTransformation
    {
        private Image input;
        private Image output;
        [DataMember]
        private Kernel[] kernels;
        [DataMember]
        private int kernelSide;
        private int stride;
        private bool padding;

        /// <summary>Empty constructor. It does not actually initialize the fields.</summary>
        protected Convolution() { }

        /// <summary>Creates a new instance of the <code>Convolution</code> class.</summary>
        /// <param name="input">The input image of the convolutional layer.</param>
        /// <param name="output">The output image of the convolutional layer.</param>
        /// <param name="depth">The depth of the convolutional layer.</param>
        /// <param name="kernelSide">The lenght of the side of each kernel in the convolutional layer.</param>
        /// <param name="stride">The stride to be used by this convolutional layer.</param>
        /// <param name="padding"><code>true</code> if zero padding is to be used in the convolutional layer, <code>false</code> if valid padding is to be used.</param>
        public Convolution(Image input, Image output, int depth, int kernelSide, int stride, bool padding)
        {
            this.input = input;
            this.output = output;
            this.kernels = new Kernel[depth];
            for (int i = 0; i < depth; i++)
            {
                this.kernels[i] = new Kernel(input.Depth, kernelSide, stride, padding);
            }
            this.kernelSide = kernelSide;
            this.stride = stride;
            this.padding = padding;
        }
        
        /// <summary>The input image of this convolutional layer.</summary>
        public Image Input
        {
            get{ return this.input; }
        }

        /// <summary>The output image of this convolutional layer.</summary>
        public Image Output
        {
            get { return this.output; }
        }

        /// <summary>The lenght of the side of each kernel in this convolutional layer.</summary>
        public int KernelSide
        {
            get { return this.kernelSide; }
        }

        /// <summary>The amount of weights in this convolutional layer.</summary>
        public int Params
        {
            get { return this.kernels.Length * this.KernelSide * this.kernelSide; }
        }

        /// <summary>The instance of <code>ConvLayerInfo</code> containing information about this convolutional layer.</summary>
        public virtual ITransofrmationInfo Info
        {
            get { return new ConvLayerInfo(this.KernelSide, this.kernels.Length, this.stride, this.padding); }
        }

        /// <summary>Sets the input and output image of this convolutional layer. Only to be used when strictly necessary.</summary>
        /// <param name="input">The input image to be set.</param>
        /// <param name="output">The output image to be set.</param>
        public void SetLayers(Image input, Image output)
        {
            this.input = input;
            this.output = output;
        }

        /// <summary>Backpropagates the given error trough this convolutional layer, updating the weights in its kernels.</summary>
        /// <param name="error2">The error to be backpropagated. It must refer to the latest feeding process.</param>
        /// <param name="error1">The image to be written the error of the input image into.</param>
        /// <param name="rate">The learning rate at which the weights of the kernels are to be updated.</param>
        public void BackPropagate(Image error2, Image error1, double rate)
        {
            Array.Clear(error1.Raw, 0, error1.Raw.Length);
            for (int i = 0; i < this.kernels.Length; i++)
            {
                this.kernels[i].BackPropagate(this.Input, this.Output, error2, error1, i, (float)rate);
            }
        }

        /// <summary>Feeds the input image trough the convolutional layer into the output image.</summary>
        public void Feed()
        {
            for (int i = 0; i < this.kernels.Length; i++)
            {
                this.kernels[i].Apply(this.Input, this.Output, i);
            }
        }

        /// <summary>Copies of this instance of the <code>Convolution</code> class into another.</summary>
        /// <param name="convolution">The instance to be copied into.</param>
        /// <param name="input">The innput image to be set for the copy.</param>
        /// <param name="output">The output image ot be set for the copy.</param>
        protected virtual void CloneTo(Convolution convolution, Image input, Image output)
        {
            convolution.input = input;
            convolution.output = output;
            convolution.kernels = new Kernel[this.kernels.Length];
            for (int i = 0; i < this.kernels.Length; i++)
            {
                convolution.kernels[i] = (Kernel)this.kernels[i].Clone();
            }
            convolution.kernelSide = this.kernelSide;
        }

        /// <summary>Creates a copy of this instance of the <code>Convolution</code> class.</summary>
        /// <param name="input">The input image of the copy.</param>
        /// <param name="output">The output image of the copy.</param>
        /// <returns>The generated instance of the <code>Convolution</code> class.</returns>
        public virtual IImageTransformation Clone(Image input, Image output)
        {
            Convolution retVal = new Convolution();
            this.CloneTo(retVal, input, output);
            return retVal;
        }
    }
}
