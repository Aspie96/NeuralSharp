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

namespace NeuralNetwork.Convolutional
{
    /// <summary>Represents information about a convolutional layer in a convolutional neural network.</summary>
    public struct ConvLayerInfo : ITransofrmationInfo
    {
        private int kernels;
        private int kernelSide;
        private int stride;
        private bool padding;
        
        /// <summary>Creates an instance of <code>ConvLayerInfo</code>.</summary>
        /// <param name="kernelSide">The side of the kernels in the convolutional layers represented by this info.</param>
        /// <param name="kernels">The number of kernels in the convolutional layers represented by this info.</param>
        /// <param name="stride">The stride to be used in the convolutional layers represented by this info.</param>
        /// <param name="padding"><code>true</code> if zero padding is to be used in the convolutional layers represented by this info, <code>false</code> if valid padding is to be used.</param>
        public ConvLayerInfo(int kernelSide, int kernels, int stride, bool padding)
        {
            this.kernels = kernels;
            this.kernelSide = kernelSide;
            this.stride = stride;
            this.padding = padding;
        }

        /// <summary>The number of kernels in the convolutional layers represented by this info.</summary>
        public int Kernels
        {
            get { return this.kernels; }
        }

        /// <summary>The lenght of the side of each kernel in the convolutional layers represented by this info.</summary>
        public int KernelSide
        {
            get { return this.kernelSide; }
        }

        /// <summary><code>true</code> if zero padding is used by convolutional layers represented by this info, <code>false</code> if valid padding is used.</summary>
        public bool Padding
        {
            get { return this.padding; }
        }

        /// <summary>Creates an instance of the <code>Convolution</code> class according to this info.</summary>
        /// <param name="image1">The input image of the convolutional layer.</param>
        /// <param name="image2">The output image of the convolutional layer.</param>
        /// <returns>The generated instance of the <code>Convolution</code> class.</returns>
        public IImageTransformation GetTransformation(Image image1, Image image2)
        {
            return new Convolution(image1, image2, this.Kernels, this.kernelSide, this.stride, this.Padding);
        }

        /// <summary>Gets the size of the output image of a convolutional layer represented by this info given the size of the input image.</summary>
        /// <param name="beforeDepth">The depth of the input image.</param>
        /// <param name="beforeWidth">The width of the input image.</param>
        /// <param name="beforeHeight">The height of the input image.</param>
        /// <param name="depth">The depth of the output image.</param>
        /// <param name="widht">The width of the output image.</param>
        /// <param name="height">The height of the output image.</param>
        public void SizeAfter(int beforeDepth, int beforeWidth, int beforeHeight, out int depth, out int widht, out int height)
        {
            depth = this.Kernels;
            widht = beforeWidth + 1 - this.kernelSide;
            height = beforeHeight + 1 - this.kernelSide;
        }
    }
}
