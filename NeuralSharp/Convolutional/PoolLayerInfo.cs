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
    /// <summary>Represents information about a pooling layer.</summary>
    public struct PoolLayerInfo : ITransofrmationInfo
    {
        private int xScale;
        private int yScale;

        /// <summary>Creates a new instance of <code>PoolLayerInfo</code>.</summary>
        /// <param name="xScale">The scaling factor along the horizontal axis of the pooling layers represented by this info.</param>
        /// <param name="yScale">The scaling factor along the vertical axis of the pooling layers represented by this info.</param>
        public PoolLayerInfo(int xScale, int yScale)
        {
            this.xScale = xScale;
            this.yScale = yScale;
        }

        /// <summary>The horizontal scaling factor of the pooling layers represented by this info.</summary>
        public int XScale
        {
            get { return this.xScale; }
        }

        /// <summary>The vertical scaling factor of the pooling layers represented by this info.</summary>
        public int YScale
        {
            get { return this.YScale; }
        }

        /// <summary>Creates an instance of the <code>Pooling</code> class according to this info.</summary>
        /// <param name="image1">The input image of the pooling layer.</param>
        /// <param name="image2">The output image of the pooling layer.</param>
        /// <returns>The generated instance of the <code>Pooling</code>.</returns>
        public IImageTransformation GetTransformation(Image image1, Image image2)
        {
            return new MaxPooling(image1, image2, this.XScale, this.YScale);
        }

        /// <summary>Gets the size of the output image of a pooling layer represented by this info given the size of the input image.</summary>
        /// <param name="beforeDepth">The depth of the input image.</param>
        /// <param name="beforeWidth">The width of the input image.</param>
        /// <param name="beforeHeight">The height of the input image.</param>
        /// <param name="depth">The depth of the output image.</param>
        /// <param name="widht">The width of the otuput image.</param>
        /// <param name="height">The height of the output image.</param>
        public void SizeAfter(int beforeDepth, int beforeWidth, int beforeHeight, out int depth, out int widht, out int height)
        {
            depth = beforeDepth;
            widht = beforeWidth / this.XScale;
            height = beforeHeight / this.YScale;
        }
    }
}
