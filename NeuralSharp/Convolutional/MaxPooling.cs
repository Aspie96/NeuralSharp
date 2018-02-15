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

using System.Runtime.Serialization;

namespace NeuralNetwork.Convolutional
{
    /// <summary>Represents a max pooling layer in a convolutional neural network.</summary>
    [DataContract]
    public class MaxPooling : Pooling
    {
        /// <summary>Empty constructor. It does not set the fields.</summary>
        protected MaxPooling() { }

        /// <summary>Creates a new instance of the <code>MaxPooling</code> class</summary>
        /// <param name="input">The input image of the pooling layer.</param>
        /// <param name="output">The output image of the pooling layer.</param>
        /// <param name="xScale">The scaling factor along the horizontal axis.</param>
        /// <param name="yScale">The scaling factor along the vertical axis.</param>
        public MaxPooling(Image input, Image output, int xScale, int yScale) : base(input, output, xScale, yScale) { }

        /// <summary>Gets a value for the output image from the input image.</summary>
        /// <param name="w">The W coordinate of the value.</param>
        /// <param name="x">The X coordinate of the value.</param>
        /// <param name="y">The Y coordinate of the value.</param>
        /// <returns>The value to be set in the output image.</returns>
        protected override float GetValue(int w, int x, int y)
        {
            float maxValue = 0.0F;
            x *= this.XScale;
            y *= this.YScale;
            for (int i = 0; i < this.XScale; i++)
            {
                for (int j = 0; j < this.YScale; j++)
                {
                    float val = this.Input.Raw[w, i + x, j + y];
                    if (val > maxValue)
                    {
                        maxValue = val;
                    }
                }
            }
            return maxValue;
        }

        /// <summary>Backpropagates a value of the given error.</summary>
        /// <param name="error2">The error to be backpropagated. It must refer to the latest feeding process.</param>
        /// <param name="error1">The image to be written the error of the input image into.</param>
        /// <param name="w">The W coordinate of the value to be backpropagated.</param>
        /// <param name="x">The X coordinate of the value to be backpropagated.</param>
        /// <param name="y">The Y coordinate of the value to be backpropagated.</param>
        protected override void BackPropagateValue(Image error2, Image error1, int w, int x, int y)
        {
            float err = error2.Raw[w, x, y];
            x *= this.XScale;
            y *= this.YScale;
            float maxValue = 0.0F;
            int maxValueX = x;
            int maxValueY = y;
            for (int i = x; i < x + this.XScale; i++)
            {
                for (int j = y; j < y + this.YScale; j++)
                {
                    error1.Raw[w, i, j] = 0.0F;
                    float val = this.Input.Raw[w, i, j];
                    if (val > maxValue)
                    {
                        maxValue = val;
                        maxValueX = i;
                        maxValueY = j;
                    }
                }
            }
            error1.Raw[w, maxValueX, maxValueY] = err;
        }

        /// <summary>The instance of <code>PoolLayerInfo</code> containing information about this pooling layer.</summary>
        public override ITransofrmationInfo Info
        {
            get { return new PoolLayerInfo(this.XScale, this.YScale); }
        }

        /// <summary>A copy of this <code>MaxPooling</code> instance.</summary>
        /// <param name="input">The input image of the copy.</param>
        /// <param name="output">The output image of the copy.</param>
        /// <returns>The generated instance of the <code>MaxPooling</code> class.</returns>
        public override IImageTransformation Clone(Image input, Image output)
        {
            MaxPooling retVal = new MaxPooling();
            base.CloneTo(retVal, input, output);
            return retVal;
        }
    }
}
