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
    /// <summary>Represents a pooling layer in a convolutional neural network.</summary>
    [DataContract]
    public abstract class Pooling : IImageTransformation
    {
        private Image input;
        private Image output;
        [DataMember]
        private int xScale;
        [DataMember]
        private int yScale;

        /// <summary>Empty constructor. It does not initialize the fields.</summary>
        protected Pooling() { }

        /// <summary>Creates a new instance of the <code>Pooling</code> class.</summary>
        /// <param name="input">The input image of the pooling layer.</param>
        /// <param name="output">The output image of the pooling layer.</param>
        /// <param name="xScale">Th scaling factor along the horizontal axis.</param>
        /// <param name="yScale">The scaling factor along the horizontal axis.</param>
        public Pooling(Image input, Image output, int xScale, int yScale)
        {
            this.input = input;
            this.output = output;
            this.xScale = xScale;
            this.yScale = yScale;
        }

        /// <summary>Information about this pooling layer.</summary>
        public abstract ITransofrmationInfo Info { get; }

        /// <summary>Gets the value for the output image at the given position.</summary>
        /// <param name="w">The W coordinate for the value.</param>
        /// <param name="x">The X coordinate for the value.</param>
        /// <param name="y">The Y coordinate for the value.</param>
        /// <returns>The value for the output image.</returns>
        protected abstract float GetValue(int w, int x, int y);

        /// <summary>Backpropagates a value of the error trough this layer.</summary>
        /// <param name="error2">The error to be backpropagated. It must refer to the latest feeding process.</param>
        /// <param name="error1">The image to be written the error of the input image into.</param>
        /// <param name="w">The W coordinate of the value.</param>
        /// <param name="x">The X coordinate of the value.</param>
        /// <param name="y">The Y coordinate of the value.</param>
        protected abstract void BackPropagateValue(Image error2, Image error1, int w, int x, int y);

        /// <summary>Creates a copy of this pooling layer.</summary>
        /// <param name="input">The input image to be set for the copy of this pooling layer.</param>
        /// <param name="output">The output image to be set for the copy of this pooling layer.</param>
        /// <returns>The generated instance.</returns>
        public abstract IImageTransformation Clone(Image input, Image output);

        /// <summary>The input image of this pooling layer.</summary>
        public Image Input
        {
            get { return this.input; }
        }

        /// <summary>The output image of this pooling layer.</summary>
        public Image Output
        {
            get { return this.output; }
        }

        /// <summary>The horizontal scaling factor of this layer.</summary>
        public int XScale
        {
            get { return this.xScale; }
        }

        /// <summary>The vertical scaling factor of this layer.</summary>
        public int YScale
        {
            get { return this.yScale; }
        }

        /// <summary>Always <code>0</code>.</summary>
        public int Params
        {
            get { return 0; }
        }

        /// <summary>Sets the input and output image for this layer. To be used only when strictly necessary.</summary>
        /// <param name="input">The input image to be set.</param>
        /// <param name="output">The output image to be set.</param>
        public void SetLayers(Image input, Image output)
        {
            this.input = input;
            this.output = output;
        }

        /// <summary>Feeds the input image trough this layer into the output.</summary>
        public void Feed()
        {
            for (int i = 0; i < this.output.Depth; i++)
            {
                for (int j = 0; j < this.Output.Width; j++)
                {
                    for (int k = 0; k < this.Output.Height; k++)
                    {
                        this.Output.SetValue(i, j, k, this.GetValue(i, j, k));
                    }
                }
            }
        }

        /// <summary>Backpropagates the given error trough this pooling layer.</summary>
        /// <param name="error2">The error to be backpropagated. It must refer to the latest feeding process.</param>
        /// <param name="error1">The image to be written the error of the input image into.</param>
        /// <param name="rate">The learning rate at which the weights are to be updated. Not used.</param>
        public void BackPropagate(Image error2, Image error1, double rate)
        {
            for (int i = 0; i < this.output.Depth; i++)
            {
                for (int j = 0; j < this.Output.Width; j++)
                {
                    for (int k = 0; k < this.Output.Height; k++)
                    {
                        this.BackPropagateValue(error2, error1, i, j, k);
                    }
                }
            }
        }

        /// <summary>Copies this instance of the <code>Pooling</code> class into another.</summary>
        /// <param name="pooling">The instance of the <code>Pooling</code> class to be copied into.</param>
        /// <param name="input">The input image to be set for the copy.</param>
        /// <param name="output">The output image to be setfor the copy.</param>
        protected virtual void CloneTo(Pooling pooling, Image input, Image output)
        {
            pooling.input = input;
            pooling.output = output;
            pooling.xScale = this.xScale;
            pooling.yScale = this.yScale;
        }
    }
}
