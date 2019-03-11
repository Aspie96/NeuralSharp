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
    /// <summary>Represents a layer which turns an image into an array.</summary>
    public class ImageToArray : IImageArrayLayer
    {
        private int inputDepth;
        private int inputWidth;
        private int inputHeight;
        private int outputSkip;
        private Image input;
        private double[] output;

        /// <summary>Either creates a siamese of the given <code>ImageToArray</code> instance or clones it.</summary>
        /// <param name="original">The original instance to be created a siamese of or cloned.</param>
        /// <param name="siamese"><code>true</code> if a siamese is to be created, <code>false</code> if a clone is.</param>
        protected ImageToArray(ImageToArray original, bool siamese)
        {
            this.inputDepth = original.InputDepth;
            this.inputWidth = original.InputWidth;
            this.inputHeight = original.InputHeight;
        }

        /// <summary>Creates an instance of the <code>ImageToArray</code> class.</summary>
        /// <param name="inputDepth">The depth of the input of the layer.</param>
        /// <param name="inputWidth">The width of the input of the layer.</param>
        /// <param name="inputHeight">The height of the input of the layer.</param>
        /// <param name="createIO">Whether the input image and the output array are to be created.</param>
        public ImageToArray(int inputDepth, int inputWidth, int inputHeight, bool createIO = false)
        {
            this.inputDepth = inputDepth;
            this.inputWidth = inputWidth;
            this.inputHeight = inputHeight;
            if (createIO)
            {
                this.SetInputGetOutput(new Image(inputDepth, inputWidth, inputHeight));
            }
        }

        /// <summary>The input image of the layer.</summary>
        public Image Input
        {
            get { return this.input; }
        }

        /// <summary>The output array of the layer.</summary>
        public double[] Output
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

        /// <summary>The lenght of the output of the layer.</summary>
        public int OutputSize
        {
            get { return this.InputDepth * this.InputWidth * this.InputHeight; }
        }

        /// <summary>The index of the first used entry of the input array.</summary>
        public int OutputSkip
        {
            get { return this.outputSkip; }
        }
        
        /// <summary>The amount of parameters of the layer. Always <code>0</code>.</summary>
        public int Parameters
        {
            get { return 0; }
        }
        
        /// <summary>Feeds the layer forward.</summary>
        /// <param name="learning">Whether the layer is being used in a training session.</param>
        public void Feed(bool learning)
        {
            this.Input.ToArray(this.Output, this.OutputSkip);
        }
        
        /// <summary>Backpropagates an error trough the layer.</summary>
        /// <param name="outputError">The error to be backpropagated.</param>
        /// <param name="inputError">The image to be written the input error into.</param>
        /// <param name="learning">Whether the layer is being used in a training session.</param>
        public void BackPropagate(double[] outputError, Image inputError, bool learning)
        {
            inputError.FromArray(outputError);
        }

        /// <summary>Updates the weights of the layer.</summary>
        /// <param name="rate">The learning rate to be used.</param>
        /// <param name="momentum">The momentum to be used.</param>
        public void UpdateWeights(double rate, double momentum = 0.0) { }

        /// <summary>Sets the input image and the output array of the layer.</summary>
        /// <param name="input">The input image to be set.</param>
        /// <param name="outputArray">The output array to be set.</param>
        /// <param name="outputSkip">The index of the first entry of the output array to be used.</param>
        public void SetInputAndOutput(Image input, double[] outputArray, int outputSkip)
        {
            this.input = input;
            this.output = outputArray;
            this.outputSkip = outputSkip;
        }

        /// <summary>Sets the input image and the output array of the layer.</summary>
        /// <param name="input">The input image to be set.</param>
        /// <param name="output">The output array to be set.</param>
        public void SetInputAndOutput(Image input, double[] output)
        {
            this.SetInputAndOutput(input, output, 0);
        }

        /// <summary>Sets the input image and creates and sets the output array of the layer.</summary>
        /// <param name="input">The input image to be set.</param>
        /// <returns>The created output array.</returns>
        public double[] SetInputGetOutput(Image input)
        {
            double[] retVal = Backbone.CreateArray<double>(this.OutputSize);
            this.SetInputAndOutput(input, retVal);
            return retVal;
        }
        
        /// <summary>Cretes a siamese of the layer.</summary>
        /// <returns>The created instance of the <code>ImageToArray</code> class.</returns>
        public IUntypedLayer CreateSiamese()
        {
            return new ImageToArray(this, true);
        }

        /// <summary>Creates a siamese of the layer.</summary>
        /// <returns>The created instance of the <code>ImageToArray</code> class.</returns>
        public IUntypedLayer Clone()
        {
            return new ImageToArray(this, false);
        }
    }
}
