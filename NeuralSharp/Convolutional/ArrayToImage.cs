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

namespace NeuralSharp.Convolutional
{
    /// <summary>Represents a layer which converts an array to an image.</summary>
    public class ArrayToImage : IArrayImageLayer
    {
        private int inputSkip;
        private int inputSize;
        private int outputDepth;
        private int outputWidth;
        private int outputHeight;
        private Image output;
        private float[] input;
        private object siameseID;

        /// <summary>Either creates a siamese of the given <code>ArrayToImage</code> instance or clones it.</summary>
        /// <param name="original">The original instance to be created a siamese of or cloned.</param>
        /// <param name="siamese"><code>true</code> if a siamese is to be created, <code>false</code> if a clone is.</param>
        protected ArrayToImage(ArrayToImage original, bool siamese)
        {
            this.inputSize = original.InputSize;
            this.outputDepth = original.OutputDepth;
            this.outputWidth = original.OutputWidth;
            this.outputHeight = original.OutputHeight;
            if (siamese)
            {
                this.siameseID = original.SiameseID;
            }
            else
            {
                this.siameseID = new object();
            }
        }

        /// <summary>Create an instance of the <code>ArrayToImage</code> class.</summary>
        /// <param name="outputDepth">The depth of the output of the layer.</param>
        /// <param name="outputWidth">The width of the output of the layer.</param>
        /// <param name="outputHeight">The height of the output of the layer.</param>
        /// <param name="createIO">Whether the input array and the output image of the layer are to be created.</param>
        public ArrayToImage(int outputDepth, int outputWidth, int outputHeight, bool createIO = false)
        {
            this.inputSize = outputDepth * outputWidth * outputHeight;
            this.outputDepth = outputDepth;
            this.outputWidth = outputWidth;
            this.outputHeight = outputHeight;
            if (createIO)
            {
                this.SetInputGetOutput(Backbone.CreateArray<float>(this.inputSize));
            }
            this.siameseID = new object();
        }

        /// <summary>The input array of the layer.</summary>
        public float[] Input
        {
            get { return this.input; }
        }

        /// <summary>The output image of the layer.</summary>
        public Image Output
        {
            get { return this.output; }
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

        /// <summary>The length of the input of the layer.</summary>
        public int InputSize
        {
            get { return this.inputSize; }
        }

        /// <summary>The index of the first used entry of the input array.</summary>
        public int InputSkip
        {
            get { return this.inputSkip; }
        }

        /// <summary>The amount of parameters of the layer. Always <code>0</code>.</summary>
        public int Parameters
        {
            get { return 0; }
        }

        /// <summary>The siamese identifier of the layer.</summary>
        public object SiameseID
        {
            get { return this.siameseID; }
        }

        /// <summary>Feeds the layer forward.</summary>
        /// <param name="learning">Whether the layer is being used in a training session. Unused.</param>
        public void Feed(bool learning)
        {
            this.Output.FromArray(this.Input, this.InputSkip);
        }

        /// <summary>Backpropagates the given error trough the layer.</summary>
        /// <param name="outputError">The output error to be backpropagated.</param>
        /// <param name="inputError">The array to be written the input error into.</param>
        /// <param name="learning">Whether the layer is being used in a training session. Unused.</param>
        public void BackPropagate(Image outputError, float[] inputError, bool learning)
        {
            outputError.ToArray(inputError);
        }

        /// <summary>Updates the weights of the layer. Does nothing.</summary>
        /// <param name="rate">The learning rate to be used.</param>
        /// <param name="momentum">The momentum to be used.</param>
        public void UpdateWeights(float rate, float momentum = 0.0F) { }

        /// <summary>Sets the input array and the output image of the layer.</summary>
        /// <param name="inputArray">The input array to be set.</param>
        /// <param name="inputSkip">The index of the first entry of the given array to be used.</param>
        /// <param name="output">The output image to be set.</param>
        public void SetInputAndOutput(float[] inputArray, int inputSkip, Image output)
        {
            this.input = inputArray;
            this.inputSkip = inputSkip;
            this.output = output;
        }

        /// <summary>Sets the input array and the output image of the layer.</summary>
        /// <param name="input">The input array to be set.</param>
        /// <param name="output">The output image to be set.</param>
        public void SetInputAndOutput(float[] input, Image output)
        {
            this.SetInputAndOutput(input, 0, output);
        }

        /// <summary>Sets the input array of the layer and creates and sets an output image.</summary>
        /// <param name="input">The input array to be set.</param>
        /// <returns>The created output image.</returns>
        public Image SetInputGetOutput(float[] input)
        {
            Image retVal = new Image(this.OutputDepth, this.OutputWidth, this.OutputHeight);
            this.SetInputAndOutput(input, retVal);
            return retVal;
        }
        
        /// <summary>Creates a siamese of this layer.</summary>
        /// <returns>The created <code>ArrayToImage</code> instance.</returns>
        public virtual ILayer<float[], Image> CreateSiamese()
        {
            return new ArrayToImage(this, true);
        }

        /// <summary>Creates a clone of this layer.</summary>
        /// <returns>The created <code>ArrayToImage</code> instance.</returns>
        public virtual ILayer<float[], Image> Clone()
        {
            return new ArrayToImage(this, false);
        }

        /// <summary>Computes the amount of parameters of the layer.</summary>
        /// <param name="siameseIDs">The siamese identifiers to be excluded. The siamese identifiers of the layer will be added to the list.</param>
        /// <returns>The amount of parameters of the layer.</returns>
        public int CountParameters(List<object> siameseIDs)
        {
            if (!siameseIDs.Contains(this.SiameseID))
            {
                siameseIDs.Add(this.siameseID);
            }
            return 0;
        }
    }
}
