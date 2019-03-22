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
    /// <summary>Represents a softmax neurons string.</summary>
    public class SoftmaxNeuronsString : NeuronsString
    {
        /// <summary>Either creates a siamese of the given <code>SoftmaxNeuronsString</code> class or clones it.</summary>
        /// <param name="original">The original instance to be created a siamese of or cloned.</param>
        /// <param name="siamese"><code>true</code> if a siamese is to be created, <code>false</code> if a clone is.</param>
        protected SoftmaxNeuronsString(SoftmaxNeuronsString original, bool siamese) : base(original, siamese) { }

        /// <summary>Creates an instance of the <code>SoftmaxNeuronsString</code> class.</summary>
        /// <param name="length">The lenght of the layer.</param>
        /// <param name="createIO">Whether the input array and the output array are to be created.</param>
        public SoftmaxNeuronsString(int length, bool createIO = false) : base(length, createIO) { }
        
        /// <summary>Applies the softmax function.</summary>
        /// <param name="array">The array to be applied the softmax function to.</param>
        public static void Softmax(float[] array)
        {
            SoftmaxNeuronsString.Softmax(array, array);
        }
        
        /// <summary>Applies the softmax function.</summary>
        /// <param name="array">The input array.</param>
        /// <param name="output">The output array.</param>
        public static void Softmax(float[] array, float[] output)
        {
            float expSum = 0.0F;
            for (int i = 0; i < array.Length; i++)
            {
                expSum += output[i] =(float)Math.Exp(array[i]);
            }
            for (int i = 0; i < array.Length; i++)
            {
                output[i] /= expSum;
            }
        }
        
        /// <summary>Applies the derivative of the softmax function.</summary>
        /// <param name="array">The input.</param>
        /// <param name="output">The output.</param>
        public static void SoftmaxDerivative(float[] array, float[,] output)
        {
            for (int i = 0; i < array.Length; i++)
            {
                for (int j = 0; j < array.Length; j++)
                {
                    output[i, j] = -array[i] * array[j];
                }
                output[i, i] = array[i] * (1.0F - array[i]);
            }
        }

        /// <summary>Backpropagates the given error trough the network.</summary>
        /// <param name="outputErrorArray">The error to be backpropagated.</param>
        /// <param name="outputErrorSkip">The index of the first entry of the output error to be used.</param>
        /// <param name="inputErrorArray">The array to be written the input error into.</param>
        /// <param name="inputErrorSkip">The index of the first entry of the input error to be used.</param>
        /// <param name="learning">Whether the layer is being used in a learning session.</param>
        public override void BackPropagate(float[] outputErrorArray, int outputErrorSkip, float[] inputErrorArray, int inputErrorSkip, bool learning)
        {
            Backbone.BackpropagateSoftmax(this.Input, this.InputSkip, this.Output, this.OutputSkip, this.Length, outputErrorArray, outputErrorSkip, inputErrorArray, inputErrorSkip, learning);
        }
        
        /// <summary>Feeds the layer forward.</summary>
        /// <param name="learning">Whether the layer is being used in a training session.</param>
        public override void Feed(bool learning = false)
        {
            Backbone.ApplySoftmax(this.Input, this.InputSkip, this.Output, this.OutputSkip, this.Length);
        }

        /// <summary>Creates a siamese of the layer.</summary>
        /// <returns>The created instance of the <code>SoftmaxNeuronsString</code> class.</returns>
        public override ILayer<float[], float[]> CreateSiamese()
        {
            return new SoftmaxNeuronsString(this, true);
        }

        /// <summary>Creates a clone of the layer.</summary>
        /// <returns>The created instance of the <code>SoftmaxNeuronsString</code> class.</returns>
        public override ILayer<float[], float[]> Clone()
        {
            return new SoftmaxNeuronsString(this, false);
        }
    }
}
