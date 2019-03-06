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
    [DataContract]
    public class SoftmaxNeuronsString : NeuronsString
    {
        protected SoftmaxNeuronsString(SoftmaxNeuronsString original, bool siamese) : base(original, siamese) { }

        public SoftmaxNeuronsString(int length, bool createIO = false) : base(length, createIO) { }
        
        public static void Softmax(double[] array)
        {
            SoftmaxNeuronsString.Softmax(array, array);
        }
        
        public static void Softmax(double[] array, double[] output)
        {
            double expSum = 0.0;
            for (int i = 0; i < array.Length; i++)
            {
                expSum += output[i] = Math.Exp(array[i]);
            }
            for (int i = 0; i < array.Length; i++)
            {
                output[i] /= expSum;
            }
        }
        
        public static void SoftmaxDerivative(double[] array, double[,] output)
        {
            for (int i = 0; i < array.Length; i++)
            {
                for (int j = 0; j < array.Length; j++)
                {
                    output[i, j] = -array[i] * array[j];
                }
                output[i, i] = array[i] * (1.0 - array[i]);
            }
        }

        public override void BackPropagate(double[] outputErrorArray, int outputErrorSkip, double[] inputErrorArray, int inputErrorSkip, bool learning)
        {
            Backbone.BackpropagateSoftmax(this.Input, this.InputSkip, this.Output, this.OutputSkip, this.Length, outputErrorArray, outputErrorSkip, inputErrorArray, inputErrorSkip, learning);
        }
        
        public override void Feed(bool learning = false)
        {
            Backbone.ApplySoftmax(this.Input, this.InputSkip, this.Output, this.OutputSkip, this.Length);
        }

        public override IUntypedLayer CreateSiamese()
        {
            return new SoftmaxNeuronsString(this, true);
        }

        public override IUntypedLayer Clone()
        {
            return new SoftmaxNeuronsString(this, false);
        }
    }
}
