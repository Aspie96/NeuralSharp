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
    public class ImageToArray : IImageArrayLayer
    {
        [DataMember]
        private int inputDepth;
        [DataMember]
        private int inputWidth;
        [DataMember]
        private int inputHeight;
        [DataMember]
        private int outputSkip;
        private Image input;
        private double[] output;

        protected ImageToArray(ImageToArray original, bool siamese)
        {
            this.inputDepth = original.InputDepth;
            this.inputWidth = original.InputWidth;
            this.inputHeight = original.InputHeight;
        }

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

        public Image Input
        {
            get { return this.input; }
        }

        public double[] Output
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

        public int OutputSize
        {
            get { return this.InputDepth * this.InputWidth * this.InputHeight; }
        }

        public int OutputSkip
        {
            get { return this.outputSkip; }
        }
        
        public int Parameters
        {
            get { return 0; }
        }
        
        public void Feed(bool learning)
        {
            this.Input.ToArray(this.Output, this.OutputSkip);
        }
        
        public void BackPropagate(double[] outputError, Image inputError, bool learning)
        {
            inputError.FromArray(outputError);
        }

        public void UpdateWeights(double rate, double nextMomentum = 0.0) { }

        public void SetInputAndOutput(Image input, double[] outputArray, int outputSkip)
        {
            this.input = input;
            this.output = outputArray;
            this.outputSkip = outputSkip;
        }

        public void SetInputAndOutput(Image input, double[] output)
        {
            this.SetInputAndOutput(input, output, 0);
        }

        public double[] SetInputGetOutput(Image input)
        {
            double[] retVal = Backbone.CreateArray<double>(this.OutputSize);
            this.SetInputAndOutput(input, retVal);
            return retVal;
        }
        
        public IUntypedLayer CreateSiamese()
        {
            throw new NotImplementedException();
        }

        public IUntypedLayer Clone()
        {
            throw new NotImplementedException();
        }
    }
}
