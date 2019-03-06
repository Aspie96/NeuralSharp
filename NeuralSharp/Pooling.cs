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
    public abstract class Pooling : IImagesLayer
    {
        private Image input;
        private Image output;
        [DataMember]
        private int inputDepth;
        [DataMember]
        private int inputWidth;
        [DataMember]
        private int inputHeight;
        [DataMember]
        private int outputDepth;
        [DataMember]
        private int outputWidth;
        [DataMember]
        private int outputHeight;
        [DataMember]
        private int xScale;
        [DataMember]
        private int yScale;
        
        protected Pooling(Pooling original, bool siamese)
        {
            this.inputDepth = original.InputDepth;
            this.inputWidth = original.InputWidth;
            this.inputHeight = original.inputHeight;
            this.outputDepth = original.OutputDepth;
            this.outputWidth = original.OutputWidth;
            this.outputHeight = original.OutputHeight;
            this.xScale = original.XScale;
            this.yScale = original.YScale;
        }

        public Pooling(int inputDepth, int inputWidth, int inputHeight, int xScale, int yScale, bool createIO = false)
        {
            this.inputDepth = inputDepth;
            this.inputWidth = inputWidth;
            this.inputHeight = inputHeight;
            this.outputDepth = inputDepth;
            this.outputWidth = inputWidth / xScale;
            this.outputHeight = inputHeight / yScale;
            if (createIO)
            {
                this.input = new Image(inputDepth, inputWidth, inputHeight);
                this.output = new Image(outputDepth, outputWidth, outputHeight);
            }
            this.xScale = xScale;
            this.yScale = yScale;
        }

        public Image Input
        {
            get { return this.input; }
        }

        public Image Output
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

        public int OutputDepth
        {
            get { return this.outputDepth; }
        }

        public int OutputWidth
        {
            get { return this.outputWidth; }
        }

        public int OutputHeight
        {
            get { return this.outputHeight; }
        }
        
        public int XScale
        {
            get { return this.xScale; }
        }
        
        public int YScale
        {
            get { return this.yScale; }
        }
        
        public int Parameters
        {
            get { return 0; }
        }

        public abstract void Feed(bool learning = false);
        public abstract void BackPropagate(Image outputError, Image inputError, bool learning);
        public abstract ILayer<Image, Image> CreateSiamese(bool createIO = false);
        public abstract IUntypedLayer CreateSiamese();
        public abstract IUntypedLayer Clone();

        public void UpdateWeights(double rate, double momentum = 0.0) { }
        
        public void SetInputAndOutput(Image input, Image output)
        {
            this.input = input;
            this.output = output;
        }

        public Image SetInputGetOutput(Image input)
        {
            this.input = input;
            return this.output = new Image(this.OutputDepth, this.OutputWidth, this.OutputHeight);
        }
    }
}
