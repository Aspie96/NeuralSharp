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
    public class MaxPooling : Pooling
    {
        protected MaxPooling(MaxPooling original, bool siamese) : base(original, siamese) { }

        public MaxPooling(int inputDepth, int inputWidth, int inputHeight, int xScale, int yScale, bool createIO = false) : base(inputDepth, inputWidth, inputHeight, xScale, yScale, createIO) { }
        
        public override void Feed(bool learning = false)
        {
            Backbone.ApplyMaxPool(this.Input.Raw, this.InputDepth, this.InputWidth, this.InputHeight, this.Output.Raw, this.OutputDepth, this.OutputWidth, this.OutputHeight, this.XScale, this.YScale);
        }

        public override void BackPropagate(Image outputError, Image inputError, bool learning)
        {
            Backbone.BackpropagateMaxPool(this.Input.Raw, this.InputDepth, this.InputWidth, this.InputHeight, this.Output.Raw, this.OutputDepth, this.OutputWidth, this.OutputHeight, this.XScale, this.YScale, outputError.Raw, outputError.Depth, outputError.Width, outputError.Height, inputError.Raw, inputError.Depth, inputError.Width, inputError.Height, learning);
        }

        public override IUntypedLayer CreateSiamese()
        {
            return new MaxPooling(this, true);
        }

        public override IUntypedLayer Clone()
        {
            return new MaxPooling(this, false);
        }

        public override ILayer<Image, Image> CreateSiamese(bool createIO = false)
        {
            throw new NotImplementedException();
        }
    }
}
