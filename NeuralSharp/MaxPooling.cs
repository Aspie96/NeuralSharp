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
    /// <summary>Represents a max pooling layer.</summary>
    public class MaxPooling : Pooling
    {
        /// <summary>Either creates a siamese of the given <code>MaxPooling</code> instance or clones it.</summary>
        /// <param name="original">The original instance to be created a siamese of or cloned.</param>
        /// <param name="siamese"><code>true</code> if a siamese is to be created, <code>false</code> if a clone is.</param>
        protected MaxPooling(MaxPooling original, bool siamese) : base(original, siamese) { }

        /// <summary>Creates a max pooling layer.</summary>
        /// <param name="inputDepth">The depth of the input of the layer.</param>
        /// <param name="inputWidth">The width of the input of the layer.</param>
        /// <param name="inputHeight">The height of the input of the layer.</param>
        /// <param name="xScale">The X scale coefficient.</param>
        /// <param name="yScale">The Y scale coefficient.</param>
        /// <param name="createIO">Whether the input image and the output image are to be created.</param>
        public MaxPooling(int inputDepth, int inputWidth, int inputHeight, int xScale, int yScale, bool createIO = false) : base(inputDepth, inputWidth, inputHeight, xScale, yScale, createIO) { }
        
        /// <summary>Feeds the layer forward.</summary>
        /// <param name="learning">Whether the layer is being used in a training session.</param>
        public override void Feed(bool learning = false)
        {
            Backbone.ApplyMaxPool(this.Input.Raw, this.InputDepth, this.InputWidth, this.InputHeight, this.Output.Raw, this.OutputDepth, this.OutputWidth, this.OutputHeight, this.XScale, this.YScale);
        }

        /// <summary>Backpropagates the given error trough the layer.</summary>
        /// <param name="outputError">The output error to be backpropagated.</param>
        /// <param name="inputError">The image to be written the input error into.</param>
        /// <param name="learning">Whether the layer is being used in a training session.</param>
        public override void BackPropagate(Image outputError, Image inputError, bool learning)
        {
            Backbone.BackpropagateMaxPool(this.Input.Raw, this.InputDepth, this.InputWidth, this.InputHeight, this.Output.Raw, this.OutputDepth, this.OutputWidth, this.OutputHeight, this.XScale, this.YScale, outputError.Raw, outputError.Depth, outputError.Width, outputError.Height, inputError.Raw, inputError.Depth, inputError.Width, inputError.Height, learning);
        }

        /// <summary>Cretes a siamese of the layer.</summary>
        /// <returns>The created instance of the <code>MaxPooling</code> class.</returns>
        public override IUntypedLayer CreateSiamese()
        {
            return new MaxPooling(this, true);
        }

        /// <summary>Creates a clone of the layer.</summary>
        /// <returns>The created clone.</returns>
        public override IUntypedLayer Clone()
        {
            return new MaxPooling(this, false);
        }
    }
}
