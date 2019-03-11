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
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralSharp
{
    /// <summary>Creates a tanh neurons string.</summary>
    public class TanhNeuronsString : NeuronsString
    {
        /// <summary>Either creates a siamese of the given <code>TanhNeuronsString</code> instance or clones it.</summary>
        /// <param name="original">The original instance to be created a siamese of or cloned.</param>
        /// <param name="siamese"><code>true</code> if a siamese is to be created, <code>false</code> if a clone is.</param>
        protected TanhNeuronsString(TanhNeuronsString original, bool siamese) : base(original, siamese) { }

        /// <summary>Creates an instance of the <code>TanhNeuronsString</code> class.</summary>
        /// <param name="length">The lenght of the layer.</param>
        /// <param name="createIO">Whether the input and the output layer are to be created.</param>
        public TanhNeuronsString(int length, bool createIO = false) : base(length, createIO) { }
        
        /// <summary>The activation function.</summary>
        /// <param name="input">The input.</param>
        /// <returns>The output.</returns>
        protected override double Activation(double input)
        {
            return Math.Tanh(input);
        }

        /// <summary>The derivative of the activation function.</summary>
        /// <param name="input">The input.</param>
        /// <param name="output">The output.</param>
        /// <returns>The derivative.</returns>
        protected override double ActivationDerivative(double input, double output)
        {
            return 1.0 - output * output;
        }

        /// <summary>Creates a siamese of the layer.</summary>
        /// <returns>The created siamese.</returns>
        public override IUntypedLayer CreateSiamese()
        {
            return new TanhNeuronsString(this, true);
        }

        /// <summary>Creates a clone of the layer.</summary>
        /// <returns>The created clone.</returns>
        public override IUntypedLayer Clone()
        {
            return new TanhNeuronsString(this, false);
        }
    }
}
