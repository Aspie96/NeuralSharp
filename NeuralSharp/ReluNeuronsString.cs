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
using System.Runtime.Serialization;
using System.Text;
using System.Threading.Tasks;

namespace NeuralSharp
{
    /// <summary>Represents a neurons string whose activation function is the relu activation.</summary>
    public class ReluNeuronsString : NeuronsString
    {
        /// <summary>Either creates a siamese of the given <code>ReluNeuronsString</code> instance or clones it.</summary>
        /// <param name="original">The original instance to be created a siamese of or a clone.</param>
        /// <param name="siamese"><code>true</code> if a siamese is to be created, <code>false</code> if a clone is.</param>
        protected ReluNeuronsString(ReluNeuronsString original, bool siamese) : base(original, siamese) { }

        /// <summary>Creates an instance of the <code>ReluNeuronsString</code> class.</summary>
        /// <param name="length">The lenght of the layer.</param>
        /// <param name="createIO">Whether the input array and the output array of the layer are to be created.</param>
        public ReluNeuronsString(int length, bool createIO = false) : base(length, createIO) { }
        
        /// <summary>The activation function of the network.</summary>
        /// <param name="input">The input.</param>
        /// <returns>The output.</returns>
        protected override double Activation(double input)
        {
            return Math.Max(input, 0.0);
        }

        /// <summary>The derivative of the activation function.</summary>
        /// <param name="input">The input of the activation function.</param>
        /// <param name="output">The output of the activation function.</param>
        /// <returns>The derivative.</returns>
        protected override double ActivationDerivative(double input, double output)
        {
            return (output > 0.0 ? 1.0 : 0.0);
        }
    }
}
