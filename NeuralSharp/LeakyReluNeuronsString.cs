/*
    (C) 2018 Valentino Giudice

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

namespace NeuralNetwork
{
    /// <summary>Represents a layer using the leaky relu activation function.</summary>
    [DataContract]
    public class LeakyReluNeuronsString : NeuronsString
    {
        [DataMember]
        private double alpha;

        /// <summary>The parameter of the leaky relu.</summary>
        public double Alpha
        {
            get { return this.alpha; }
        }

        /// <summary>Creates a new instance of the <code>LeakyReluNeuronsString</code> class.</summary>
        /// <param name="length">The lenght of the layer.</param>
        /// <param name="alpha">The parameter of the leaky relu.</param>
        public LeakyReluNeuronsString(int length, double alpha) : base(length)
        {
            this.alpha = alpha;
        }

        /// <summary>Returns the value of the activation function for a given input value.</summary>
        /// <param name="input">The input to be given to the activation function.</param>
        /// <returns>The output of the activation function.</returns>
        protected override double Activation(double input)
        {
            if (input <= 0.0)
            {
                return this.alpha * input;
            }
            return input;
        }

        /// <summary>Returns the derivative of the activation function for the given input and output value.</summary>
        /// <param name="input">The input value.</param>
        /// <param name="output">The output value.</param>
        /// <returns>The derviative of the activation function.</returns>
        protected override double ActivationDerivative(double input, double output)
        {
            if (input <= 0.0)
            {
                return this.alpha;
            }
            return 1.0;
        }

        /// <summary>Creates a copy of this instance of the <code>LeakyReluNeuronsString</code> class.</summary>
        /// <returns>The generated instance of the <code>LeakyReluNeuronsString</code> class.</returns>
        public override object Clone()
        {
            return new LeakyReluNeuronsString(this.Length, this.Alpha);
        }
    }
}
