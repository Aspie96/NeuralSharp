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
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    /// <summary>Represents a bias neuron.</summary>
    public class BiasNeuron : FreeNeuron
    {
        /// <summary>Creates a new instance of the <code>BiasNeuron</code> class.</summary>
        public BiasNeuron()
        {
            this.Feed(0.0);
        }

        /// <summary>Returns the value of the activation function for an input.</summary>
        /// <param name="input">The input to be given to the activation function.</param>
        /// <returns>Always <code>1</code>.</returns>
        protected override double Activation(double input)
        {
            return 1.0;
        }

        /// <summary>Returns the derivative of the activation function for the given input and output.</summary>
        /// <param name="input">The input value of the activation function.</param>
        /// <param name="output">The output value of the activation function.</param>
        /// <returns>Always <code>0</code>.</returns>
        protected override double ActivationDerivative(double input, double output)
        {
            return 0.0;
        }
    
        /// <summary>Creates a copy of this instance of the <code>BiasNeuron</code> class.</summary>
        /// <returns>The generated instance of the <code>BiasNeuron</code> class.</returns>
        public override object Clone()
        {
            return new BiasNeuron();
        }
    }
}
