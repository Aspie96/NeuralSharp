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
    /// <summary>Represents a neurons string whose activation function is LeakyRelu.</summary>
    public class LeakyReluNeuronsString : NeuronsString
    {
        private float alpha;

        /// <summary>Either creates a siamese of the given <code>LeakyReluNeuronsString</code> class or clones it.</summary>
        /// <param name="original">The original instance to be created a siamese of or cloned.</param>
        /// <param name="siamese"><code>true</code> if a siamese is to be created, <code>false</code> if a clone is.</param>
        protected LeakyReluNeuronsString(LeakyReluNeuronsString original, bool siamese) : base(original, siamese)
        {
            this.alpha = original.Alpha;
        }

        /// <summary>Creates an instance of the <code>LeakyReluNeuronsString</code> class.</summary>
        /// <param name="length">The lenght of the layer.</param>
        /// <param name="alpha">The alpha coefficient of the layer.</param>
        /// <param name="createIO">Whether the input array and the output array are to be created.</param>
        public LeakyReluNeuronsString(int length, float alpha, bool createIO = false) : base(length, createIO)
        {
            this.alpha = alpha;
        }
        
        /// <summary>The slope of the layer.</summary>
        public float Alpha
        {
            get { return this.alpha; }
        }

        /// <summary>The leaky relu function.</summary>
        /// <param name="input">The input of the function.</param>
        /// <returns>The output of the function.</returns>
        protected override float Activation(float input)
        {
            return (input < 0.0F ? input * this.Alpha : input);
        }

        /// <summary>The derivative of the leaky relu function.</summary>
        /// <param name="input">The input of the function.</param>
        /// <param name="output">The output of the function.</param>
        /// <returns>The derivative of the function.</returns>
        protected override float ActivationDerivative(float input, float output)
        {
            return (output > 0.0F ? 1.0F : this.Alpha);
        }

        /// <summary>Cretes a siamese of the layer.</summary>
        /// <returns>The created instance of the <code>LeakyReluNeuronsString</code> class.</returns>
        public override ILayer<float[], float[]> CreateSiamese()
        {
            return new LeakyReluNeuronsString(this, true);
        }

        /// <summary>Creates a clone of the layer.</summary>
        /// <returns>The created instance of the <code>LeakyReluNeuronsString</code> class.</returns>
        public override ILayer<float[], float[]> Clone()
        {
            return new LeakyReluNeuronsString(this, false);
        }
    }
}
