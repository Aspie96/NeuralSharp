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
using System.Text;
using System.Threading.Tasks;

namespace NeuralSharp
{
    /// <summary>Represents a layer.</summary>
    public interface IUntypedLayer
    {
        /// <summary>The amount of parameters of the layer.</summary>
        int Parameters { get; }

        /// <summary>Feeds the layer forward.</summary>
        /// <param name="learning">Whether the layer is being used in a training session.</param>
        void Feed(bool learning = false);

        /// <summary>Updates the weights of the layer.</summary>
        /// <param name="rate">The learning rate to be used.</param>
        /// <param name="momentum">The momentum to be used.</param>
        void UpdateWeights(double rate, double momentum = 0.0);

        /// <summary>Creates a siamese of the layer.</summary>
        /// <returns>The created siamese.</returns>
        IUntypedLayer CreateSiamese();

        /// <summary>Clones the layer.</summary>
        /// <returns>The created clone.</returns>
        IUntypedLayer Clone();
    }
}
