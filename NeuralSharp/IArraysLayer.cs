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
    /// <summary>Represents a layer whose input and output are arrays.</summary>
    public interface IArraysLayer : ILayer<double[], double[]>
    {
        /// <summary>The length of the input of the layer.</summary>
        int InputSize { get; }

        /// <summary>The index of the first used entry of the input array.</summary>
        int InputSkip { get; }

        /// <summary>The length of the output of the layer.</summary>
        int OutputSize { get; }

        /// <summary>The index of the first used entry of the output array.</summary>
        int OutputSkip { get; }

        /// <summary>Backpropagates the given error trough the layer.</summary>
        /// <param name="outputErrorArray">The output error to be backpropagated.</param>
        /// <param name="outputErrorSkip">The index of the first entry of the output error array to be used.</param>
        /// <param name="inputErrorArray">The array to be written the input error into.</param>
        /// <param name="inputErrorSkip">The index of the first entry of the input error array to be used.</param>
        /// <param name="learning">Whether the layer is being used in a training session.</param>
        void BackPropagate(double[] outputErrorArray, int outputErrorSkip, double[] inputErrorArray, int inputErrorSkip, bool learning);

        /// <summary>Sets the input array and the output array of the layer.</summary>
        /// <param name="inputArray">The input array to be set.</param>
        /// <param name="inputSkip">The index of the first entry of the input array to be used.</param>
        /// <param name="outputArray">The output array to be set.</param>
        /// <param name="outputSkip">The index of the first entry of the output array to be used.</param>
        void SetInputAndOutput(double[] inputArray, int inputSkip, double[] outputArray, int outputSkip);

        /// <summary>Sets the input array of the layer and creates and sets the output array.</summary>
        /// <param name="inputArray">The input array to be set.</param>
        /// <param name="inputSkip">The index of the first entry of the input array to be used.</param>
        /// <returns>The created output array.</returns>
        double[] SetInputGetOutput(double[] inputArray, int inputSkip);
    }
}
