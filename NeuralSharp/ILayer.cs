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
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    /// <summary>Represents a layer in a neural network.</summary>
    public interface ILayer : ICloneable
    {
        /// <summary>The lenght of the layer.</summary>
        int Length { get; }

        /// <summary>Feeds a value to a neuron of the layer.</summary>
        /// <param name="index">The value to be fed.</param>
        /// <param name="input">The index of the neuron to be fed to.</param>
        void Feed(int index, double input);

        /// <summary>Backpropagates an error trough this layer, updating it accoring to the derivatives of the neurons.</summary>
        /// <param name="error">The eror to be backpropagated.</param>
        /// <param name="skip">The amount of positions to skip in the error array.</param>
        void BackPropagate(double[] error, int skip = 0);

        /// <summary>Returns the latest output of a neuron of this layer.</summary>
        /// <param name="index">The index of the neuron.</param>
        /// <returns>The latest output of the chosen neuron.</returns>
        double GetLastOutput(int index);

        /// <summary>Gets the latest outputs of the neurons of this layer.</summary>
        /// <param name="output">The array to be written the output into.</param>
        /// <param name="skip">The amount of positions to skip in the output array.</param>
        void GetLastOutput(double[] output, int skip = 0);

        /// <summary>Returns the latest input of a neuron of this layer.</summary>
        /// <param name="index">The index of the neuron.</param>
        /// <returns>The latest input fed to the chosen neuron.</returns>
        double GetLastInput(int index);

        /// <summary>Gets the latest inputs fed to this layer.</summary>
        /// <param name="input">The array to be written the input into.</param>
        /// <param name="skip">The amount of positions to be skipped in the input array.</param>
        void GetLastInput(double[] input, int skip = 0);

        /// <summary>Feeds the given input trough the layer.</summary>
        /// <param name="input">The input array to be fed.</param>
        /// <param name="skip">The amount of positions to be skipped in the input array.</param>
        void Feed(double[] input, int skip = 0);

        /// <summary>Method to be called after every neuron of the layer has been feed.</summary>
        void FeedEnd();
    }
}
