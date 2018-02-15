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
    /// <summary>Represents a connection matrix in a neural network.</summary>
    public interface IConnectionMatrix
    {
        /// <summary>The lenght of the input layer of this connection matrix.</summary>
        int Inputs { get; }

        /// <summary>The lenght of the output layer of this connection matrix.</summary>
        int Outputs { get; }

        /// <summary>The amount of parameters in this connection matrix.</summary>
        int Params { get; }

        /// <summary>Sets the input and the output layer for this connection matrix. Only to be used when strictly necessary.</summary>
        /// <param name="layer1">The input layer to be set.</param>
        /// <param name="layer2">The output layer to be set.</param>
        void SetLayers(ILayer layer1, ILayer layer2);

        /// <summary>Feeds the output of the input layer trough this connection matrix into the output layer.</summary>
        void Feed();

        /// <summary>Backpropagates the given error trough this connection matrix, updating its parameters.</summary>
        /// <param name="error2">The error array to be backpropagated. It will be modified by being backpropagated trough the output layer.</param>
        /// <param name="error1">The array to be written the error of the input layer into.</param>
        /// <param name="rate">The learning rate at which the parameters are to be updated.</param>
        void BackPropagate(double[] error2, double[] error1, double rate);

        /// <summary>Backpropagates the given error trough the network and stores the sums the parameters gradients to their stored values.</summary>
        /// <param name="error2">The error array to be backpropagated. It will be modified by being backpropagated trough the output layer.</param>
        /// <param name="error1">The array to be written the error of the input layer into.</param>
        void BackPropagateToDeltas(double[] error2, double[] error1);

        /// <summary>Updates the parameters using the stored parameters gradients and sets the stored gradients to <code>0</code>.</summary>
        /// <param name="rate">The learning rate at which to the parameters are to be updated.</param>
        void ApplyDeltas(double rate);

        /// <summary>Creates a copy of this connection matrix.</summary>
        /// <param name="layer1">The input layer of the new instance.</param>
        /// <param name="layer2">The output layer of the new instance.</param>
        /// <returns>The generated instance.</returns>
        IConnectionMatrix Clone(ILayer layer1, ILayer layer2);
    }
}
