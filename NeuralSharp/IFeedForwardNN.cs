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
    /// <summary>Represents a neural network.</summary>
    public interface IFeedForwardNN : IForwardLearner<double[], double[], double[], double[]>
    {
        /// <summary>The amount of inputs of this network.</summary>
        int Inputs { get; }

        /// <summary>The amount of outputs of this network.</summary>
        int Outputs { get; }

        /// <summary>The amount of parameters of this network.</summary>
        int Params { get; }

        /// <summary>Feeds the given inputs trough the network.</summary>
        /// <param name="inputs">Inputs to be fed.</param>
        void Feed(params double[] inputs);

        /// <summary>Feeds an input trough this network and gets an error array for the generated outputs.</summary>
        /// <param name="input">The input to be fed.</param>
        /// <param name="expected">The expected outputs.</param>
        /// <param name="error">The array to be written the error into.</param>
        /// <returns>A value representing how big the error is.</returns>
        double GetError(double[] input, double[] expected, double[] error);
    }
}
