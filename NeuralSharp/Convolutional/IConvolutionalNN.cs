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

using System.Collections.Generic;

namespace NeuralNetwork.Convolutional
{
    /// <summary>Represents an entity which behaves like a convolutional neural network.</summary>
    public interface IConvolutionalNN : IForwardLearner<Image, double[], Image, double[]>
    {
        /// <summary>The amount of outputs of this convolutional neural network.</summary>
        int Outputs { get; }

        /// <summary>The amount of parameters.</summary>
        int Params { get; }

        /// <summary>The depth of the input image.</summary>
        int InputDepth{get;}

        /// <summary>The width of the input image.</summary>
        int InputWidth { get; }

        /// <summary>The hight of the input image.</summary>
        int InputHeight{ get; }

        /// <summary>Feeds an image trough this network and gets an error array for the generated outputs.</summary>
        /// <param name="input">The image to be fed.</param>
        /// <param name="expected">The expected outputs.</param>
        /// <param name="error">The array to be written the error into.</param>
        /// <returns>A value representing how big the error is.</returns>
        double GetError(Image input, double[] expected, double[] error);

        /// <summary>Learns from input and output pairs.</summary>
        /// <param name="inputs">The input images to learn from.</param>
        /// <param name="outputs">The output arrays to learn from.</param>
        /// <param name="maxError">The maximum error value to be aimed for.</param>
        /// <param name="maxSteps">The maximum amount of steps in which to try to reach the maximum error or below.</param>
        /// <returns><code>false</code> if the average error in the last step of training is still above the maximum, <code>true</code> otherwise.</returns>
        bool Learn(IEnumerable<Image> inputs, IEnumerable<double[]> outputs, double maxError, int maxSteps);
    }
}
