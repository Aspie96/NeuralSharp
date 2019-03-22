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
    /// <summary>Represents a layer in a learner.</summary>
    /// <typeparam name="TIn">The input type of the layer.</typeparam>
    /// <typeparam name="TOut">The output type of the layer.</typeparam>
    public interface ILayer<TIn, TOut> where TIn : class where TOut : class
    {
        /// <summary>The input object of the layer.</summary>
        TIn Input { get; }

        /// <summary>The output object.</summary>
        TOut Output { get; }

        /// <summary>The amount of parameters of the layer.</summary>
        int Parameters { get; }

        /// <summary>The siamese identifier of the layer.</summary>
        object SiameseID { get; }

        /// <summary>Counts the amount of parameters of the layer.</summary>
        /// <param name="siameseIDs">The siamese identifiers to be excluded.</param>
        /// <returns>The amount of parameters of the layer.</returns>
        int CountParameters(List<object> siameseIDs);

        /// <summary>Feeds the layer forward.</summary>
        /// <param name="learning">Whether the layer is being used in a training session.</param>
        void Feed(bool learning = false);

        /// <summary>Updates the weights of the layer.</summary>
        /// <param name="rate">The learning rate to be used.</param>
        /// <param name="momentum">The momentum to be used.</param>
        void UpdateWeights(float rate, float momentum = 0.0F);
        
        /// <summary>Backpropagates the given error trough the layer.</summary>
        /// <param name="outputError">The output error to be backpropagated.</param>
        /// <param name="inputError">The object to be written the input error into.</param>
        /// <param name="learning">Whether the layer is being used in a training session.</param>
        void BackPropagate(TOut outputError, TIn inputError, bool learning);

        /// <summary>Sets the input object and the output object of the layer.</summary>
        /// <param name="input">The input object to be set.</param>
        /// <param name="output">The output object to be set.</param>
        void SetInputAndOutput(TIn input, TOut output);

        /// <summary>Sets the input object of the layer and creates and sets the output object.</summary>
        /// <param name="input">The input object to be set.</param>
        /// <returns>The created output object.</returns>
        TOut SetInputGetOutput(TIn input);

        /// <summary>Creates a siamese of the layer.</summary>
        /// <returns>The created siamese.</returns>
        ILayer<TIn, TOut> CreateSiamese();

        /// <summary>Clones the layer.</summary>
        /// <returns>The clone.</returns>
        ILayer<TIn, TOut> Clone();
    }
}
