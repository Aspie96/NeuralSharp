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
using System.IO;

namespace NeuralNetwork.Recurrent
{
    /// <summary>Represents an entity which behaves like a learner with an inner state.</summary>
    /// <typeparam name="TIn">The type of input.</typeparam>
    /// <typeparam name="TOut">The type of output.</typeparam>
    /// <typeparam name="TInErr">The type of input error.</typeparam>
    /// <typeparam name="TOutErr">The type of output error.</typeparam>
    public interface IRecurrentLearner<TIn, TOut, TInErr, TOutErr> : ICloneable where TIn : class where TOut : class where TInErr : class where TOutErr : class
    {
        /// <summary>Feeds the given input trough this learner, updating its state.</summary>
        /// <param name="input">The input to be fed.</param>
        /// <param name="output">The output.</param>
        void Feed(TIn input, TOut output);

        /// <summary>Feeds the given input trough this learner, updating its state.</summary>
        /// <param name="input">The input to be fed.</param>
        void Feed(TIn input);

        /// <summary>Sets an error for this learner.</summary>
        /// <param name="error">The error to be set. It must refer to the latest feeding process.</param>
        void BackPropagate(TOutErr error);

        /// <summary>Applies the parameter changes to the learner and prepares it for a new sequence.</summary>
        /// <param name="rate">The learning rate at which the parameters are to be updated.</param>
        void Reset(double rate);

        /// <summary>Exports this learner to a file.</summary>
        /// <param name="stream">The stream to be exported into.</param>
        void Save(Stream stream);

        /// <summary>Exports this instance of the <code>PurelyConvolutionalNN</code> to a file.</summary>
        /// <param name="fileName">The name of the file to be created.</param>
        void Save(string fileName);
    }
}
