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
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    /// <summary>Represents an entity which is capable of learning input to output relationships.</summary>
    /// <typeparam name="TIn">Type of input.</typeparam>
    /// <typeparam name="TOut">Type of output.</typeparam>
    /// <typeparam name="TInErr">Type of input error.</typeparam>
    /// <typeparam name="TOutErr">Type of output error.</typeparam>
    public interface IForwardLearner<TIn, TOut, TInErr, TOutErr> : ICloneable where TIn : class where TOut : class where TInErr : class where TOutErr : class
    {
        /// <summary>Feeds the given input array trough the learner.</summary>
        /// <param name="input">The input to be fed.</param>
        /// <param name="output">The output.</param>
        void Feed(TIn input, TOut output);

        /// <summary>Backpropagates the given error trough the learner, updating its parameters.</summary>
        /// <param name="error">The error to be backpropagated. It must refer to the latest feeding process.</param>
        /// <param name="rate">The learning rate at which the parameters are to be updated.</param>
        void BackPropagate(TOutErr error, double rate);

        /// <summary>Backpropagates the given error trough the learner.</summary>
        /// <param name="error">The error to be backpropagated. It must refer to the latest feeding process.</param>
        /// <param name="inputError">The input error.</param>
        /// <param name="rate">The learning rate at which the parameters are to be updated.</param>
        void BackPropagate(TOutErr error, TInErr inputError, double rate);

        /// <summary>Learns from the given input and output pairs.</summary>
        /// <param name="inputs">The inputs to learn from.</param>
        /// <param name="outputs">The outputs to learn from.</param>
        /// <returns><code>false</code> if, at the last step, the error was greater than the maximum error, <code>true</code> otherwise.</returns>
        void Learn(IEnumerable<TIn> inputs, IEnumerable<TOut> outputs);

        /// <summary>Exports this learner to a stream.</summary>
        /// <param name="stream">The stream to be exported into.</param>
        void Save(Stream stream);

        /// <summary>Exports this learner to a file.</summary>
        /// <param name="fileName">The name of the file to be exported.</param>
        void Save(string fileName);
    }
}
