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
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace NeuralSharp
{
    /// <summary>Represents a learner which, given the same inputs, always returns the same output.</summary>
    /// <typeparam name="TIn">The input type of the learner.</typeparam>
    /// <typeparam name="TOut">The output type of the learner.</typeparam>
    /// <typeparam name="TErrFunc">The error function type of the learner.</typeparam>
    public abstract class ForwardLearner<TIn, TOut, TErrFunc> where TIn : class where TOut : class where TErrFunc : IError<TOut>
    {
        /// <summary>Represents a function which gets learning parameters at each step.</summary>
        /// <param name="batchSize">The batch size.</param>
        /// <param name="batchIndex">The batch index.</param>
        /// <param name="batchesCount">The batches count.</param>
        /// <param name="epoch">The epoch.</param>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="momentum">The momentum to be used.</param>
        public delegate void LearningParametersFunction(int batchSize, int batchIndex, int batchesCount, int epoch, out float learningRate, out float momentum);
        
        /// <summary>Creates an object which can be used as output error.</summary>
        /// <returns>The created object.</returns>
        protected abstract TOut NewError();

        /// <summary>Feeds the learner forward.</summary>
        /// <param name="input">The object to be read the input from.</param>
        /// <param name="output">The object to be written the output into.</param>
        /// <param name="learning">Whether the learner is being used in a training session.</param>
        public abstract void Feed(TIn input, TOut output, bool learning = false);

        /// <summary>Backpropagates the given error trough the learner.</summary>
        /// <param name="outputError">The output error to be backpropagated.</param>
        /// <param name="learning">Whether the learner is being used in a training session.</param>
        public abstract void BackPropagate(TOut outputError, bool learning = true);

        /// <summary>Backpropagates the given error trough the learner.</summary>
        /// <param name="outputError">The output error to be backpropagated.</param>
        /// <param name="inputError">The object to be written the input error into.</param>
        /// <param name="learning">Whether the learner is being used in a training session.</param>
        public abstract void BackPropagate(TOut outputError, TIn inputError, bool learning);

        /// <summary>Updates the weights of the learner.</summary>
        /// <param name="rate">The learning rate to be used.</param>
        /// <param name="momentum">The momentum to be used.</param>
        public abstract void UpdateWeights(float rate, float momentum = 0.0F);

        /// <summary>Gets the error of the learner, given its actual output and expected output.</summary>
        /// <param name="output">The actual output of the learner.</param>
        /// <param name="expectedOutput">The expected output of the learner.</param>
        /// <param name="error">The object to be written the output error into.</param>
        /// <param name="errorFunction">The error function to be used.</param>
        /// <returns>The output error of the learner.</returns>
        public abstract float GetError(TOut output, TOut expectedOutput, TOut error, TErrFunc errorFunction);

        /// <summary>Feeds the learner forward and gets its error, given the expected output.</summary>
        /// <param name="input">The object to be read the input from.</param>
        /// <param name="expectedOutput">The expected output of the learner.</param>
        /// <param name="error">The object to be written the error into.</param>
        /// <param name="errorFunction">The error function to be used.</param>
        /// <param name="learning">Whether the learner is being used in a training session.</param>
        /// <returns>The error of the learner.</returns>
        public abstract float FeedAndGetError(TIn input, TOut expectedOutput, TOut error, TErrFunc errorFunction, bool learning);

        /// <summary>Gets the best parameters during training.</summary>
        /// <param name="batchSize">The batch size.</param>
        /// <param name="batchIndex">The batch index.</param>
        /// <param name="batchesCount">The amount of batches.</param>
        /// <param name="epoch">The epoch index.</param>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="momentum">The momentum.</param>
        protected virtual void DefaultLearningParameters(int batchSize, int batchIndex, int batchesCount, int epoch, out float learningRate, out float momentum)
        {
            //learningRate = 0.5 / (1 + batchSize * (batchesCount * epoch + batchIndex + 1) * 0.00006);
            learningRate = 0.0001F;// *(float)Math.Exp(-0.000005 * batchSize * (batchesCount * epoch + batchIndex + 1));
            momentum = 0.0F;
        }

        /// <summary>Learns using the given input and output pairs.</summary>
        /// <param name="inputs">The inputs to be learned from.</param>
        /// <param name="outputs">The outputs to be learned from.</param>
        /// <param name="maxError">The maximum error to be aimed for.</param>
        /// <param name="maxSteps">The maximum amount of steps.</param>
        /// <param name="batchSize">The batch size to be used.</param>
        /// <param name="errorFunction">The error function to be used.</param>
        /// <param name="learningParametersFunction">A function getting the learning parameters at each step.</param>
        /// <returns>Whether the maximum accepted error has been reached.</returns>
        public virtual float Learn(IEnumerable<TIn> inputs, IEnumerable<TOut> outputs, float maxError, int maxSteps, int batchSize, TErrFunc errorFunction, LearningParametersFunction learningParametersFunction = null)
        {
            int entries = (Math.Min(inputs.Count(), outputs.Count()) / batchSize) * batchSize;
            int[] indices = new int[entries];
            for (int i = 0; i < entries; i++)
            {
                indices[i] = i;
            }
            //RandomGenerator.ShuffleArray(indices);
            TOut error = this.NewError();
            float errorValue;
            int epoch = 0;
            do
            {
                errorValue = 0;
                for (int i = 0; i < entries; i += batchSize)
                {
                    for (int j = 0; j < batchSize; j++)
                    {
                        errorValue += this.FeedAndGetError(inputs.ElementAt(indices[i + j]), outputs.ElementAt(indices[i + j]), error, errorFunction, true);
                        this.BackPropagate(error, true);

                    }
                    (learningParametersFunction ?? this.DefaultLearningParameters)(batchSize, i / batchSize, entries / batchSize, epoch, out float learningRate, out float momentum);
                    this.UpdateWeights(learningRate * batchSize, momentum);
                }
                errorValue /= entries;
                epoch++;
                //Thread.Sleep(60 * 1000);
                Console.WriteLine(errorValue + " " + epoch);
            } while (epoch < maxSteps && errorValue > maxError);
            return errorValue;
        }
    }
}
