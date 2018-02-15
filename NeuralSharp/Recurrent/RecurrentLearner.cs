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

namespace NeuralNetwork.Recurrent
{
    /// <summary>Represents a recurrent learner which returns an array.</summary>
    /// <typeparam name="TIn">The type of input.</typeparam>
    public abstract class RecurrentLearner<TIn> : IRecurrentLearner<TIn, double[], TIn, double[]> where TIn : class
    {
        /// <summary>The amount of outputs.</summary>
        public abstract int Outputs { get; }

        /// <summary>Sets an error for this learner.</summary>
        /// <param name="error">The error array to be set. It must refer to the latest feeding process.</param>
        public abstract void BackPropagate(double[] error);

        /// <summary>Creates a copy of this learner.</summary>
        /// <returns>The generated instance.</returns>
        public abstract object Clone();

        /// <summary>Feeds the given input trough this learner, updating its state.</summary>
        /// <param name="input">The input to be fed.</param>
        /// <param name="output">The array to be written the output into.</param>
        public abstract void Feed(TIn input, double[] output);

        /// <summary>Feeds the given input trough this learner, updating its state.</summary>
        /// <param name="input">The input to be fed.</param>
        public abstract void Feed(TIn input);

        /// <summary>Applies the parameter changes to this learner and prepares it for a new sequence.</summary>
        /// <param name="rate">The learning rate at which the parameters are to be updated.</param>
        public abstract void Reset(double rate);

        /// <summary>Exports the learner to a stream.</summary>
        /// <param name="stream">The stream to be exported into.</param>
        public abstract void Save(Stream stream);

        /// <summary>Returns the learning rate to be used for training.</summary>
        /// <returns>The learning rate to be used for training.</returns>
        protected virtual double GetLearningRate()
        {
            return 0.01;
        }

        /// <summary>Feeds an input trough this network and gets an error array for the generated outputs.</summary>
        /// <param name="input">The input to be fed.</param>
        /// <param name="expected">The expected outputs.</param>
        /// <param name="error">The array to be written the error into.</param>
        /// <returns>A number representing how big the error is.</returns>
        public virtual double GetError(TIn input, double[] expected, double[] error)
        {
            double retVal = 0;
            this.Feed(input, error);
            for (int i = 0; i < this.Outputs; i++)
            {
                error[i] = expected[i] - error[i];
                retVal += error[i] * error[i];
            }
            retVal = Math.Sqrt(retVal);
            return retVal;
        }

        /// <summary>Learns from sequences of inputs and outputs.</summary>
        /// <param name="inputs">The sequences of inputs to learn from.</param>
        /// <param name="outputs">The outputs to learn from.</param>
        /// <param name="maxError">The maximum error value to be aimed for.</param>
        /// <param name="maxSteps">The maximum amount of steps in which to try to reach the maximum error or below.</param>
        /// <returns><code>false</code> if at the last step the average error was greater than the maximum error, <code>true</code> otherwise.</returns>
        public virtual bool Learn(IEnumerable<IEnumerable<TIn>> inputs, IEnumerable<double[]> outputs, double maxError, int maxSteps)
        {
            int entries = inputs.Count();
            int[] indices = new int[entries];
            for (int i = 0; i < entries; i++)
            {
                indices[i] = i;
            }
            double[] error = new double[this.Outputs];
            double scalarError;
            int step = 0;
            RandomGenerator.ShuffleArray(indices);
            this.Reset(0.0);
            do
            {
                step++;
                scalarError = 0;
                double rate = this.GetLearningRate();
                for (int i = 0; i < entries; i++)
                {
                    IEnumerable<TIn> inputSequence = inputs.ElementAt(indices[i]);
                    for (int j = 0; j < inputSequence.Count() - 1; j++)
                    {
                        this.Feed(inputSequence.ElementAt(j));
                    }
                    var err = this.GetError(inputSequence.ElementAt(inputSequence.Count() - 1), outputs.ElementAt(indices[i]), error);
                    scalarError += err;
                    this.BackPropagate(error);
                    this.Reset(rate);
                }
                scalarError /= entries;
            } while (scalarError > maxError && step < maxSteps);
            return scalarError <= maxError;
        }

        /// <summary>Learns from sequences of inputs and sequences of outputs.</summary>
        /// <param name="inputs">The sequences of inputs to learn from.</param>
        /// <param name="outputs">The sequences of outputs to learn from.</param>
        /// <param name="maxError">The maximum error value to be aimed for.</param>
        /// <param name="maxSteps">The maximum amount of steps in which to try to reach the maximum error or below.</param>
        /// <returns><code>false</code> if at the last step the average error was greater than the maximum error, <code>true</code> otherwise.</returns>
        public virtual bool Learn(IEnumerable<IEnumerable<TIn>> inputs, IEnumerable<IEnumerable<double[]>> outputs, double maxError, int maxSteps)
        {
            int entries = inputs.Count();
            int[] indices = new int[entries];
            for (int i = 0; i < entries; i++)
            {
                indices[i] = i;
            }
            double[] error = new double[this.Outputs];
            double scalarError;
            int step = 0;
            RandomGenerator.ShuffleArray(indices);
            this.Reset(0.0);
            do
            {
                int backfeeds = 0;
                step++;
                scalarError = 0;
                double rate = this.GetLearningRate();
                for (int i = 0; i < entries; i++)
                {
                    IEnumerable<TIn> inputSequence = inputs.ElementAt(indices[i]);
                    IEnumerable<double[]> outputSequence = outputs.ElementAt(indices[i]);
                    for (int j = 0; j < inputSequence.Count(); j++)
                    {
                        double[] output = outputSequence.ElementAt(j);
                        if (output == null)
                        {
                            this.Feed(inputSequence.ElementAt(j));
                        }
                        else
                        {
                            scalarError += this.GetError(inputSequence.ElementAt(j), output, error);
                            this.BackPropagate(error);
                            backfeeds++;
                        }
                    }
                    this.Reset(rate);
                }
                scalarError /= backfeeds;
            } while (scalarError > maxError && step < maxSteps);
            return scalarError <= maxError;
        }

        /// <summary>Exports this learner to a file.</summary>
        /// <param name="fileName">The name of the file to be created.</param>
        public virtual void Save(string fileName)
        {
            FileStream fs = new FileStream(fileName, FileMode.Create);
            this.Save(fs);
            fs.Close();
        }
    }
}
