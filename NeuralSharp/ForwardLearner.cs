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
using System.Runtime.Serialization;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    /// <summary>Represents an entity capable of relating inputs with arrays.</summary>
    /// <typeparam name="TIn">Type of input</typeparam>
    [DataContract]
    public abstract class ForwardLearner<TIn> : IForwardLearner<TIn, double[], TIn, double[]> where TIn : class
    {
        /// <summary>The default maximum error to be aimed for.</summary>
        protected const double DefaultMaxError = 0.01;

        /// <summary>The default maximum amount of steps in which to try to reach the maximum error.</summary>
        protected const int DefaultMaxSteps = 50000;

        /// <summary>The amount of outputs.</summary>
        public abstract int Outputs { get; }

        /// <summary>Backpropagates the given error trough the learner, updating its parameters.</summary>
        /// <param name="error">The error to be backpropagated. It must refer to the latest feeding process.</param>
        /// <param name="rate">The learning rate at which the parameters are to be updated.</param>
        public abstract void BackPropagate(double[] error, double rate);

        /// <summary>Backpropagates the given error trough the learner.</summary>
        /// <param name="error">The error to be backpropagated. It must refer to the latest feeding process.</param>
        /// <param name="inputError">The array to be written the input error into.</param>
        /// <param name="rate">The learning rate at which the parameters are to be updated.</param>
        public abstract void BackPropagate(double[] error, TIn inputError, double rate);

        /// <summary>Creates a copy of the learner.</summary>
        /// <returns>The generated instance.</returns>
        public abstract object Clone();

        /// <summary>Feeds the given input array trough the learner.</summary>
        /// <param name="input">The input to be fed.</param>
        /// <param name="output">The array to be written the output into.</param>
        public abstract void Feed(TIn input, double[] output);

        /// <summary>Exports this learner to a stream.</summary>
        /// <param name="stream">The stream to be exported into.</param>
        public abstract void Save(Stream stream);

        /// <summary>Returns the learning rate to be used during training at the given step.</summary>
        /// <param name="step">The step.</param>
        /// <returns>The learning rate to be used.</returns>
        protected virtual double GetLearningRate(int step)
        {
            double initial = 0.01;
            //double coefficient = 100;
            //return initial / (1 + (step / coefficient));
            //return initial / Math.Sqrt(step);
            return initial;
        }

        /// <summary>Feeds an input trough this learner and gets an error array for the generated outputs.</summary>
        /// <param name="input">The input to be fed.</param>
        /// <param name="expected">The expected outputs.</param>
        /// <param name="error">The array to be written the error into.</param>
        /// <returns>A value representing how big the error is.</returns>
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

        /// <summary>Learns from the given input and output pairs.</summary>
        /// <param name="inputs">The inputs to learn from.</param>
        /// <param name="outputs">The outputs to learn from.</param>
        /// <param name="maxError">The maximum error to be aimed for.</param>
        /// <param name="maxSteps">The maximum amount of steps in which to try to reach the maximum error or below.</param>
        /// <returns><code>false</code> if, at the last step, the error was greater than the maximum error, <code>true</code> otherwise.</returns>
        public bool Learn(IEnumerable<TIn> inputs, IEnumerable<double[]> outputs, double maxError, int maxSteps)
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
            do
            {
                step++;
                scalarError = 0;
                double rate = this.GetLearningRate(step);
                for (int i = 0; i < entries; i++)
                {
                    var err = this.GetError(inputs.ElementAt(indices[i]), outputs.ElementAt(indices[i]), error);
                    scalarError += err;
                    this.BackPropagate(error, rate);
                }
                scalarError /= entries;
            } while (scalarError > maxError && step < maxSteps);
            return scalarError <= maxError;
        }

        /// <summary>Learns from the given input and output pairs.</summary>
        /// <param name="inputs">The inputs to learn from.</param>
        /// <param name="outputs">The outputs to learn from.</param>
        public virtual void Learn(IEnumerable<TIn> inputs, IEnumerable<double[]> outputs)
        {
            this.Learn(inputs, outputs, ForwardLearner<TIn>.DefaultMaxError, ForwardLearner<TIn>.DefaultMaxSteps);
        }

        /// <summary>Exports this learner to a file.</summary>
        /// <param name="fileName">The name of the file to be exported.</param>
        public virtual void Save(string fileName)
        {
            FileStream fs = new FileStream(fileName, FileMode.Create);
            this.Save(fs);
            fs.Close();
        }
    }
}
