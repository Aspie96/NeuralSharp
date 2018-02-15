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
using System.Linq;

namespace NeuralNetwork.Recurrent
{
    /// <summary>Represents an entity capable of learning sequences of arrays.</summary>
    public abstract class SequenceLearner : RecurrentLearner<double[]>
    {
        /// <summary>Learns from a set of sequences of arrays.</summary>
        /// <param name="sequences">The sequences to learn from.</param>
        /// <param name="maxError">The maximum error to be aimed for.</param>
        /// <param name="maxSteps">The maximum amount of steps in which to try to reach the maximum error or below.</param>
        /// <returns><code>false</code> if at the last step the average error was greater than the maximum error, <code>true</code> otherwise.</returns>
        public bool Learn(IEnumerable<IEnumerable<double[]>> sequences, double maxError, int maxSteps)
        {
            int entries = sequences.Count();
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
                    IEnumerable<double[]> sequence = sequences.ElementAt(indices[i]);
                    for (int j = 0; j < sequence.Count() - 1; j++)
                    {
                        scalarError += this.GetError(sequence.ElementAt(j), sequence.ElementAt(j + 1), error);
                        this.BackPropagate(error);
                    }
                    backfeeds += sequence.Count() - 1;
                    this.Reset(rate);
                }
                scalarError /= backfeeds;
            } while (scalarError > maxError && step < maxSteps);
            return scalarError <= maxError;
        }
    }
}
