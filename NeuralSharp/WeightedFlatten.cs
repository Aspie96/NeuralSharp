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
    /// <summary>Represents a weighted flat connection matrix.</summary>
    public class WeightedFlatten : FlattenConnectionMatrix
    {
        private double[] weights;
        private double[] deltas;

        /// <summary>Creates a new <code>WeightedFlatten</code> instance.</summary>
        /// <param name="layer1">The input layer of the connection matrix.</param>
        /// <param name="layer2">The otput layer of the connection matrix.</param>
        public WeightedFlatten(ILayer layer1, ILayer layer2) : base(layer1, layer2)
        {
            this.weights = new double[layer1.Length];
            this.deltas = new double[layer1.Length];
            for (int i = 0; i < this.weights.Length; i++)
            {
                this.weights[i] = RandomGenerator.GetNormalNumber(1.0);
            }
        }
        
        /// <summary>The amount of weights of this connection matrix.</summary>
        public override int Params
        {
            get { return this.weights.Length; }
        }

        /// <summary>Feeds the output of the input layer trough this matrix into the output layer.</summary>
        public override void Feed()
        {
            for (int i = 0; i < this.Length; i++)
            {
                this.Layer2.Feed(i, this.weights[i] * this.Layer1.GetLastOutput(i));
            }
        }

        /// <summary>Backpropagates the given error trough this connection matrix.</summary>
        /// <param name="error2">The error array to be backpropagated. It will be modified by being backpropagated trough the output layer.</param>
        /// <param name="error1">The array to be written the error of the input layer into.</param>
        /// <param name="rate">The learning rate at which the weights are to be updated.</param>
        public override void BackPropagate(double[] error2, double[] error1, double rate)
        {
            this.Layer2.BackPropagate(error2);
            for (int i = 0; i < this.Length; i++)
            {
                this.weights[i] += this.weights[i] * this.Layer1.GetLastOutput(i) * rate;
                error1[i] = this.weights[i] * error2[i];
            }
        }

        /// <summary>Backpropagates the given error trough the network and stores the sums the weight gradients to their stored values.</summary>
        /// <param name="error2">The error array to be backpropagated. It will be modified by being backpropagated trough the output layer.</param>
        /// <param name="error1">The array to be written the error of the input layer into.</param>
        public override void BackPropagateToDeltas(double[] error2, double[] error1)
        {
            this.Layer2.BackPropagate(error2);
            for (int i = 0; i < this.Length; i++)
            {
                this.weights[i] += this.weights[i] * this.Layer1.GetLastOutput(i);
                error1[i] = this.weights[i] * error2[i];
            }
        }

        /// <summary>Updates the weights using the stored weight gradients and sets the stored gradients to <code>0</code>.</summary>
        /// <param name="rate">The learning rate at which to the weights are to be updated.</param>
        public override void ApplyDeltas(double rate)
        {
            for (int i = 0; i < this.Outputs; i++)
            {
                this.weights[i] += this.deltas[i] * rate;
            }
        }
    }
}
