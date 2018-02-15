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

using System.Runtime.Serialization;

namespace NeuralNetwork
{
    /// <summary>Represents a connection matrix with a bias neuron.</summary>
    [DataContract]
    public class BiasedConnectionMatrix : ConnectionMatrix
    {
        [DataMember]
        private double[] biases;
        private double[] biasDeltas;

        /// <summary>Empty constructor. It does not initialize the fields.</summary>
        protected BiasedConnectionMatrix() { }

        /// <summary>Creates a new instance of the <code>BiasedConnectionMatrix</code> class.</summary>
        /// <param name="layer1">The input layer of the connection matrix.</param>
        /// <param name="layer2">The output layer of the connection matrix.</param>
        public BiasedConnectionMatrix(ILayer layer1, ILayer layer2) : base(layer1, layer2)
        {
            this.biases = new double[layer2.Length];
            this.biasDeltas = new double[layer2.Length];
            double variance = 2.0 / (this.Inputs + this.Outputs);
            for (int i = 0; i < biases.Length; i++)
            {
                biases[i] = RandomGenerator.GetNormalNumber(variance);
            }
        }

        /// <summary>Backpropagates the given error trough this connection matrix, updating its weights.</summary>
        /// <param name="error2">The error array to be backpropagated. It will be modified by being backpropagated trough the output layer.</param>
        /// <param name="error1">The array to be written the error of the input layer into.</param>
        /// <param name="rate">The learning rate at which the weights are to be updated.</param>
        public override void BackPropagate(double[] error2, double[] error1, double rate)
        {
            base.BackPropagate(error2, error1, rate);
            if (rate != 0.0)
            {
                for (int i = 0; i < this.Outputs; i++)
                {
                    this.biases[i] += rate * error2[i];
                }
            }
        }

        /// <summary>Backpropagates the given error trough the network and stores the sums the weight gradients to their stored values.</summary>
        /// <param name="error2">The error array to be backpropagated. It will be modified by being backpropagated trough the output layer.</param>
        /// <param name="error1">The array to be written the error of the input layer into.</param>
        public override void BackPropagateToDeltas(double[] error2, double[] error1)
        {
            base.BackPropagateToDeltas(error2, error1);
            for (int i = 0; i < this.Outputs; i++)
            {
                this.biasDeltas[i] += error2[i];
            }
        }

        /// <summary>Feeds the output of the input layer trough this connection matrix into the output layer.</summary>
        public override void Feed()
        {
            for (int i = 0; i < this.Layer2.Length; i++)
            {
                double sum = 0.0;
                for (int j = 0; j < this.Layer1.Length; j++)
                {
                    sum += this.Weights[j, i] * this.Layer1.GetLastOutput(j);
                }
                sum += this.biases[i];
                this.Layer2.Feed(i, sum);
            }
            this.Layer2.FeedEnd();
        }

        /// <summary>Updates the weights using the stored weight gradients and sets the stored gradients to <code>0</code>.</summary>
        /// <param name="rate">The learning rate at which to the weights are to be updated.</param>
        public override void ApplyDeltas(double rate)
        {
            base.ApplyDeltas(rate);
            for (int i = 0; i < this.Outputs; i++)
            {
                this.biases[i] += this.biasDeltas[i] * rate;
                this.biasDeltas[i] = 0;
            }
        }

        [OnDeserialized]
        private void SetValuesOnDeserialized(StreamingContext context)
        {
            this.biasDeltas = new double[this.Outputs];
        }

        /// <summary>Copises this instance of the <code>BiasedConnectionMatrix</code> class into another.</summary>
        /// <param name="connectionMatrix">The connection matrix to be copied into.</param>
        /// <param name="layer1">The input layer of the copied instance.</param>
        /// <param name="layer2">The output layer of the copied instance.</param>
        protected void CloneTo(BiasedConnectionMatrix connectionMatrix, ILayer layer1, ILayer layer2)
        {
            base.CloneTo(connectionMatrix, layer1, layer2);
            connectionMatrix.biases = (double[])this.biases.Clone();
            connectionMatrix.biasDeltas = new double[layer2.Length];
        }

        /// <summary>Creates a copy of this instance of the <code>BiasedConnectionMatrix</code> class.</summary>
        /// <param name="layer1">The input layer of the new instance.</param>
        /// <param name="layer2">The output layer of the new instance.</param>
        /// <returns>The generated instance of the <code>BiasedConnectionMatrix</code> class.</returns>
        public override IConnectionMatrix Clone(ILayer layer1, ILayer layer2)
        {
            BiasedConnectionMatrix retVal = new BiasedConnectionMatrix();
            this.CloneTo(retVal, layer1, layer2);
            return retVal;
        }
    }
}
