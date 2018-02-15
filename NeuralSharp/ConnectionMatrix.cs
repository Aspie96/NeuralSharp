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
using System.Runtime.Serialization;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    /// <summary>Represents a matrix of axons between two neurons layers.</summary>
    [DataContract]
    public class ConnectionMatrix : IConnectionMatrix
    {
        private double[,] weights;
        private ILayer layer1;
        private ILayer layer2;
        private double[] flatWeights;
        [DataMember]
        private int inputs;
        [DataMember]
        private int outputs;
        private double[,] deltas;

        /// <summary>Empty constructor. It does not initialize the fields.</summary>
        protected ConnectionMatrix() { }

        /// <summary>Creates a new instance of the <code>ConnectionMatrix</code> class.</summary>
        /// <param name="layer1">The input layer of the connection matrix.</param>
        /// <param name="layer2">The output layer of the connection matrix.</param>
        public ConnectionMatrix(ILayer layer1, ILayer layer2)
        {
            this.inputs = layer1.Length;
            this.outputs = layer2.Length;
            this.layer1 = layer1;
            this.layer2 = layer2;
            this.weights = new double[layer1.Length, layer2.Length];
            this.deltas = new double[layer1.Length, layer2.Length];
            double variance = 2.0 / (this.Inputs + this.Outputs);
            for (int i = 0; i < this.inputs; i++)
            {
                for (int j = 0; j < this.Outputs; j++)
                {
                    this.weights[i, j] = RandomGenerator.GetNormalNumber(variance);
                }
            }
        }
        
        /// <summary>The lenght of the input layer of this connection matrix.</summary>
        public int Inputs
        {
            get { return this.inputs; }
        }

        /// <summary>The lenght of the output layer of this connection matrix.</summary>
        public int Outputs
        {
            get { return this.outputs; }
        }

        [DataMember]
        private double[] FlatWeights
        {
            get
            {
                double[] retVal = new double[this.weights.Length];
                Buffer.BlockCopy(this.weights, 0, retVal, 0, sizeof(double) * this.weights.Length);
                return retVal;
            }
            set { this.flatWeights = value; }
        }
        
        /// <summary>The input layer of this connection matrix.</summary>
        public ILayer Layer1
        {
            get { return this.layer1; }
        }

        /// <summary>The output layer of this connection matrix.</summary>
        public ILayer Layer2
        {
            get { return this.layer2; }
        }

        /// <summary>The weights of this connection matrix.</summary>
        protected double[,] Weights
        {
            get { return this.weights; }
        }

        /// <summary>The stored weight gradients.</summary>
        protected double[,] Deltas
        {
            get { return this.deltas; }
        }
        
        /// <summary>The amount of weights in this connection matrix.</summary>
        public int Params
        {
            get { return this.weights.Length; }
        }

        /// <summary>Sets the input and the output layer for this connection matrix. Only to be used when strictly necessary.</summary>
        /// <param name="layer1">The input layer to be set.</param>
        /// <param name="layer2">The output layer to be set.</param>
        public void SetLayers(ILayer layer1, ILayer layer2)
        {
            this.layer1 = layer1;
            this.layer2 = layer2;
        }

        /// <summary>Feeds the output of the input layer trough this connection matrix into the output layer.</summary>
        public virtual void Feed()
        {
            for (int i = 0; i < this.Layer2.Length; i++)
            {
                double sum = 0.0;
                for (int j = 0; j < this.Layer1.Length; j++)
                {
                    sum += this.Weights[j, i] * this.Layer1.GetLastOutput(j);
                }
                this.Layer2.Feed(i, sum);
            }
            this.Layer2.FeedEnd();
        }

        /// <summary>Backpropagates the given error trough this connection matrix, updating its weights.</summary>
        /// <param name="error2">The error array to be backpropagated. It will be modified by being backpropagated trough the output layer.</param>
        /// <param name="error1">The array to be written the error of the input layer into.</param>
        /// <param name="rate">The learning rate at which the weights are to be updated.</param>
        public virtual void BackPropagate(double[] error2, double[] error1, double rate)
        {
            this.Layer2.BackPropagate(error2);
            for (int i = 0; i < this.Inputs; i++)
            {
                error1[i] = 0;
                for (int j = 0; j < this.Outputs; j++)
                {
                    error1[i] += (int)(error2[j] * this.Weights[i, j] * 1000) / 1000.0;
                    this.Weights[i, j] += (int)(rate * this.Layer1.GetLastOutput(i) * error2[j] * 1000) / 1000.0;
                }
            }
        }

        /// <summary>Backpropagates the given error trough the network and stores the sums the weight gradients to their stored values.</summary>
        /// <param name="error2">The error array to be backpropagated. It will be modified by being backpropagated trough the output layer.</param>
        /// <param name="error1">The array to be written the error of the input layer into.</param>
        public virtual void BackPropagateToDeltas(double[] error2, double[] error1)
        {
            this.Layer2.BackPropagate(error2);
            for (int i = 0; i < this.Inputs; i++)
            {
                error1[i] = 0;
                for (int j = 0; j < this.Outputs; j++)
                {
                    error1[i] += error2[j] * this.Weights[i, j];
                    this.Deltas[i, j] += this.Layer1.GetLastOutput(i) * error2[j];
                }
            }
        }

        /// <summary>Updates the weights using the stored weight gradients and sets the stored gradients to <code>0</code>.</summary>
        /// <param name="rate">The learning rate at which to the weights are to be updated.</param>
        public virtual void ApplyDeltas(double rate)
        {
            for (int i = 0; i < this.Inputs; i++)
            {
                for (int j = 0; j < this.Outputs; j++)
                {
                    this.Weights[i, j] += this.Deltas[i, j] * rate;
                    this.Deltas[i, j] = 0.0;
                }
            }
        }

        [OnDeserialized]
        private void SetValuesOnDeserialized(StreamingContext context)
        {
            this.weights = new double[this.Inputs, this.Outputs];
            this.deltas = new double[this.Inputs, this.Outputs];
            Buffer.BlockCopy(this.flatWeights, 0, this.Weights, 0, sizeof(double) * this.flatWeights.Length);
            this.FlatWeights = null;
        }

        /// <summary>Copises this instance of the <code>ConnectionMatrix</code> class into another.</summary>
        /// <param name="connectionMatrix">The connection matrix to be copied into.</param>
        /// <param name="layer1">The input layer of the copied instance.</param>
        /// <param name="layer2">The output layer of the copied instance.</param>
        protected virtual void CloneTo(ConnectionMatrix connectionMatrix, ILayer layer1, ILayer layer2)
        {
            connectionMatrix.inputs = layer1.Length;
            connectionMatrix.outputs = layer2.Length;
            connectionMatrix.layer1 = layer1;
            connectionMatrix.layer2 = layer2;
            connectionMatrix.weights = (double[,])this.Weights.Clone();
            connectionMatrix.deltas = new double[layer1.Length, layer2.Length];
        }

        /// <summary>Creates a copy of this instance of the <code>ConnectionMatrix</code> class.</summary>
        /// <param name="layer1">The input layer of the new instance.</param>
        /// <param name="layer2">The output layer of the new instance.</param>
        /// <returns>The generated instance of the <code>ConnectionMatrix</code> class.</returns>
        public virtual IConnectionMatrix Clone(ILayer layer1, ILayer layer2)
        {
            ConnectionMatrix retVal = new ConnectionMatrix();
            this.CloneTo(retVal, layer1, layer2);
            return retVal;
        }
    }
}
