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
    /// <summary>Represents a flat connection matrix.</summary>
    [DataContract]
    public class FlattenConnectionMatrix : IConnectionMatrix
    {
        private ILayer layer1;
        private ILayer layer2;

        /// <summary>Creates a new <code>FlattenConnectionMatrix</code> instance.</summary>
        /// <param name="layer1">The input layer of the connection matrix.</param>
        /// <param name="layer2">The otput layer of the connection matrix.</param>
        public FlattenConnectionMatrix(ILayer layer1, ILayer layer2)
        {
            this.layer1 = layer1;
            this.layer2 = layer2;
        }

        /// <summary>The lenght of the input layer of this connection matrix.</summary>
        public int Inputs
        {
            get { return this.Layer1.Length; }
        }

        /// <summary>The lenght of the output layer of this connection matrix.</summary>
        public int Outputs
        {
            get { return this.Layer2.Length; }
        }

        /// <summary>The lenght of the input and output layer of this connection matrix.</summary>
        public int Length
        {
            get { return this.Inputs; }
        }

        /// <summary>The input layer of this connection matrix.</summary>
        protected internal ILayer Layer1
        {
            get { return this.layer1; }
        }

        /// <summary>The output layer of this connection matrix.</summary>
        protected internal ILayer Layer2
        {
            get { return this.layer2; }
        }

        /// <summary>The weights of this connection matrix.</summary>
        public virtual int Params
        {
            get { return 0; }
        }

        /// <summary>Sets the input and the output layer for this connection matrix. Only to be used when strictly necessary.</summary>
        /// <param name="layer1">The input layer to be set.</param>
        /// <param name="layer2">The output layer to be set.</param>
        public void SetLayers(ILayer layer1, ILayer layer2)
        {
            this.layer1 = layer1;
            this.layer2 = layer2;
        }
        
        /// <summary>Feeds the output of the input layer trought this connection matrix into the output matrix.</summary>
        public virtual void Feed()
        {
            for (int i = 0; i < this.Length; i++)
            {
                this.Layer2.Feed(i, this.Layer1.GetLastOutput(i));
            }
            this.Layer2.FeedEnd();
        }

        /// <summary>Backpropagates the given error trough this connection matrix.</summary>
        /// <param name="error2">The error array to be backpropagated. It will be modified by being backpropagated trough the output layer.</param>
        /// <param name="error1">The array to be written the error of the input layer into.</param>
        /// <param name="rate">The learning rate at which the weights are to be updated.</param>
        public virtual void BackPropagate(double[] error2, double[] error1, double rate)
        {
            this.Layer2.BackPropagate(error2);
            Array.Copy(error2, error1, this.Length);
        }

        /// <summary>Backpropagates the given error trough the network and stores the sums the weight gradients to their stored values.</summary>
        /// <param name="error2">The error array to be backpropagated. It will be modified by being backpropagated trough the output layer.</param>
        /// <param name="error1">The array to be written the error of the input layer into.</param>
        public virtual void BackPropagateToDeltas(double[] error2, double[] error1)
        {
            this.Layer2.BackPropagate(error2);
            Array.Copy(error2, error1, this.Length);
        }

        /// <summary>Updates the weights using the stored weight gradients and sets the stored gradients to <code>0</code>.</summary>
        /// <param name="rate">The learning rate at which to the weights are to be updated.</param>
        public virtual void ApplyDeltas(double rate) { }

        /// <summary>Creates a copy of this instance of the <code>FlattenConnectionMatrix</code> class.</summary>
        /// <param name="layer1">The input layer of the new instance.</param>
        /// <param name="layer2">The output layer of the new instance.</param>
        /// <returns>The generated instance of the <code>FlattenConnectionMatrix</code> class.</returns>
        public virtual IConnectionMatrix Clone(ILayer layer1, ILayer layer2)
        {
            return new FlattenConnectionMatrix(layer1, layer2);
        }
    }
}
