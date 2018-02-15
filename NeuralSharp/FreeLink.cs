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
    /// <summary>Represents a link between two free neurons.</summary>
    public class FreeLink : IConnectionMatrix
    {
        private FreeNeuron neuron1;
        private FreeNeuron neuron2;
        private double weight;
        private double delta;

        /// <summary>Empty constructor. It does not initialize the fields.</summary>
        protected FreeLink() { }

        /// <summary>Creates a new instance of the <code>FreeNeuron</code> class.</summary>
        /// <param name="neuron1">The input neuron. It must be an instance of the <code>FreeNeuron</code> class.</param>
        /// <param name="neuron2">The output neuron. It must be an instance of the <code>FreeNeuron</code> class.</param>
        public FreeLink(FreeNeuron neuron1, FreeNeuron neuron2)
        {
            this.neuron1 = neuron1;
            this.neuron2 = neuron2;
            this.weight = RandomGenerator.GetDouble() * 2 - 1;
        }

        /// <summary>Always <code>1</code>.</summary>
        public int Inputs
        {
            get { return 1; }
        }

        /// <summary>Always <code>1</code>.</summary>
        public int Outputs
        {
            get { return 1; }
        }

        /// <summary>The weight of this connection.</summary>
        public double Weight
        {
            get { return this.weight; }
            set { this.weight = value; }
        }

        /// <summary>Always <code>1</code>.</summary>
        public int Params
        {
            get { return 1; }
        }

        /// <summary>Sets the input and the output neuron. Only to be used when strictly necessary.</summary>
        /// <param name="layer1">The input neuron to be set.</param>
        /// <param name="layer2">The output neuron to be set.</param>
        public void SetLayers(ILayer layer1, ILayer layer2)
        {
            this.neuron1 = (FreeNeuron)layer1;
            this.neuron2 = (FreeNeuron)layer2;
        }

        /// <summary>Backpropagates the given error trough this connection, updating its weight.</summary>
        /// <param name="error2">The error to be backpropagated. It is updated by being backpropagated trough the output neuron.</param>
        /// <param name="error1">The array to be written the input array into.</param>
        /// <param name="rate">The learning rate at which to the weight is to be updated.</param>
        public void BackPropagate(double[] error2, double[] error1, double rate)
        {
            this.neuron2.BackPropagate(error2);
            error1[0] = error2[0] * this.weight;
            this.weight += rate * this.neuron1.LastOutput * error2[0];
        }

        /// <summary>Backpropagates the given error trough this connection and sums the weight gradient to the stored value.</summary>
        /// <param name="error2">The error to be backpropagated. It is updated by being backpropagated trough the output neuron.</param>
        /// <param name="error1">The array to be written the input error into.</param>
        public void BackPropagateToDeltas(double[] error2, double[] error1)
        {
            this.neuron2.BackPropagate(error2);
            error1[0] = error2[0] * this.weight;
            this.delta += this.neuron1.LastOutput * error2[0];
        }

        /// <summary>Updates the weight using the stored gradient value and sets the gradient to <code>0</code>.</summary>
        /// <param name="rate">The learning rate at which the weight is to be updated.</param>
        public void ApplyDeltas(double rate)
        {
            this.weight += this.delta * rate;
            this.delta = 0;
        }

        /// <summary>Feeds the output of the input neuron trough this connection into the output neuron.</summary>
        public void Feed()
        {
            this.neuron2.Feed(this.neuron1.LastOutput);
        }

        /// <summary>Copises this instance of the <code>FreeNeuron</code> class into another.</summary>
        /// <param name="freeLink">The connection matrix to be copied into.</param>
        /// <param name="layer1">The input layer of the copied instance. It must be an instance of the <code>FreeNeuron</code> class.</param>
        /// <param name="layer2">The output layer of the copied instance. It must be an instance of the <code>FreeNeuron</code> class.</param>
        protected void CloneTo(FreeLink freeLink, ILayer layer1, ILayer layer2)
        {
            freeLink.neuron1 = (FreeNeuron)layer1;
            freeLink.neuron2 = (FreeNeuron)layer2;
            freeLink.weight = this.weight;
        }

        /// <summary>Creates a copy of this instance of the <code>FreeNeuron</code> class.</summary>
        /// <param name="layer1">The input layer of the new instance. It must be an instance of the <code>FreeNeuron</code> class.</param>
        /// <param name="layer2">The output layer of the new instance. It must be an instance of the <code>FreeNeuron</code> class.</param>
        /// <returns>The generated instance of the <code>FreeNeuron</code> class.</returns>
        public virtual IConnectionMatrix Clone(ILayer layer1, ILayer layer2)
        {
            FreeLink retVal = new FreeLink();
            this.CloneTo(retVal, layer1, layer2);
            return retVal;
        }
    }
}
