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
using System.Collections.ObjectModel;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    /// <summary>Represents a layer containing several segments.</summary>
    public class CompoundLayer : ILayer
    {
        private int length;
        private ILayer[] layers;

        /// <summary>Empty constructor. It does not initialize the fields.</summary>
        private CompoundLayer() { }

        /// <summary>Creates a new instance of the <code>CompoundLayer</code> class.</summary>
        /// <param name="layers">The segments of the layer to be joined.</param>
        public CompoundLayer(params ILayer[] layers)
        {
            this.layers = (ILayer[])layers.Clone();
            this.length = 0;
            for (int i = 0; i < layers.Length; i++)
            {
                this.length += layers[i].Length;
            }
        }

        /// <summary>The amount of neurons in this layer.</summary>
        public int Length
        {
            get { return this.length; }
        }

        /// <summary>Backpropagates the given error trough this layer.</summary>
        /// <param name="error">The error array to be backpropagated. It will be updated using the derivatives of the activation functions.</param>
        /// <param name="skip">The amount of positions to be skipped within the error array.</param>
        public void BackPropagate(double[] error, int skip = 0)
        {
            int index = skip;
            for (int i = 0; i < this.layers.Length; i++)
            {
                this.layers[i].BackPropagate(error, index);
                index += this.layers[i].Length;
            }
        }

        /// <summary>Feeds an value to a neuron.</summary>
        /// <param name="index">The value to be fed.</param>
        /// <param name="input">The index of the neuron to be fed to.</param>
        public void Feed(int index, double input)
        {
            int layer = 0;
            while (this.layers[layer].Length <= index)
            {
                layer++;
                index -= this.layers[layer].Length;
            }
            this.layers[layer].Feed(index, input);
        }

        /// <summary>Feeds an input array to the layer.</summary>
        /// <param name="input">The input array to be fed.</param>
        /// <param name="skip">The amount of positions to be skipped within the input array.</param>
        public void Feed(double[] input, int skip = 0)
        {
            int index = skip;
            for (int i = 0; i < this.layers.Length; i++)
            {
                this.layers[i].Feed(input, index);
                index += this.layers[i].Length;
            }
        }

        /// <summary>Returns the latest output of a neuron within this layer.</summary>
        /// <param name="index">The index of the chosen neuron.</param>
        /// <returns>The latest output of the chosen neuron.</returns>
        public double GetLastOutput(int index)
        {
            foreach (var item in this.layers)
            {
                if (index < item.Length)
                {
                    return item.GetLastOutput(index);
                }
                index -= item.Length;
            }
            int layer = 0;
            while (this.layers[layer].Length <= index)
            {
                index -= this.layers[layer++].Length;
            }
            return this.layers[layer].GetLastOutput(index);
        }

        /// <summary>Gets the latest outputs of this layer.</summary>
        /// <param name="output">The array to be written the output into.</param>
        /// <param name="skip">The amount of position to be skipped in the output array.</param>
        public void GetLastOutput(double[] output, int skip = 0)
        {
            int index = skip;
            for (int i = 0; i < this.layers.Length; i++)
            {
                this.layers[i].GetLastOutput(output, index);
                index += this.layers[i].Length;
            }
        }

        /// <summary>Returns the latest input fed to a neuron of this layer.</summary>
        /// <param name="index">The index of the neuron.</param>
        /// <returns>The latest input fed to the chosen neuron.</returns>
        public double GetLastInput(int index)
        {
            int layer = 0;
            while (this.layers[layer].Length <= index)
            {
                index -= this.layers[layer++].Length;
            }
            return this.layers[layer].GetLastInput(index);
        }

        /// <summary>Gets the latest inputs fed trough this layer.</summary>
        /// <param name="input">The array to be written the input into.</param>
        /// <param name="skip">The amount of positions to be skipped in the input array.</param>
        public void GetLastInput(double[] input, int skip = 0)
        {
            int index = skip;
            for (int i = 0; i < this.layers.Length; i++)
            {
                this.layers[i].GetLastInput(input, index);
                index += this.layers[i].Length;
            }
        }

        /// <summary>Copies this instance of the <code>CompoundLayer</code> into another. It creates a copy of each inner segment.</summary>
        /// <param name="layer">The instance to be copied into.</param>
        protected void CloneTo(CompoundLayer layer)
        {
            layer.length = this.length;
            layer.layers = new ILayer[this.layers.Length];
            for (int i = 0; i < this.layers.Length; i++)
            {
                layer.layers[i] = (ILayer)layer.layers[i].Clone();
            }
        }

        /// <summary>Cretes a copy of this instance of the <code>CompundLayer</code> class.</summary>
        /// <returns>The generated copy of the <code>CompoundLayer</code> class. It contains a copy of each inner segment.</returns>
        public virtual object Clone()
        {
            CompoundLayer retVal = new CompoundLayer();
            this.CloneTo(retVal);
            return retVal;
        }

        /// <summary>Method to be called after every neuron of the layer has been fed.</summary>
        public virtual void FeedEnd()
        {
            for (int i = 0; i < this.layers.Length; i++)
            {
                this.layers[i].FeedEnd();
            }
        }
    }
}
