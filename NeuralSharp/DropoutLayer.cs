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
using System.Collections.ObjectModel;
using System.Linq;
using System.Runtime.Serialization;
using System.Text;
using System.Threading.Tasks;

namespace NeuralSharp
{
    /// <summary>Represents a dropout layer.</summary>
    public class DropoutLayer : NeuronsString
    {
        private bool[] dropped;
        private float dropChance;

        /// <summary>Either creates a siamese of the given <code>DropoutLayer</code> instance or clones it.</summary>
        /// <param name="original">The original instance to be created a siamese of or cloned.</param>
        /// <param name="siamese"><code>true</code> if a siamese is to be created, <code>false</code> if a clone is.</param>
        protected DropoutLayer(DropoutLayer original, bool siamese) : base(original, siamese)
        {
            this.dropped = Backbone.CreateArray<bool>(original.Length);
            this.dropChance = original.DropChance;
        }

        /// <summary>Creates an instance of the <code>DropoutLayer</code> class.</summary>
        /// <param name="length">The lenght of the layer.</param>
        /// <param name="dropChance">The dropout chance of the layer.</param>
        /// <param name="createIO">Whether the input array and the output array of the layer are to be created.</param>
        public DropoutLayer(int length, float dropChance, bool createIO = false) : base(length, createIO)
        {
            this.dropped = Backbone.CreateArray<bool>(this.Length);
            this.dropChance = dropChance;
        }

        /// <summary>The dropout chance of the layer.</summary>
        public float DropChance
        {
            get { return this.dropChance; }
        }

        /// <summary>Feeds this layer forward.</summary>
        /// <param name="learning">Whether the layer is being used in a training session.</param>
        public override void Feed(bool learning)
        {
            Backbone.ApplyDropout(this.Input, this.InputSkip, this.Output, this.OutputSkip, this.Length, this.dropped, this.DropChance, learning);
        }

        /// <summary>Backpropagates the given error trough the layer.</summary>
        /// <param name="outputErrorArray">The output error to be backpropagated.</param>
        /// <param name="outputErrorSkip">The index of the first entry of the output error array to be used.</param>
        /// <param name="inputErrorArray">The array to be written the input error into.</param>
        /// <param name="inputErrorSkip">The index of the first entry of the input error array to be used.</param>
        /// <param name="learning">Whether the layer is being used in a training session.</param>
        public override void BackPropagate(float[] outputErrorArray, int outputErrorSkip, float[] inputErrorArray, int inputErrorSkip, bool learning)
        {
            Backbone.BackpropagateDropout(this.Input, this.InputSkip, this.Output, this.OutputSkip, this.Length, this.dropped, this.DropChance, learning, outputErrorArray, outputErrorSkip, inputErrorArray, inputErrorSkip);
        }

        /// <summary>Creates a siamese of the layer.</summary>
        /// <returns>The created instance of the <code>DropoutLayer</code> class.</returns>
        public override ILayer<float[], float[]> CreateSiamese()
        {
            return new DropoutLayer(this, true);
        }

        /// <summary>Creates a clone of the layer.</summary>
        /// <returns>The created instance of the <code>DropoutLayer</code> class.</returns>
        public override ILayer<float[], float[]> Clone()
        {
            return new DropoutLayer(this, false);
        }
    }
}
