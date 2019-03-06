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
    /// <summary>Represents a connection matrix which uses bias.</summary>
    [DataContract]
    public class BiasedConnectionMatrix : ConnectionMatrix
    {
        [DataMember]
        private double[] biases;
        private double[] biasGradients;
        private double[] biasMomentum;
        
        /// <summary>Either creates a siamese of the given <code>BiasedConnectionMatrix</code> instance or clones it.</summary>
        /// <param name="original">The original instance to be created a siamese of or cloned.</param>
        /// <param name="siamese"><code>true</code> if a siamese is to be created, <code>false</code> if a clone is.</param>
        protected BiasedConnectionMatrix(BiasedConnectionMatrix original, bool siamese) : base(original, siamese)
        {
            if (siamese)
            {
                this.biases = original.biases;
                this.biasGradients = original.biasGradients;
                this.biasMomentum = original.biasMomentum;
            }
            else
            {
                this.biases = Backbone.CreateArray<double>(original.OutputSize);
                this.biasGradients = Backbone.CreateArray<double>(original.OutputSize);
                this.biasMomentum = Backbone.CreateArray<double>(original.OutputSize);
            }
        }

        /// <summary>Creates an instance of the <code>BiasedConnectionMatrix</code> class.</summary>
        /// <param name="inputSize">The length of the input of the layer.</param>
        /// <param name="outputSize">The lenght of the output of the layer.</param>
        /// <param name="createIO">Whether the input array and the output array are to be created.</param>
        public BiasedConnectionMatrix(int inputSize, int outputSize, bool createIO = false) : base(inputSize, outputSize, createIO)
        {
            this.biases = Backbone.CreateArray<double>(this.OutputSize);
            this.biasGradients = Backbone.CreateArray<double>(this.OutputSize);
            this.biasMomentum = Backbone.CreateArray<double>(this.OutputSize);
        }

        [OnDeserialized]
        private void SetValuesOnDeserialized(StreamingContext context)
        {
            this.biasGradients = Backbone.CreateArray<double>(this.OutputSize);
            this.biasMomentum = Backbone.CreateArray<double>(this.OutputSize);
        }

        /// <summary>Feeds the layer forward.</summary>
        /// <param name="learning">Whether the layer is being used in a training session. Unused.</param>
        public override void Feed(bool learning = false)
        {
            Backbone.ApplyBiasedConnectionMatrix(this.Input, this.InputSkip, this.InputSize, this.Output, this.OutputSkip, this.OutputSize, this.Weights, this.biases);
        }

        /// <summary>Backpropagates the given error trough the layer.</summary>
        /// <param name="outputErrorArray">The output error to be backpropagated.</param>
        /// <param name="outputErrorSkip">The index of the first position of the output error array to be used.</param>
        /// <param name="inputErrorArray">The array to be written the input error into.</param>
        /// <param name="inputErrorSkip">The index of the first position of the input error array to be used.</param>
        /// <param name="learning"></param>
        public override void BackPropagate(double[] outputErrorArray, int outputErrorSkip, double[] inputErrorArray, int inputErrorSkip, bool learning)
        {
            Backbone.BackpropagateBiasedConnectionMatrix(this.Input, this.InputSkip, this.InputSize, this.Output, this.OutputSkip, this.OutputSize, this.Weights, this.biases, outputErrorArray, outputErrorSkip, inputErrorArray, inputErrorSkip, this.Gradients, this.biasGradients, learning);
        }
        
        /// <summary>Updates the weights of the layer.</summary>
        /// <param name="rate">The learning rate to be used.</param>
        /// <param name="momentum">The momentum to be used.</param>
        public override void UpdateWeights(double rate, double momentum = 0)
        {
            Backbone.UpdateBiasedConnectionMatrix(Weights, Gradients, Momentum, biases, biasGradients, this.biasMomentum, InputSize, OutputSize, rate, momentum);
        }

        /// <summary>Creates a siamese of the layer.</summary>
        /// <returns>The created instance of the <code>BiasedConnectionMatrix</code> class.</returns>
        public override IUntypedLayer CreateSiamese()
        {
            return new BiasedConnectionMatrix(this, true);
        }

        /// <summary>Creates a clone of the layer.</summary>
        /// <returns>The created instance of the <code>BiasedConnectionMatrix</code> class.</returns>
        public override IUntypedLayer Clone()
        {
            return new BiasedConnectionMatrix(this, false);
        }
    }
}
