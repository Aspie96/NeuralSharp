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
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralSharp
{
    /// <summary>Represents a learner whose input and output have the same type and size and of which the input or output of at least one inner layer is an array.</summary>
    /// <typeparam name="TData">The input and output type.</typeparam>
    /// <typeparam name="TErrFunc">The error function type.</typeparam>
    public abstract class Autoencoder<TData, TErrFunc> : ForwardLearner<TData, TData, TErrFunc>, ILayer<TData, TData> where TData : class where TErrFunc : IError<TData>
    {
        private ILayer<TData, float[]> encoder;
        private ILayer<float[], TData> decoder;
        private float[] error;
        private int codeSize;
        private object siameseID;

        /// <summary>Either creates a siamese of the given <code>Autoencoder</code> instance or clones it.</summary>
        /// <param name="original">The original instance to be created a siamese of or cloned.</param>
        /// <param name="siamese"><code>true</code> if a siamese is to be created, <code>false</code> if a clone is.</param>
        protected Autoencoder(Autoencoder<TData, TErrFunc> original, bool siamese)
        {
            if (siamese)
            {
                this.encoder = original.encoder.CreateSiamese();
                this.decoder = original.decoder.CreateSiamese();
                this.codeSize = original.CodeSize;
                this.error = Backbone.CreateArray<float>(original.CodeSize);
                this.siameseID = original.SiameseID;
            }
            else
            {
                this.encoder = original.encoder.Clone();
                this.decoder = original.decoder.Clone();
                this.codeSize = original.CodeSize;
                this.error = Backbone.CreateArray<float>(original.CodeSize);
                this.siameseID = new object();
            }
        }

        /// <summary>Creates an instance of the <code>Autoencoder</code> class.</summary>
        /// <param name="encoder">The first part of the autoencoder.</param>
        /// <param name="decoder">The second part of the autoencoder.</param>
        /// <param name="codeSize">The lenght of the output of the first part of the autoencoder and the input of the second.</param>
        public Autoencoder(ILayer<TData, float[]> encoder, ILayer<float[], TData> decoder, int codeSize)
        {
            this.encoder = encoder;
            this.decoder = decoder;
            this.codeSize = codeSize;
            this.error = Backbone.CreateArray<float>(codeSize);
            this.siameseID = new object();
        }

        /// <summary>The amount of parameters of the autoencoder.</summary>
        public int Parameters
        {
            get { return this.encoder.Parameters + this.decoder.Parameters; }
        }

        /// <summary>The input object of the learner.</summary>
        public TData Input
        {
            get { return this.encoder.Input; }
        }

        /// <summary>The output object of the learner.</summary>
        public TData Output
        {
            get { return this.decoder.Output; }
        }

        /// <summary>The lenght of the output of the encoder and the input of the decoder.</summary>
        public int CodeSize
        {
            get { return this.codeSize; }
        }

        /// <summary>The first part of the autoencoder.</summary>
        protected ILayer<TData, float[]> Encoder
        {
            get { return this.encoder; }
        }

        /// <summary>The second part of the autoencoder.</summary>
        protected ILayer<float[], TData> Decoder
        {
            get { return this.decoder; }
        }

        /// <summary>The error.</summary>
        protected float[] Error
        {
            get { return this.error; }
        }

        /// <summary>The siamese identificator of the learner.</summary>
        public object SiameseID
        {
            get { return this.siameseID; }
        }

        /// <summary>Creates a siamese of the learner.</summary>
        /// <returns>The created instance of the <code>Autoencoder</code> class.</returns>
        public abstract ILayer<TData, TData> CreateSiamese();

        /// <summary>Creates a clone of the layer.</summary>
        /// <returns>The created instance of the <code>Autoencoder</code> class.</returns>
        public abstract ILayer<TData, TData> Clone();

        /// <summary>Backpropagates the given error trough the learner.</summary>
        /// <param name="outputError">The output error to be backpropagated.</param>
        /// <param name="inputError">The object to be written the input error into.</param>
        /// <param name="learning">Whether the learner is being used in a training session.</param>
        public override void BackPropagate(TData outputError, TData inputError, bool learning)
        {
            this.decoder.BackPropagate(outputError, this.Error, learning);
            this.encoder.BackPropagate(this.Error, inputError, learning);
        }

        /// <summary>Feeds the learner forward.</summary>
        /// <param name="learning">Whether the learner is being used in a training session.</param>
        public void Feed(bool learning = false)
        {
            this.encoder.Feed(learning);
            this.decoder.Feed(learning);
        }

        /// <summary>Sets the input object and the output object of the learner.</summary>
        /// <param name="input">The input object to be set.</param>
        /// <param name="output">The output object to be set.</param>
        public void SetInputAndOutput(TData input, TData output)
        {
            float[] array = this.encoder.SetInputGetOutput(input);
            this.decoder.SetInputAndOutput(array, output);
        }

        /// <summary>Sets the input object of the layer and creates and sets an output object.</summary>
        /// <param name="input">The input object to be set.</param>
        /// <returns>The created output object.</returns>
        public TData SetInputGetOutput(TData input)
        {
            float[] array = this.encoder.SetInputGetOutput(input);
            return this.decoder.SetInputGetOutput(array);
        }

        /// <summary>Updates the weights of the learner.</summary>
        /// <param name="rate">The learning rate to be used.</param>
        /// <param name="momentum">The momentum to be used.</param>
        public override void UpdateWeights(float rate, float momentum = 0)
        {
            this.encoder.UpdateWeights(rate, momentum);
            this.decoder.UpdateWeights(rate, momentum);
        }

        /// <summary>Counts the amount of parameters of the learner.</summary>
        /// <param name="siameseIDs">The siamese identificators to be excluded. The siamese identificators of the learner will be added to the list.</param>
        /// <returns>The amount of parameters.</returns>
        public int CountParameters(List<object> siameseIDs)
        {
            if (siameseIDs.Contains(this.SiameseID))
            {
                return 0;
            }
            siameseIDs.Add(this.SiameseID);
            return this.Encoder.CountParameters(siameseIDs) + this.decoder.CountParameters(siameseIDs);
        }
    }
}
