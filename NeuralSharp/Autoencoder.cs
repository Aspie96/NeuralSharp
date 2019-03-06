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
    /// <typeparam name="T">The input and output type.</typeparam>
    public abstract class Autoencoder<T> : ForwardLearner<T, T>, ILayer<T, T> where T : class
    {
        private ILayer<T, double[]> encoder;
        private ILayer<double[], T> decoder;
        private int codeSize;
        
        /// <summary>Either creates a siamese of the given <code>Autoencoder</code> instance or clones it.</summary>
        /// <param name="original">The original instance to be created a siamese of or cloned.</param>
        /// <param name="siamese"><code>true</code> if a siamese is to be created, <code>false</code> if a clone is.</param>
        protected Autoencoder(Autoencoder<T>original, bool siamese)
        {
            if (siamese)
            {
                this.encoder = (ILayer<T, double[]>)original.encoder.CreateSiamese();
                this.decoder = (ILayer<double[], T>)original.decoder.CreateSiamese();
                this.codeSize = original.CodeSize;
            }
            else
            {
                this.encoder = (ILayer<T, double[]>)original.encoder.Clone();
                this.decoder = (ILayer<double[], T>)original.decoder.Clone();
                this.codeSize = original.CodeSize;
            }
        }

        /// <summary>Creates an instance of the <code>Autoencoder</code> class.</summary>
        /// <param name="encoder">The first part of the autoencoder.</param>
        /// <param name="decoder">The second part of the autoencoder.</param>
        /// <param name="codeSize">The lenght of the output of the first part of the autoencoder and the input of the second.</param>
        protected Autoencoder(ILayer<T, double[]> encoder, ILayer<double[], T> decoder, int codeSize)
        {
            this.encoder = encoder;
            this.decoder = decoder;
            this.codeSize = codeSize;
        }

        /// <summary>The amount of parameters of the autoencoder.</summary>
        public override int Parameters
        {
            get { return this.encoder.Parameters + this.decoder.Parameters; }
        }

        /// <summary>The input object of the learner.</summary>
        public T Input
        {
            get { return this.encoder.Input; }
        }

        /// <summary>The output object of the learner.</summary>
        public T Output
        {
            get { return this.decoder.Output;  }
        }

        /// <summary>The lenght of the output of the encoder and the input of the decoder.</summary>
        public int CodeSize
        {
            get { return this.codeSize; }
        }

        /// <summary>The first part of the autoencoder.</summary>
        protected ILayer<T, double[]> Encoder
        {
            get { return this.encoder; }
        }

        /// <summary>The second part of the autoencoder.</summary>
        protected ILayer<double[], T> Decoder
        {
            get { return this.Decoder; }
        }
        
        /// <summary>Creates a siamese of the learner.</summary>
        /// <returns>The created instance of the <code>Autoencoder</code> class.</returns>
        public abstract IUntypedLayer CreateSiamese();

        /// <summary>Creates a clone of the layer.</summary>
        /// <returns>The created instance of the <code>Autoencoder</code> class.</returns>
        public abstract IUntypedLayer Clone();

        /// <summary>Backpropagates the given error trough the learner.</summary>
        /// <param name="outputError">The output error to be backpropagated.</param>
        /// <param name="inputError">The object to be written the input error into.</param>
        /// <param name="learning">Whether the learner is being used in a training session.</param>
        public override void BackPropagate(T outputError, T inputError, bool learning)
        {
            
        }
        
        /// <summary>Feeds the learner forward.</summary>
        /// <param name="learning">Whether the learner is being used in a training session.</param>
        public void Feed(bool learning = false)
        {
            this.encoder.Feed(learning);
            this.decoder.Feed(learning);
        }

        /// <summary>Exports the autoencoder to a stream.</summary>
        /// <param name="stream">The stream to be exported the autoencoder into.</param>
        public override void Save(Stream stream)
        {
            throw new NotImplementedException();
        }

        /// <summary>Sets the input object and the output object of the learner.</summary>
        /// <param name="input">The input object to be set.</param>
        /// <param name="output">The output object to be set.</param>
        public void SetInputAndOutput(T input, T output)
        {
            double[] array = this.encoder.SetInputGetOutput(input);
            this.decoder.SetInputAndOutput(array, output);
        }

        /// <summary>Sets the input object of the layer and creates and sets an output object.</summary>
        /// <param name="input">The input object to be set.</param>
        /// <returns>The created output object.</returns>
        public T SetInputGetOutput(T input)
        {
            double[] array = this.encoder.SetInputGetOutput(input);
            return this.decoder.SetInputGetOutput(array);
        }

        /// <summary>Updates the weights of the learner.</summary>
        /// <param name="rate">The learning rate to be used.</param>
        /// <param name="momentum">The momentum to be used.</param>
        public override void UpdateWeights(double rate, double momentum = 0)
        {
            this.encoder.UpdateWeights(rate, momentum);
            this.decoder.UpdateWeights(rate, momentum);
        }
    }
}
