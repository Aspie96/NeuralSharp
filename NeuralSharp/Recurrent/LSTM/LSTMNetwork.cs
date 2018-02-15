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

using System.IO;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Json;

namespace NeuralNetwork.Recurrent.LSTM
{
    /// <summary>Represents an LSTM neural network.</summary>
    [DataContract]
    public class LSTMNetwork : SequenceLearner
    {
        [DataMember]
        private LSTMBlock unit;
        private double[] middle;
        [DataMember]
        private FeedForwardNN fullyConnected;

        /// <summary>Empty constructor. It does not initialize the fields.</summary>
        protected LSTMNetwork() { }
        
        /// <summary>Creates an instance of the <code>LSTMNetwork</code> class.</summary>
        /// <param name="inputs">The amount of inputs of the network.</param>
        /// <param name="memory">The size of the state of the network.</param>
        /// <param name="outputs">The amount of outputs of the network.</param>
        /// <param name="classification">Indicates wether the network is to be used for classification purposes.</param>
        public LSTMNetwork(int inputs, int memory, int outputs, bool classification = false)
        {
            this.unit = new LSTMBlock(inputs, memory);
            this.middle = new double[memory];
            this.fullyConnected = new FeedForwardNN(memory, outputs, classification);
        }

        /// <summary>The amount of outputs of this network.</summary>
        public override int Outputs
        {
            get { return this.fullyConnected.Outputs; }
        }

        /// <summary>Sets an error for this network.</summary>
        /// <param name="error">The error array to be set. It must refer to the latest feeding process.</param>
        public override void BackPropagate(double[] error)
        {
            this.fullyConnected.BackPropagateToDeltas(error, this.middle);
            this.unit.BackPropagate(this.middle);
        }

        /// <summary>Feeds the given input trough this network, updating its state.</summary>
        /// <param name="input">The input to be fed.</param>
        /// <param name="output">The array to be written the output into.</param>
        public override void Feed(double[] input, double[] output)
        {
            this.unit.Feed(input, this.middle);
            this.fullyConnected.Feed(this.middle, output);
        }

        /// <summary>Feeds the given input trough this network, updating its state.</summary>
        /// <param name="input">The input to be fed.</param>
        public override void Feed(double[] input)
        {
            this.unit.Feed(input);
        }

        /// <summary>Applies the weight changes to this network and prepares it for a new sequence.</summary>
        /// <param name="rate">The learning rate at which the parameters are to be updated.</param>
        public override void Reset(double rate)
        {
            this.fullyConnected.ApplyDeltas(rate);
            this.unit.Reset(rate);
        }

        /// <summary>Clones this instance of <code>LSTMNetwork</code> into another.</summary>
        /// <param name="network">The instance to be copied into.</param>
        protected void CloneTo(LSTMNetwork network)
        {
            network.unit = (LSTMBlock)this.unit.Clone();
            network.middle = new double[this.middle.Length];
            network.fullyConnected = (FeedForwardNN)this.fullyConnected.Clone();
        }

        /// <summary>Creates a copy of this instance of the <code>LSTMNetwork</code> class.</summary>
        /// <returns>The generated instance of the <code>LSTMNetwork</code> class.</returns>
        public override object Clone()
        {
            LSTMNetwork retVal = new LSTMNetwork();
            this.CloneTo(retVal);
            return retVal;
        }

        /// <summary>Exports this instance of the <code>LSTMNetwork</code> class to a stream.</summary>
        /// <param name="stream">The stream to be exported into.</param>
        public override void Save(Stream stream)
        {
            new DataContractJsonSerializer(typeof(LSTMNetwork)).WriteObject(stream, this);
        }

        /// <summary>Imports a saved instance of the <code>LSTMNetwork</code> class from a stream.</summary>
        /// <param name="stream">The stream to be imported from.</param>
        /// <returns>The imported instance.</returns>
        public static LSTMNetwork Load(Stream stream)
        {
            return (LSTMNetwork)(new DataContractJsonSerializer(typeof(LSTMNetwork)).ReadObject(stream));
        }

        /// <summary>Imports a saved instance of the <code>LSTMNetwork</code> class from a file.</summary>
        /// <param name="fileName">The name of the file to be imported.</param>
        /// <returns>The imported instance.</returns>
        public static LSTMNetwork Load(string fileName)
        {
            FileStream fs = new FileStream(fileName, FileMode.Open);
            LSTMNetwork retVal = LSTMNetwork.Load(fs);
            fs.Close();
            return retVal;
        }


        [OnDeserialized]
        private void SetValuesOnDeserialized(StreamingContext context)
        {
            this.middle = new double[this.unit.Outputs];
        }
    }
}
