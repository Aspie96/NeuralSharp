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
using System.IO;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Json;

namespace NeuralNetwork.Recurrent
{
    /// <summary>Represents an Elman neural network.</summary>
    [DataContract]
    [KnownType(typeof(LeakyReluNeuronsString))]
    [KnownType(typeof(SoftmaxNeuronsString))]
    public class ElmanNetwork : SequenceLearner
    {
        [DataMember]
        private LinearNeuronsString inputLayer;
        [DataMember]
        private NeuronsString outputLayer;
        [DataMember]
        private NeuronsString hiddenLayer;
        [DataMember]
        private LinearNeuronsString contextLayer;
        private CompoundLayer firstLayer;
        [DataMember]
        private BiasedConnectionMatrix matrix1;
        [DataMember]
        private FlattenConnectionMatrix matrix2;
        [DataMember]
        private BiasedConnectionMatrix matrix3;
        private ArrayInfiniteTimeMemory unfolded;
        private ArrayInfiniteTimeMemory unfoldedErrors;
        [DataMember]
        private ConnectionMatrix startingMatrix;
        private double[] error1;
        private double[] error2;

        /// <summary>Empty constructor. It does not initialize the fields.</summary>
        protected ElmanNetwork() { }

        /// <summary>Creates a new instance of the <code>ElmanNetwork</code> class.</summary>
        /// <param name="inputs">The number of inputs of the network.</param>
        /// <param name="hidden">The number of newurons in the hidden layer. Also the size of the inner state of the network.</param>
        /// <param name="outputs">The number of outputs of the networks.</param>
        /// <param name="classification">Indicates wether the network is to be used for classifcation purposes.</param>
        public ElmanNetwork(int inputs, int hidden, int outputs, bool classification = false)
        {
            this.inputLayer = new LinearNeuronsString(inputs);
            this.hiddenLayer = new LeakyReluNeuronsString(hidden, 0.1);
            if (classification)
            {
                this.outputLayer = new SoftmaxNeuronsString(outputs);
            }
            else
            {
                this.outputLayer = new NeuronsString(outputs);
            }
            this.contextLayer = new LinearNeuronsString(hidden);
            this.firstLayer = new CompoundLayer(this.contextLayer, this.inputLayer);
            this.matrix1 = new BiasedConnectionMatrix(this.firstLayer, this.hiddenLayer);
            this.matrix2 = new FlattenConnectionMatrix(this.hiddenLayer, this.contextLayer);
            this.matrix3 = new BiasedConnectionMatrix(this.hiddenLayer, this.outputLayer);
            this.unfolded = new ArrayInfiniteTimeMemory(this.firstLayer.Length);
            this.unfoldedErrors = new ArrayInfiniteTimeMemory(hidden);
            this.startingMatrix = new ConnectionMatrix(new BiasNeuron(), this.contextLayer);
            this.error1 = new double[Math.Max(this.firstLayer.Length, outputs)];
            this.error2 = new double[Math.Max(this.firstLayer.Length, outputs)];
            this.startingMatrix.Feed();
        }

        /// <summary>The amount of weights of this network.</summary>
        public int Params
        {
            get { return this.matrix1.Params + this.matrix2.Params + this.matrix3.Params + this.startingMatrix.Params; }
        }

        /// <summary>The amount of inputs of this network.</summary>
        public int Inputs
        {
            get { return this.inputLayer.Length; }
        }

        /// <summary>The amount of outputs of this network.</summary>
        public override int Outputs
        {
            get { return this.outputLayer.Length; }
        }

        /// <summary>Feeds the given input trough this network, updating its state.</summary>
        /// <param name="input">The array to be fed.</param>
        /// <param name="output">The array to be written the output into.</param>
        public override void Feed(double[] input, double[] output)
        {
            this.Feed(input);
            this.matrix3.Feed();
            this.outputLayer.GetLastOutput(output);
            this.unfoldedErrors.Add(true);
        }
        
        /// <summary>Feeds the given input trough this network, updating its state.</summary>
        /// <param name="input">The array to be fed.</param>
        public override void Feed(double[] input)
        {
            this.inputLayer.Feed(input);
            double[] mem = unfolded.Add();
            this.firstLayer.GetLastInput(mem);
            this.matrix1.Feed();
            this.matrix2.Feed();
        }

        /// <summary>The size of the hidden layer in this network. Also the size of its inner state.</summary>
        public int Hidden
        {
            get { return this.hiddenLayer.Length; }
        }
        
        /// <summary>Sets an error for this network.</summary>
        /// <param name="error">The error array to be set. It must refer to the latest feeding process.</param>
        public override void BackPropagate(double[] error)
        {
            double[] unfoldedError = new double[this.Hidden];
            this.matrix3.BackPropagateToDeltas(error, error1);
            Array.Copy(unfoldedError, this.unfoldedErrors[this.unfoldedErrors.Length - 1], unfoldedError.Length);
        }
        
        /// <summary>Applies the weight changes to the network and prepares it for a new sequence.</summary>
        /// <param name="rate">The learning rate at which the weights are to be updated.</param>
        public override void Reset(double rate)
        {
            this.matrix3.ApplyDeltas(rate);
            Array.Clear(this.error1, 0, this.error1.Length);
            for (int i = 0; i < this.unfolded.Length; i++)
            {
                int step = this.unfolded.Length - i - 1;
                
                for (int j = 0; j < this.Hidden; j++)
                {
                    this.error2[j] = this.unfoldedErrors[step][j] + this.error1[j];
                }
                this.firstLayer.Feed(this.unfolded[step]);
                this.matrix1.Feed();
                this.matrix1.BackPropagateToDeltas(this.error2, this.error1);
                double[] aux = this.error1;
                this.error1 = this.error2;
                this.error2 = aux;
            }
            this.startingMatrix.BackPropagate(this.error2, this.error1, rate);
            this.matrix1.ApplyDeltas(rate);


            this.unfolded.Clear();
            this.unfoldedErrors.Clear();
            this.startingMatrix.Feed();
        }

        [OnDeserialized]
        private void SetValuesOnDeserialized(StreamingContext context)
        {
            this.firstLayer = new CompoundLayer(this.contextLayer, this.inputLayer);
            this.error1 = new double[Math.Max(this.firstLayer.Length, this.outputLayer.Length)];
            this.error2 = new double[Math.Max(this.firstLayer.Length, this.outputLayer.Length)];
            this.unfolded = new ArrayInfiniteTimeMemory(this.firstLayer.Length);
            this.matrix1.SetLayers(this.firstLayer, this.hiddenLayer);
            this.matrix2.SetLayers(this.hiddenLayer, this.contextLayer);
            this.matrix3.SetLayers(this.hiddenLayer, this.outputLayer);
            this.startingMatrix.SetLayers(new BiasNeuron(), this.contextLayer);
            this.startingMatrix.Feed();
        }

        /// <summary>Exports this instance of the <code>ElmanNetwork</code> class to a stream.</summary>
        /// <param name="stream">The stream to be exported into.</param>
        public override void Save(Stream stream)
        {
            new DataContractJsonSerializer(typeof(ElmanNetwork)).WriteObject(stream, this);
        }

        /// <summary>Imports an instance of the <code>ElmanNetwork</code> class from a stream.</summary>
        /// <param name="stream">The stream to be imported from.</param>
        /// <returns>The imported instance.</returns>
        public static ElmanNetwork Load(Stream stream)
        {
            return (ElmanNetwork)(new DataContractJsonSerializer(typeof(ElmanNetwork)).ReadObject(stream));
        }

        /// <summary>Imports an instance of the <code>ElmanNetwork</code> class from a file.</summary>
        /// <param name="fileName">The name of the file to be imported.</param>
        /// <returns>The imported instance.</returns>
        public static ElmanNetwork Load(string fileName)
        {
            FileStream fs = new FileStream(fileName, FileMode.Open);
            ElmanNetwork retVal = ElmanNetwork.Load(fs);
            fs.Close();
            return retVal;
        }

        /// <summary>Copies this instance of <code>ElmanNetwork</code> into anothers.</summary>
        /// <param name="network">Instance to be copied into.</param>
        protected void CloneTo(ElmanNetwork network)
        {
            network.inputLayer = (LinearNeuronsString)this.inputLayer.Clone();
            network.hiddenLayer = (NeuronsString)this.hiddenLayer.Clone();
            network.outputLayer = (NeuronsString)this.outputLayer.Clone();
            network.contextLayer = (LinearNeuronsString)this.contextLayer.Clone();
            network.firstLayer = new CompoundLayer(network.contextLayer, network.inputLayer);
            network.matrix1 = (BiasedConnectionMatrix)this.matrix1.Clone(network.firstLayer, network.hiddenLayer);
            network.matrix2 = (FlattenConnectionMatrix)this.matrix2.Clone(network.hiddenLayer, network.contextLayer);
            network.matrix3 = (BiasedConnectionMatrix)this.matrix3.Clone(network.hiddenLayer, network.outputLayer);
            network.unfolded = new ArrayInfiniteTimeMemory(this.firstLayer.Length);
            network.startingMatrix = (ConnectionMatrix)this.startingMatrix.Clone(new BiasNeuron(), network.contextLayer);
            network.error1 = new double[this.error1.Length];
            network.error2 = new double[this.error1.Length];
            network.startingMatrix.Feed();
        }

        /// <summary>Creates a copy of this instance of the <code>ElmanNetwork</code> class.</summary>
        /// <returns>The generated instance of the <code>ElmanNetwork</code> class.</returns>
        public override object Clone()
        {
            ElmanNetwork retVal = new ElmanNetwork();
            this.CloneTo(retVal);
            return retVal;
        }
    }
}
