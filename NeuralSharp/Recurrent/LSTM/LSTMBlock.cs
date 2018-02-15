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

namespace NeuralNetwork.Recurrent.LSTM
{
    /// <summary>Represents a LSTM unit.</summary>
    [DataContract]
    public class LSTMBlock : SequenceLearner
    {
        [DataMember]
        private LinearNeuronsString input;
        [DataMember]
        private LinearNeuronsString output;
        [DataMember]
        private TanhNeuronsString inputActivation;
        [DataMember]
        private NeuronsString inputGate;
        [DataMember]
        private NeuronsString forgetGate;
        [DataMember]
        private NeuronsString outputGate;
        private CompoundLayer gates;
        [DataMember]
        private LinearNeuronsString context;
        private CompoundLayer firstLayer;
        [DataMember]
        private LinearNeuronsString state;
        [DataMember]
        private BiasedConnectionMatrix w;
        [DataMember]
        private FlattenConnectionMatrix contextMatrix;
        private double[] error1;
        private double[] error2;
        private ArrayInfiniteTimeMemory unfolded;
        private ArrayInfiniteTimeMemory unfoldedOutptuts;
        private ArrayInfiniteTimeMemory unfoldedGates;
        private ArrayInfiniteTimeMemory unfoldedStates;
        private ArrayInfiniteTimeMemory errors;
        [DataMember]
        private bool peephole;

        /// <summary>Empty constructor. It does not initialize the fields.</summary>
        protected LSTMBlock() { }

        /// <summary>Creates a new instance of the <code>LSTMBlock</code> class.</summary>
        /// <param name="inputs">The amount of inputs of the block.</param>
        /// <param name="outputs">The amount of utputs of the block.</param>
        /// <param name="peephole">Indicates wether a peephole is to be used.</param>
        public LSTMBlock(int inputs, int outputs, bool peephole = false)
        {
            this.input = new LinearNeuronsString(inputs);
            this.output = new LinearNeuronsString(outputs);
            this.peephole = peephole;
            this.inputActivation = new TanhNeuronsString(outputs);
            this.inputGate = new NeuronsString(outputs);
            this.forgetGate = new NeuronsString(outputs);
            this.outputGate = new NeuronsString(outputs);
            this.gates = new CompoundLayer(this.inputActivation, this.inputGate, this.forgetGate, this.outputGate);
            this.state = new LinearNeuronsString(outputs);
            this.output = new LinearNeuronsString(outputs);
            this.context = new LinearNeuronsString(outputs);
            this.firstLayer = new CompoundLayer(this.context, this.input);
            this.w = new BiasedConnectionMatrix(this.firstLayer, this.gates);
            if (peephole)
            {
                this.contextMatrix = new FlattenConnectionMatrix(this.state, this.context);
            }
            else
            {
                this.contextMatrix = new FlattenConnectionMatrix(this.output, this.context);
            }
            this.error1 = new double[this.gates.Length];
            this.error2 = new double[this.gates.Length];
            this.unfolded = new ArrayInfiniteTimeMemory(inputs);
            this.unfoldedOutptuts = new ArrayInfiniteTimeMemory(outputs);
            this.unfoldedOutptuts.Add(new double[this.Outputs]);
            this.unfoldedGates = new ArrayInfiniteTimeMemory(this.gates.Length);
            this.unfoldedStates = new ArrayInfiniteTimeMemory(outputs);
            this.unfoldedStates.Add(new double[this.Outputs]);
            this.errors = new ArrayInfiniteTimeMemory(outputs);
        }

        /// <summary>The amount of inputs of this block.</summary>
        public int Inputs
        {
            get { return this.input.Length; }
        }

        /// <summary>The amount of outputs of this block.</summary>
        public override int Outputs
        {
            get { return this.output.Length; }
        }

        /// <summary>Feeds the given input trough this block, updating its state.</summary>
        /// <param name="input">The input to be fed.</param>
        /// <param name="output">The array to be written the output into.</param>
        public override void Feed(double[] input, double[] output)
        {
            this.errors.Add(true);
            this.contextMatrix.Feed();
            this.input.Feed(input);
            double[] unfolded = this.unfolded.Add();
            this.input.GetLastInput(unfolded);
            this.w.Feed();
            double[] unfoldedGates = this.unfoldedGates.Add();
            this.gates.GetLastInput(unfoldedGates);
            double[] state = this.unfoldedStates.Add();
            for (int i = 0; i < this.state.Length; i++)
            {
                state[i] = this.inputActivation.GetLastOutput(i) * this.inputGate.GetLastOutput(i) + this.forgetGate.GetLastOutput(i) * this.state.GetLastOutput(i);
                output[i] = Math.Tanh(state[i]) * this.outputGate.GetLastOutput(i);
            }
            this.unfoldedOutptuts.Add(output);
            this.state.Feed(state);
            this.output.Feed(output);
        }

        /// <summary>Sets an error for this block.</summary>
        /// <param name="error">The error array to be set. It must refer to the latest feeding process.</param>
        public override void BackPropagate(double[] error)
        {
            for (int i = 0; i < this.Outputs; i++)
            {
                this.errors[this.errors.Length - 1][i] += error[i];
            }
        }

        /// <summary>Applies the weight changes to this block and prepares it for a new sequence.</summary>
        /// <param name="rate">The learning rate at which the parameters are to be updated.</param>
        public override void Reset(double rate)
        {
            Array.Clear(this.error1, 0, this.error1.Length);
            double[] stateDeltas = new double[this.Outputs];
            double[] forgetCpy = new double[this.Outputs];
            for (int i = 0; i < this.unfolded.Length; i++)
            {
                int step = this.unfolded.Length - i - 1;
                this.input.Feed(this.unfolded[step]);
                if (this.peephole)
                {
                    this.context.Feed(this.unfoldedStates[step]);
                }
                else
                {
                    this.context.Feed(this.unfoldedOutptuts[step]);
                }
                this.gates.Feed(this.unfoldedGates[step]);

                this.output.Feed(this.unfoldedOutptuts[step + 1]);


                double[] unfoldedState = this.unfoldedStates[step + 1];
                for (int j = 0; j < this.Outputs; j++)
                {
                    double tanh = Math.Tanh(unfoldedState[j]);
                    double error = this.error1[j] + this.errors[step][j];
                    stateDeltas[j] = error * (1 - tanh * tanh) * this.outputGate.GetLastOutput(j) + stateDeltas[j] * forgetCpy[j];
                    this.error2[j] = stateDeltas[j] * this.inputGate.GetLastOutput(j);
                    this.error2[j + this.Outputs] = stateDeltas[j] * this.inputActivation.GetLastOutput(j);
                    this.error2[j + this.Outputs * 2] = step > 0 ? stateDeltas[j] * this.unfoldedStates[step][j] : 0.0;
                    this.error2[j + this.Outputs * 3] = error * tanh;
                }
                this.w.BackPropagateToDeltas(this.error2, this.error1);
                this.forgetGate.GetLastOutput(forgetCpy);
            }
            this.w.ApplyDeltas(rate);

            this.state.Zero();
            this.context.Zero();
            this.unfolded.Clear();
            this.unfoldedOutptuts.Clear();
            this.unfoldedOutptuts.Add(true);
            this.unfoldedGates.Clear();
            this.unfoldedStates.Clear();
            this.unfoldedStates.Add(true);
            this.errors.Clear();
        }
        
        /// <summary>Copies this instance of the <code>LSTMBlock</code> class into another.</summary>
        /// <param name="block">The instance to be copied into.</param>
        protected void CloneTo(LSTMBlock block)
        {
            block.input = (LinearNeuronsString)this.input.Clone();
            block.output = (LinearNeuronsString)this.output.Clone();
            block.peephole = this.peephole;
            block.inputActivation = (TanhNeuronsString)this.inputActivation.Clone();
            block.inputGate = (NeuronsString)this.inputGate.Clone();
            block.forgetGate = (NeuronsString)this.forgetGate.Clone();
            block.outputGate = (NeuronsString)this.outputGate.Clone();
            block.gates = new CompoundLayer(block.inputActivation, block.inputGate, block.forgetGate, block.outputGate);
            block.state = (LinearNeuronsString)this.state.Clone();
            block.output = (LinearNeuronsString)this.output.Clone();
            block.context = (LinearNeuronsString)this.context.Clone();
            block.firstLayer = new CompoundLayer(block.context, block.input);
            block.w = new BiasedConnectionMatrix(block.firstLayer, block.gates);
            if (this.peephole)
            {
                block.contextMatrix = new FlattenConnectionMatrix(block.state, block.context);
            }
            else
            {
                block.contextMatrix = new FlattenConnectionMatrix(block.output, block.context);
            }
            block.error1 = new double[this.gates.Length];
            block.error2 = new double[this.gates.Length];
            block.unfolded = new ArrayInfiniteTimeMemory(this.Inputs);
            block.unfoldedOutptuts = new ArrayInfiniteTimeMemory(this.Outputs);
            block.unfoldedOutptuts.Add(new double[this.Outputs]);
            block.unfoldedGates = new ArrayInfiniteTimeMemory(this.gates.Length);
            block.unfoldedStates = new ArrayInfiniteTimeMemory(this.Outputs);
            block.unfoldedStates.Add(new double[this.Outputs]);
            block.errors = new ArrayInfiniteTimeMemory(this.Outputs);
        }

        /// <summary>Creates a copy of this instance of the <code>LSTMBlock</code> class.</summary>
        /// <returns>The generated instance of the <code>LSTMBlock</code> class.</returns>
        public override object Clone()
        {
            LSTMBlock retVal = new LSTMBlock();
            this.CloneTo(retVal);
            return retVal;
        }

        /// <summary>Feeds the given input trough the block, updating its state.</summary>
        /// <param name="input">The array to be fed.</param>
        public override void Feed(double[] input)
        {
            this.errors.Add(true);
            this.contextMatrix.Feed();
            this.input.Feed(input);
            double[] unfolded = this.unfolded.Add(true);
            this.firstLayer.GetLastInput(unfolded);
            this.w.Feed();
            double[] output = new double[this.Outputs];
            double[] state = this.unfoldedStates.Add(true);
            for (int i = 0; i < this.state.Length; i++)
            {
                state[i] = this.inputActivation.GetLastOutput(i) * this.inputGate.GetLastOutput(i) + this.forgetGate.GetLastOutput(i) * this.state.GetLastOutput(i);
                output[i] = Math.Tanh(state[i]) * this.outputGate.GetLastOutput(i);
            }
            this.state.Feed(state);
            this.output.Feed(output);
        }

        /// <summary>Exports this instance of the <code>LSTMBlock</code> class to a stream.</summary>
        /// <param name="stream">The stream to be exported into.</param>
        public override void Save(Stream stream)
        {
            new DataContractJsonSerializer(typeof(LSTMBlock)).WriteObject(stream, this);
        }

        /// <summary>Imports a saved instance of the <code>LSTMBlock</code> class from a stream.</summary>
        /// <param name="stream">The stream to be imported from.</param>
        /// <returns>The imported instance.</returns>
        public static LSTMBlock Load(Stream stream)
        {
            return (LSTMBlock)(new DataContractJsonSerializer(typeof(LSTMBlock)).ReadObject(stream));
        }

        /// <summary>Imports an instance of the <code>LSTMBlock</code> class from a file.</summary>
        /// <param name="fileName">The name of the file to be imported.</param>
        /// <returns>The imported instance.</returns>
        public static LSTMBlock Load(string fileName)
        {
            FileStream fs = new FileStream(fileName, FileMode.Open);
            LSTMBlock retVal = LSTMBlock.Load(fs);
            fs.Close();
            return retVal;
        }

        [OnDeserialized]
        private void SetValuesOnDeserialized(StreamingContext context)
        {
            this.gates = new CompoundLayer(this.inputActivation, this.inputGate, this.forgetGate, this.outputGate);
            this.firstLayer = new CompoundLayer(this.context, this.input);
            this.error1 = new double[this.gates.Length];
            this.error2 = new double[this.gates.Length];
            this.unfolded = new ArrayInfiniteTimeMemory(this.Inputs);
            this.unfoldedOutptuts = new ArrayInfiniteTimeMemory(this.Outputs);
            this.unfoldedOutptuts.Add(new double[this.Outputs]);
            this.unfoldedGates = new ArrayInfiniteTimeMemory(this.gates.Length);
            this.unfoldedStates = new ArrayInfiniteTimeMemory(this.Outputs);
            this.unfoldedStates.Add(new double[this.Outputs]);
            this.errors = new ArrayInfiniteTimeMemory(this.Outputs);
            this.w.SetLayers(this.firstLayer, this.gates);
            if (peephole)
            {
                this.contextMatrix.SetLayers(this.state, this.context);
            }
            else
            {
                this.contextMatrix.SetLayers(this.output, this.context);
            }
        }
    }
}
