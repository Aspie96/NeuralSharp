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
using System.Globalization;
using System.IO;
using System.Linq;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Json;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    /// <summary>Represents a feed forward neural network.</summary>
    [DataContract]
    [KnownType(typeof(BiasedConnectionMatrix))]
    [KnownType(typeof(LinearNeuronsString))]
    [KnownType(typeof(LeakyReluNeuronsString))]
    [KnownType(typeof(SoftmaxNeuronsString))]
    public class FeedForwardNN : ForwardLearner<double[]>, IFeedForwardNN
    {
        private const double leakyAlpha = 0.1;

        [DataMember]
        private int inputs;
        [DataMember]
        private int outputs;
        [DataMember]
        private NeuronsString[] layers;
        [DataMember]
        private ConnectionMatrix[] connections;
        private double[] error1;
        private double[] error2;

        /// <summary>Empty constructor. It does not initialize the fields.</summary>
        protected FeedForwardNN() { }

        /// <summary>Creates a new instance of the <code>FeedForwardNN</code> class.</summary>
        /// <param name="inputs">The amount of inputs of the network.</param>
        /// <param name="outputs">The amount of outputs of the network.</param>
        /// <param name="depth">The amount of connection matrices in the network.</param>
        /// <param name="classification">Indicates wether the network is to be used for classifcation purposes.</param>
        /// <param name="layerLengths">The lenghts of all inner layers in the network.</param>
        public FeedForwardNN(int inputs, int outputs, int depth, bool classification, params int[] layerLengths)
        {
            this.inputs = inputs;
            this.outputs = outputs;
            this.layers = new NeuronsString[depth + 1];
            this.connections = new BiasedConnectionMatrix[depth];
            int maxLength = Math.Max(inputs, outputs);
            this.layers[0] = new LinearNeuronsString(inputs);
            for (int i = 0; i < depth - 1; i++)
            {
                this.layers[i + 1] = new LeakyReluNeuronsString(layerLengths[i], FeedForwardNN.leakyAlpha);
                this.connections[i] = new BiasedConnectionMatrix(this.layers[i], this.layers[i + 1]);
                maxLength = Math.Max(maxLength, layerLengths[i]);
            }
            if (classification)
            {
                this.layers[depth] = new SoftmaxNeuronsString(outputs);
            }
            else
            {
                this.layers[depth] = new NeuronsString(outputs);
            }
            this.connections[depth - 1] = new BiasedConnectionMatrix(this.layers[depth - 1], this.layers[depth]);
            this.error1 = new double[maxLength];
            this.error2 = new double[maxLength];
        }

        /// <summary>Creates a new instance of the <code>FeedForwardNN</code> class.</summary>
        /// <param name="inputs">The amount of inputs of the network.</param>
        /// <param name="outputs">Amount of outputs of the network.</param>
        /// <param name="classification">Indicates if the network is to be used for classification purposes.</param>
        public FeedForwardNN(int inputs, int outputs, bool classification = false) : this(inputs, outputs, 1, classification, null) { }

        /// <summary>Imports classes data from the CSV format.</summary>
        /// <param name="stream">The stream to be imported from.</param>
        /// <param name="separators">The used separators.</param>
        /// <param name="classes">The amount of classes.</param>
        /// <param name="cellsSkip">The amount of columns to be skipped.</param>
        /// <param name="maxCount">The maximum amount of classes to be imported. If <code>0</code>, all available classes are imported.</param>
        /// <returns>The imported labels.</returns>
        public static double[][] FromCsvClasses(Stream stream, char[] separators, int classes, int cellsSkip = 0, int maxCount = 0)
        {
            List<double[]> retVal = new List<double[]>();
            StreamReader sr = new StreamReader(stream, Encoding.UTF8, true, 1024, true);
            string line;
            while ((line = sr.ReadLine()) != null && (maxCount == 0 || retVal.Count < maxCount))
            {
                string[] parts = line.Split(separators);
                double[] row = new double[classes];
                row[int.Parse(parts[cellsSkip])] = 1.0;
                retVal.Add(row);
            }
            sr.Close();
            return retVal.ToArray();
        }

        /// <summary>Imports labels data from a CSV file.</summary>
        /// <param name="fileName">The name of the file to be imported.</param>
        /// <param name="classes">The amount of classes.</param>
        /// <param name="cellsSkip">The amount of columns to be skipped.</param>
        /// <param name="maxCount">The maximum amount of labels to be imported. If <code>0</code>, all available labels are imported.</param>
        /// <returns>The imported labels.</returns>
        public static double[][] FromCsvClasses(string fileName, int classes, int cellsSkip = 0, int maxCount = 0)
        {
            Stream stream = new FileStream(fileName, FileMode.Open);
            double[][] retVal = FeedForwardNN.FromCsvClasses(stream, new char[] { '\t', ',', ' ', ';' }, classes, cellsSkip, maxCount);
            stream.Close();
            return retVal;
        }

        /// <summary>Imports data from the CSV format.</summary>
        /// <param name="stream">The stream to be imported from.</param>
        /// <param name="separators">The used separators.</param>
        /// <param name="rowLengths">The row lenghts for the column groups.</param>
        /// <param name="cellsSkip">The amount of columns to be skipped.</param>
        /// <param name="maxCount">The maximum amount of rows to be imported. If <code>0</code>, all available rows are imported.</param>
        /// <returns>The imported data.</returns>
        public static double[][][] FromCsv(Stream stream, char[] separators, int[] rowLengths, int cellsSkip = 0, int maxCount = 0)
        {
            double[][][] retVal = new double[rowLengths.Length][][];
            List<double[]>[] lists = new List<double[]>[retVal.Length];
            for (int i = 0; i < lists.Length; i++)
            {
                lists[i] = new List<double[]>();
            }
            StreamReader sr = new StreamReader(stream, Encoding.UTF8, true, 1024, true);
            string line;
            while ((line = sr.ReadLine()) != null && (maxCount == 0 || lists[0].Count < maxCount))
            {
                string[] parts = line.Split(separators);
                int index = cellsSkip;
                for (int j = 0; j < retVal.Length; j++)
                {
                    double[] row = new double[rowLengths[j]];
                    for (int k = 0; k < rowLengths[j]; k++)
                    {
                        row[k] = double.Parse(parts[index], CultureInfo.InvariantCulture);
                        index++;
                    }
                    lists[j].Add(row);
                }
            }
            sr.Close();
            for (int i = 0; i < lists.Length; i++)
            {
                retVal[i] = lists[i].ToArray();
            }
            return retVal;
        }

        /// <summary>Imports data from a CSV file.</summary>
        /// <param name="fileName">The name of the file to be imported.</param>
        /// <param name="rowLengths">The lenghts of the column groups.</param>
        /// <param name="cellsSkip">The amount of columns to be skipped.</param>
        /// <param name="maxCount">The maximum amount of rows to be imported. If <code>0</code>, all available rows are imported.</param>
        /// <returns>The read data.</returns>
        public static double[][][] FromCsv(string fileName, int[] rowLengths, int cellsSkip = 0, int maxCount = 0)
        {
            Stream stream = new FileStream(fileName, FileMode.Open);
            double[][][]  retVal = FeedForwardNN.FromCsv(stream, new char[] { '\t', ',', ' ', ';' }, rowLengths, cellsSkip, maxCount);
            stream.Close();
            return retVal;
        }

        /// <summary>Imports data from the CSV format.</summary>
        /// <param name="stream">The stream to be imported from.</param>
        /// <param name="separators">The used separators.</param>
        /// <param name="rowLength">The lenghts of the column groups.</param>
        /// <param name="cellsSkip">The amount of columns to be skipped.</param>
        /// <param name="maxCount">The maximum amount of rows to be imported. If <code>0</code>, all rows available are imported.</param>
        /// <returns>The read data.</returns>
        public static double[][] FromCsv(Stream stream, char[] separators, int rowLength, int cellsSkip = 0, int maxCount = 0)
        {
            List<double[]> retVal = new List<double[]>();
            StreamReader sr = new StreamReader(stream, Encoding.UTF8, true, 1024, true);
            string line;
            while ((line = sr.ReadLine()) != null && (maxCount == 0 || retVal.Count < maxCount))
            {
                string[] parts = line.Split(separators);
                double[] row = new double[rowLength];
                for (int j = 0; j < rowLength; j++)
                {
                    row[j] = double.Parse(parts[cellsSkip + j], CultureInfo.InvariantCulture);
                }
                retVal.Add(row);
            }
            sr.Close();
            return retVal.ToArray();
        }

        /// <summary>Imports data from the CSV format.</summary>
        /// <param name="stream">The stream to be imported from.</param>
        /// <param name="rowLength">The length of each row.</param>
        /// <param name="cellsSkip">The amount of columns to be skipped.</param>
        /// <param name="maxCount">The maximum amount of rows to be imported. If <code>0</code>, all available rows are imported.</param>
        /// <returns>The imported data.</returns>
        public static double[][] FromCsv(Stream stream, int rowLength, int cellsSkip = 0, int maxCount = 0)
        {
            return FeedForwardNN.FromCsv(stream, new char[] { '\t', ',', ' ', ';' }, rowLength, cellsSkip, maxCount);
        }

        /// <summary>Imports data from a CSV file.</summary>
        /// <param name="fileName">The name of the file to be imported.</param>
        /// <param name="separators">The used eparators.</param>
        /// <param name="rowLength">The lenght of each raw.</param>
        /// <param name="cellsSkip">The amount of columns to be skipped.</param>
        /// <param name="maxCount">The maximum amount of rows to be imported. If <code>0</code>, all available rows are imported.</param>
        /// <returns>The imported data.</returns>
        public static double[][] FromCsv(string fileName, char[] separators, int rowLength, int cellsSkip = 0, int maxCount = 0)
        {
            Stream stream = new FileStream(fileName, FileMode.Open);
            double[][] retVal = FeedForwardNN.FromCsv(stream, separators, rowLength, cellsSkip, maxCount);
            stream.Close();
            return retVal;
        }

        /// <summary>Imports data from a CSV file.</summary>
        /// <param name="fileName">The name of the file to be imported.</param>
        /// <param name="rowLength">The used separators.</param>
        /// <param name="cellsSkip">The amount of columns to be skipped.</param>
        /// <param name="maxCount">The maximum number of rows to be imported. If <code>0</code>, all rows are imported.</param>
        /// <returns>The read data.</returns>
        public static double[][] FromCsv(string fileName, int rowLength, int cellsSkip = 0, int maxCount = 0)
        {
            Stream stream = new FileStream(fileName, FileMode.Open);
            double[][] retVal = FeedForwardNN.FromCsv(stream, rowLength, cellsSkip, maxCount);
            stream.Close();
            return retVal;
        }

        /// <summary>Exports data into the CSV format.</summary>
        /// <param name="stream">The stream to be written into.</param>
        /// <param name="separator">The separator to be used.</param>
        /// <param name="data">The data to be exported.</param>
        public static void ToCsv(Stream stream, char separator, params double[][][] data)
        {
            StreamWriter sw = new StreamWriter(stream, Encoding.UTF8, 1024, true);
            for (int i = 0; i < data[0].Length; i++)
            {
                for (int j = 0; j < data.Length; j++)
                {
                    for (int k = 0; k < data[j][i].Length; k++)
                    {
                        sw.Write(data[j][i][k].ToString(CultureInfo.InvariantCulture));
                        if (j != data.Length - 1 || k != data[j][i].Length - 1)
                        {
                            sw.Write(separator);
                        }
                    }
                }
                sw.WriteLine();
            }
            sw.Close();
        }

        /// <summary>Writes data into a CSV file.</summary>
        /// <param name="fileName">The name of the file to be created.</param>
        /// <param name="separator">The separator to be used.</param>
        /// <param name="data">The data to be exported.</param>
        public static void ToCsv(string fileName, char separator, params double[][][] data)
        {
            Stream stream = new FileStream(fileName, FileMode.Create);
            FeedForwardNN.ToCsv(stream, separator, data);
            stream.Close();
        }

        /// <summary>Columnwise normalizes data between the given values.</summary>
        /// <param name="data">The data to be normalized.</param>
        /// <param name="minimum">The minimum value to be normalized at.</param>
        /// <param name="maximum">The maximum value to be normalized at.</param>
        public static void NormalizeData(double[][] data, double minimum = 0.0, double maximum = 1.0)
        {
            double range = maximum - minimum;
            for (int i = 0; i < data[0].Length; i++)
            {
                double colMin = double.PositiveInfinity;
                double colMax = double.NegativeInfinity;
                for (int j = 0; j < data.Length; j++)
                {
                    colMin = Math.Min(colMin, data[j][i]);
                    colMax = Math.Max(colMax, data[j][i]);
                }
                if (colMin == colMax)
                {
                    for (int j = 0; j < data.Length; j++)
                    {
                        data[j][i] = minimum;
                    }
                }
                else
                {
                    double colRange = colMax - colMin;
                    for (int j = 0; j < data.Length; j++)
                    {
                        data[j][i] = (data[j][i] - colMin) / colRange * range + minimum;
                    }
                }
            }
        }

        /// <summary>Subtract the mean from the given data.</summary>
        /// <param name="data">The data to be subtracted the mean from.</param>
        public static void SubtractMean(double[][] data)
        {
            for (int i = 0; i < data[0].Length; i++)
            {
                double mean = 0;
                for (int j = 0; j < data.Length; j++)
                {
                    mean += data[j][i];
                }
                mean /= data.Length;
                if (mean != 0)
                {
                    for (int j = 0; j < data.Length; j++)
                    {
                        data[j][i] -= mean;
                    }
                }
            }
        }

        /// <summary>The number of connection matrices in this network.</summary>
        public int Depth
        {
            get { return this.connections.Length; }
        }

        /// <summary>The amount of inputs of this network.</summary>
        public int Inputs
        {
            get { return this.inputs; }
        }

        /// <summary>The amount of outputs of this network.</summary>
        public override int Outputs
        {
            get { return this.outputs; }
        }

        /// <summary>The amount of weights of this network.</summary>
        public int Params
        {
            get
            {
                int retVal = 0;
                for (int i = 0; i < this.Depth; i++)
                {
                    retVal += this.connections[i].Params;
                }
                return retVal;
            }
        }

        /// <summary>Feeds the given input array trough the netwrok.</summary>
        /// <param name="input">The input array to be fed.</param>
        /// <param name="output">The array to be written the output into.</param>
        public override void Feed(double[] input, double[] output)
        {
            this.Feed(input);
            this.layers[this.Depth].GetLastOutput(output);
        }

        /// <summary>Feeds the given inputs trough the network.</summary>
        /// <param name="input">Inputs to be fed.</param>
        public void Feed(params double[] input)
        {
            this.layers[0].Feed(input);
            for (int i = 0; i < this.Depth; i++)
            {
                this.connections[i].Feed();
            }
        }

        /// <summary>Backpropagates the given error trough the network, updating its weights.</summary>
        /// <param name="error">The error to be backpropagated. It must refer to the latest feeding process.</param>
        /// <param name="rate">The learning rate at which the weights are to be updated.</param>
        public override void BackPropagate(double[] error, double rate)
        {
            Array.Copy(error, this.error2, this.Outputs);
            for (int i = this.Depth - 1; i >= 0; i -= 1)
            {
                this.connections[i].BackPropagate(this.error2, this.error1, rate);
                double[] aux = this.error1;
                this.error1 = this.error2;
                this.error2 = aux;
            }
        }

        /// <summary>Backpropagates the given error trough the network.</summary>
        /// <param name="error">The error to be backpropagated. It must refer to the latest feeding process.</param>
        /// <param name="inputError">The array to be written the input error into.</param>
        /// <param name="rate">The learning rate at which the weights are to be updated.</param>
        public override void BackPropagate(double[] error, double[] inputError, double rate)
        {
            this.BackPropagate(error, rate);
            Array.Copy(this.error2, inputError, this.Inputs);
        }

        /// <summary>Backpropagates the given error trough the network and sums the weight gradients to the stored values.</summary>
        /// <param name="error">The error to be backpropagated. It must refer to the latest feeding process.</param>
        public void BackPropagateToDeltas(double[] error)
        {
            Array.Copy(error, this.error2, this.Outputs);
            for (int i = this.Depth - 1; i >= 0; i -= 1)
            {
                this.connections[i].BackPropagateToDeltas(this.error2, this.error1);
                double[] aux = this.error1;
                this.error1 = this.error2;
                this.error2 = aux;
            }
        }

        /// <summary>Backpropagates an error trough the network and sums the weight gradients to the stored values.</summary>
        /// <param name="error">The error to be backpropagated. It must refer to the latest feeding process.</param>
        /// <param name="inputError">The array to be written the input error into.</param>
        public void BackPropagateToDeltas(double[] error, double[] inputError)
        {
            this.BackPropagateToDeltas(error);
            Array.Copy(this.error2, inputError, this.Inputs);
        }

        /// <summary>Updates the weights using the stored gradient values and sets the gradients to <code>0</code>.</summary>
        /// <param name="rate">The lerning rate at the weights are to be updated.</param>
        public void ApplyDeltas(double rate)
        {
            for (int i = this.Depth - 1; i >= 0; i -= 1)
            {
                this.connections[i].ApplyDeltas(rate);
            }
        }

        [OnDeserialized]
        private void SetValuesOnDeserialized(StreamingContext context)
        {
            int maxLength = Math.Max(inputs, outputs);
            this.layers[0] = new LinearNeuronsString(inputs);
            for (int i = 0; i < this.connections.Length - 1; i++)
            {
                this.connections[i].SetLayers(this.layers[i], this.layers[i + 1]);
                maxLength = Math.Max(maxLength, this.layers[i + 1].Length);
            }
            this.connections[this.connections.Length - 1].SetLayers(this.layers[this.connections.Length - 1], this.layers[this.connections.Length]);
            this.error1 = new double[maxLength];
            this.error2 = new double[maxLength];
        }

        /// <summary>Exports this instance of the <code>FeedForwardNN</code> class to a stream.</summary>
        /// <param name="stream">The stream to be exported into.</param>
        public override void Save(Stream stream)
        {
            new DataContractJsonSerializer(typeof(FeedForwardNN)).WriteObject(stream, this);
        }

        /// <summary>Imports an instance of the <code>FeedForwardNN</code> class from a stream.</summary>
        /// <param name="stream">The stream to be imported from.</param>
        /// <returns>The imported instance.</returns>
        public static FeedForwardNN Load(Stream stream)
        {
            return (FeedForwardNN)(new DataContractJsonSerializer(typeof(FeedForwardNN)).ReadObject(stream));
        }

        /// <summary>Imports an instance of the <code>FeedForwardNN</code> class from a file.</summary>
        /// <param name="fileName">The name of the file to be imported.</param>
        /// <returns>The imported instance.</returns>
        public static FeedForwardNN Load(string fileName)
        {
            FileStream fs = new FileStream(fileName, FileMode.Open);
            FeedForwardNN retVal = FeedForwardNN.Load(fs);
            fs.Close();
            return retVal;
        }

        /// <summary>Copies this instance of the <code>FeedForwardNN</code> class into another.</summary>
        /// <param name="fnn">The instance to be copied into.</param>
        protected void CloneTo(FeedForwardNN fnn)
        {
            fnn.inputs = this.Inputs;
            fnn.outputs = this.Outputs;
            fnn.layers = new NeuronsString[this.Depth + 1];
            fnn.connections = new ConnectionMatrix[this.Depth];
            fnn.error1 = new double[this.error1.Length];
            fnn.error2 = new double[this.error2.Length];
            fnn.layers[0] = (NeuronsString)this.layers[0].Clone();
            for (int i = 0; i < this.Depth; i++)
            {
                fnn.layers[i + 1] = (NeuronsString)this.layers[i + 1].Clone();
                fnn.connections[i] = (ConnectionMatrix)this.connections[i].Clone(fnn.layers[i], fnn.layers[i + 1]);
            }
        }

        /// <summary>Creates a copy of this instance of the <code>FeedForwardNN</code> class.</summary>
        /// <returns>The generated instance of the <code>FeedForwardNN</code> class.</returns>
        public override object Clone()
        {
            FeedForwardNN retVal = new FeedForwardNN();
            this.CloneTo(retVal);
            return retVal;
        }
    }
}
