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
using NeuralNetwork.Recurrent.LSTM;

namespace NeuralNetwork.Recurrent
{
    /// <summary>Represents an entity capable of learning sequences of chars, and therefore generating text.</summary>
    [DataContract]
    public class TextGenerator : ICloneable
    {
        [DataMember]
        private LSTMNetwork rnn;
        [DataMember]
        private char[] acceptedChars;

        /// <summary>Empty constructor. It does not initialize fields.</summary>
        protected TextGenerator() { }

        /// <summary>Creates a new instance of the <code>TextGenerator</code> class.</summary>
        /// <param name="acceptedChars">Accepted characters.</param>
        /// <param name="complexity">Level of complexity of the inner neural network.</param>
        public TextGenerator(char[] acceptedChars, int complexity)
        {
            this.acceptedChars = (char[])acceptedChars.Clone();
            this.rnn = new LSTMNetwork(this.acceptedChars.Length + 2, complexity, this.acceptedChars.Length + 1, true);
        }
        
        private double[][] StringToLabels(string text, double[][] labels)
        {
            double[][] retVal = new double[text.Length + 2][];
            retVal[0] = labels[0];
            for (int i = 0; i < text.Length; i++)
            {
                retVal[i + 1] = labels[Array.IndexOf(this.acceptedChars, text[i]) + 1];
            }
            retVal[text.Length + 1] = labels[acceptedChars.Length + 1];
            return retVal;
        }

        private double[][] CreateLabels()
        {
            double[][] retVal = new double[this.acceptedChars.Length + 2][];
            for (int i = 0; i < retVal.Length; i++)
            {
                retVal[i] = new double[retVal.Length];
                retVal[i][i] = 1.0;
            }
            return retVal;
        }
        
        private double[][] StringToLabels(string text)
        {
            return this.StringToLabels(text, this.CreateLabels());
        }

        private double[][][] StringsToLabels(string[] lines)
        {
            double[][][] retVal = new double[lines.Length][][];
            double[][] labels = this.CreateLabels();
            for (int i = 0; i < lines.Length; i++)
            {
                retVal[i] = this.StringToLabels(lines[i], labels);
            }
            return retVal;
        }

        /// <summary>Learns the given strings.</summary>
        /// <param name="lines">Strings to be learned.</param>
        public void Learn(string[] lines)
        {
            double[][][] sequences = this.StringsToLabels(lines);
            this.rnn.Learn(sequences, 0.01, 20000);
        }

        private int OutputToIndex(double[] label, bool random = true, double[] normalized = null)
        {
            int retVal;
            double r;
            if (random)
            {
                retVal = -1;
                r = RandomGenerator.GetDouble();
                double c = 0.0;
                while (c < r)
                {
                    retVal++;
                    c += label[retVal];
                }
            }
            else
            {
                retVal = 0;
                for (int i = 1; i < label.Length; i++)
                {
                    if (label[i] > label[retVal])
                    {
                        retVal = i;
                    }
                }
            }
            if (normalized != null)
            {
                Array.Clear(normalized, 0, normalized.Length);
                normalized[retVal] = 1.0;
            }
            return retVal;
        }

        /// <summary>Continues the given string.</summary>
        /// <param name="str">String to be continued.</param>
        /// <param name="maxLength">Maximum length of the output string. If <code>0</code> it is unlimited.</param>
        /// <param name="random">If <code>true</code>, the string is generated randomly with the probabilities given by the network. Otherwise, it is generated by always picking the character with the highest probability.</param>
        /// <returns>The generated string.</returns>
        public string ContinueString(string str, int maxLength = 0, bool random = true)
        {
            string retVal = str;
            double[] input = new double[this.acceptedChars.Length + 2];
            double[] output = new double[this.acceptedChars.Length + 2];
            double[][] start = this.StringToLabels(str);
            this.rnn.Reset(0.0);
            for (int i = 0; i < start.Length - 2; i++)
            {
                this.rnn.Feed(start[i]);
            }
            Array.Copy(start[start.Length - 2], input, input.Length);
            bool ended = false;
            while (!ended && (maxLength <= 0 || retVal.Length < maxLength))
            {
                this.rnn.Feed(input, output);
                int label = this.OutputToIndex(output, random, input);
                if (label == 0 || label == input.Length - 1)
                {
                    ended = true;
                }
                else
                {
                    retVal += this.acceptedChars[label - 1];
                }
            }
            return retVal;
        }

        /// <summary>Exports this instance of the <code>TextGenerator</code> class to a stream.</summary>
        /// <param name="stream">The stream to be exported into.</param>
        public void Save(Stream stream)
        {
            new DataContractJsonSerializer(typeof(TextGenerator)).WriteObject(stream, this);
        }

        /// <summary>Exports this instance of the <code>TextGenerator</code> class to a file.</summary>
        /// <param name="fileName">The name of the file to be created.</param>
        public void Save(string fileName)
        {
            FileStream fs = new FileStream(fileName, FileMode.Create);
            this.Save(fs);
            fs.Close();
        }

        /// <summary>Imports an instance of the <code>TextGenerator</code> class from a stream.</summary>
        /// <param name="stream">The stream to be imported from.</param>
        /// <returns>The imported instance.</returns>
        public static TextGenerator Load(Stream stream)
        {
            return (TextGenerator)(new DataContractJsonSerializer(typeof(TextGenerator)).ReadObject(stream));
        }

        /// <summary>Imports a text generator from a file</summary>
        /// <param name="fileName">The name of the file to be imported.</param>
        /// <returns>The imported instance.</returns>
        public static TextGenerator Load(string fileName)
        {
            FileStream fs = new FileStream(fileName, FileMode.Open);
            TextGenerator retVal = TextGenerator.Load(fs);
            fs.Close();
            return retVal;
        }

        /// <summary>Copies this instance of the <code>TextGenerator</code> class into another.</summary>
        /// <param name="txtGenerator">The instance to be copied into.</param>
        protected void CloneTo(TextGenerator txtGenerator)
        {
            txtGenerator.rnn = (LSTMNetwork)this.rnn.Clone();
            txtGenerator.acceptedChars = (char[])this.acceptedChars.Clone();
        }

        /// <summary>Creates a copy of this instance of the <code>TextGenerator</code> class.</summary>
        /// <returns>The generated instance of the <code>TextGenerator</code> class.</returns>
        public virtual object Clone()
        {
            TextGenerator retVal = new TextGenerator();
            this.CloneTo(retVal);
            return retVal;
        }
    }
}
