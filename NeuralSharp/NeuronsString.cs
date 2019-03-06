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
    [DataContract]
    public class NeuronsString : IArraysLayer
    {
        private double[] input;
        private double[] output;
        private int inputSkip;
        private int outputSkip;
        [DataMember]
        private int length;

        protected NeuronsString(NeuronsString original, bool siamese)
        {
            this.length = original.Length;
        }

        public NeuronsString(int length, bool createIO = false)
        {
            if (createIO)
            {
                this.input = Backbone.CreateArray<double>(length);
                this.output = Backbone.CreateArray<double>(length);
                this.inputSkip = 0;
                this.outputSkip = 0;
            }
            this.length = length;
        }
        
        public double[] Input
        {
            get { return this.input; }
        }

        public double[] Output
        {
            get { return this.output; }
        }

        public int InputSkip
        {
            get { return this.inputSkip; }
        }

        public int OutputSkip
        {
            get { return this.outputSkip; }
        }

        public int InputSize
        {
            get { return this.length; }
        }

        public int OutputSize
        {
            get { return this.length; }
        }

        public int Length
        {
            get { return this.length; }
        }

        public int Parameters
        {
            get { return 0; }
        }

        protected virtual double Activation(double input)
        {
            return 1.0 / (1.0 + Math.Exp(-input));
        }

        protected virtual double ActivationDerivative(double input, double output)
        {
            return output * (1.0 - output);
        }

        public virtual void Feed(bool learning = false)
        {
            Backbone.ApplyNeuronsString(this.input, this.inputSkip, this.output, this.outputSkip, this.length, this.Activation);
        }

        public virtual void BackPropagate(double[] outputErrorArray, int outputErrorSkip, double[] inputErrorArray, int inputErrorSkip, bool learning)
        {
            Backbone.BackpropagateNeuronsString(this.input, this.inputSkip, this.output, this.outputSkip, this.length, outputErrorArray, outputErrorSkip, inputErrorArray, inputErrorSkip, this.ActivationDerivative, learning);
        }

        public void BackPropagate(double[] outputError, double[] inputError, bool learning)
        {
            this.BackPropagate(outputError, 0, inputError, 0, learning);
        }

        public void SetInputAndOutput(double[] inputArray, int inputSkip, double[] outputArray, int outputSkip)
        {
            this.input = inputArray;
            this.inputSkip = inputSkip;
            this.output = outputArray;
            this.outputSkip = outputSkip;
        }

        public void SetInputAndOutput(double[] input, double[] output)
        {
            this.SetInputAndOutput(input, 0, output, 0);
        }

        public double[] SetInputGetOutput(double[] inputArray, int inputSkip)
        {
            this.input = inputArray;
            this.inputSkip = inputSkip;
            this.outputSkip = 0;
            return this.output = Backbone.CreateArray<double>(this.Length);
        }

        public double[] SetInputGetOutput(double[] input)
        {
            return this.SetInputGetOutput(input, 0);
        }

        public void UpdateWeights(double rate, double momentum = 0.0) { }
        
        public virtual IUntypedLayer CreateSiamese()
        {
            return new NeuronsString(this, true);
        }

        public virtual IUntypedLayer Clone()
        {
            return new NeuronsString(this, false);
        }
    }
}
