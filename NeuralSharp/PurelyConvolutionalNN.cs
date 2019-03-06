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
using System.Runtime.Serialization;
using System.Text;
using System.Threading.Tasks;

namespace NeuralSharp
{
    [DataContract]
    public class PurelyConvolutionalNN : Sequential<Image, Image>, IImagesLayer
    {
        [DataMember]
        private Image error1;
        private Image error2;
        private bool layersConnected;

        public PurelyConvolutionalNN(PurelyConvolutionalNN original, bool siamese) : base(original, siamese)
        {
            int maxDepth = 0;
            int maxWidth = 0;
            int maxHeight = 0;
            foreach (IImagesLayer layer in original.Layers)
            {
                maxDepth = Math.Max(maxDepth, layer.OutputDepth);
                maxWidth = Math.Max(maxWidth, layer.OutputWidth);
                maxHeight = Math.Max(maxHeight, layer.OutputHeight);
            }
            this.error1 = new Image(maxDepth, maxWidth, maxHeight);
            this.error2 = new Image(maxDepth, maxWidth, maxHeight);
            this.layersConnected = false;
        }

        public PurelyConvolutionalNN(ICollection<IImagesLayer> layers, bool createIO = true) : base(layers.ToArray<IUntypedLayer>())
        {
            int maxDepth = 0;
            int maxWidth = 0;
            int maxHeight = 0;
            foreach (IImagesLayer layer in layers)
            {
                maxDepth = Math.Max(maxDepth, layer.OutputDepth);
                maxWidth = Math.Max(maxWidth, layer.OutputWidth);
                maxHeight = Math.Max(maxHeight, layer.OutputHeight);
            }
            this.error1 = new Image(maxDepth, maxWidth, maxHeight);
            this.error2 = new Image(maxDepth, maxWidth, maxHeight);
            this.layersConnected = false;
            if (createIO)
            {
                this.SetInputGetOutput(new Image(layers.ElementAt(0).InputDepth, layers.ElementAt(0).InputWidth, layers.ElementAt(0).InputHeight));
            }
        }

        public PurelyConvolutionalNN(params IImagesLayer[] layers) : this(layers, true) { }
        
        public int InputDepth
        {
            get { return ((IImagesLayer)this.Layers.First()).InputDepth; }
        }

        public int InputWidth
        {
            get { return ((IImagesLayer)this.Layers.First()).InputWidth; }
        }

        public int InputHeight
        {
            get { return ((IImagesLayer)this.Layers.First()).InputHeight; }
        }

        public int OutputDepth
        {
            get { return ((IImagesLayer)this.Layers.Last()).OutputDepth; }
        }

        public int OutputWidth
        {
            get { return ((IImagesLayer)this.Layers.Last()).OutputWidth; }
        }

        public int OutputHeight
        {
            get { return ((IImagesLayer)this.Layers.Last()).OutputHeight; }
        }

        [OnDeserialized]
        private void SetValuesOnDeserialized(StreamingContext context)
        {
            this.layersConnected = false;
        }
        
        protected override Image NewError()
        {
            return new Image(this.OutputDepth, this.OutputWidth, this.OutputHeight);
        }
        
        public override void Feed(Image input, Image output, bool learning = false)
        {
            this.Input.FromImage(input);
            this.Feed(learning);
            output.FromImage(this.Output);
        }

        public override void BackPropagate(Image outputError, bool learning = true)
        {
            this.error2.FromImage(outputError);
            for (int i = this.Layers.Count - 1; i >= 0; i--)
            {
                ((IImagesLayer)this.Layers[i]).BackPropagate(this.error2, this.error1, learning);
                Image aux = this.error1;
                this.error1 = this.error2;
                this.error2 = aux;
            }
        }

        public override void BackPropagate(Image outputError, Image inputError, bool learning)
        {
            this.BackPropagate(outputError, learning);
            inputError.FromImage(this.error2);
        }
        
        public override double GetError(Image output, Image expectedOutput, Image error)
        {
            double retVal = 0;
            /*for (int i = 0; i < this.OutputDepth; i++)
            {
                for (int j = 0; j < this.OutputWidth; j++)
                {
                    for (int k = 0; k < this.OutputHeight; k++)
                    {
                        error.Raw[i, j, k] = expectedOutput.Raw[i, j, k] - output.Raw[i, j, k];
                        retVal += error.Raw[i, j, k] * error.Raw[i, j, k];
                    }
                }
            }*/
            return retVal;
        }

        public override double FeedAndGetError(Image input, Image expectedOutput, Image error, bool learning)
        {
            this.Input.FromImage(input);
            this.Feed(learning);
            return this.GetError(this.Output, expectedOutput, error);
        }

        public override void Save(Stream stream)
        {
            new DataContractSerializer(typeof(PurelyConvolutionalNN)).WriteObject(stream, this);
        }
        
        public override IUntypedLayer CreateSiamese()
        {
            return new PurelyConvolutionalNN(this, true);
        }

        public override IUntypedLayer Clone()
        {
            return new PurelyConvolutionalNN(this, false);
        }

        public override void SetInputAndOutput(Image input, Image output)
        {
            throw new NotImplementedException();
        }

        public override Image SetInputGetOutput(Image input)
        {
            throw new NotImplementedException();
        }
        
        protected override void AddTopLayer(ILayer<Image, Image> layer)
        {
            this.Layers.Insert(0, layer);
        }

        protected override void AddBottomLayer(ILayer<Image, Image> layer)
        {
            this.Layers.Add(layer);
        }

        protected override void RemoveTopLayer()
        {
            this.Layers.RemoveAt(0);
        }

        protected override void RemoveBottomLayer()
        {
            this.Layers.RemoveAt(this.Layers.Count - 1);
        }
    }
}
