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
using System.Linq;
using System.Runtime.Serialization;
using System.Text;
using System.Threading.Tasks;

namespace NeuralSharp
{
    [DataContract]
    public class ImageDropoutLayer : IImagesLayer
    {
        private Image input;
        private Image output;
        [DataMember]
        private int depth;
        [DataMember]
        private int width;
        [DataMember]
        private int height;
        private bool[] dropped;
        [DataMember]
        private double dropChance;
        
        protected ImageDropoutLayer(ImageDropoutLayer original, bool siamese)
        {
            this.depth = original.Depth;
            this.width = original.width;
            this.height = original.Height;
            this.dropped = Backbone.CreateArray<bool>(depth * width * height);
            this.dropChance = original.DropChance;
        }

        public ImageDropoutLayer(int depth, int width, int height, double dropChance, bool createIO = false)
        {
            this.depth = depth;
            this.width = width;
            this.height = height;
            if (createIO)
            {
                this.input = new Image(depth, width, height);
                this.output = new Image(depth, width, height);
            }
            this.dropped = Backbone.CreateArray<bool>(depth * width * height);
            this.dropChance = dropChance;
        }

        public Image Input
        {
            get { return this.input; }
        }

        public Image Output
        {
            get { return this.output; }
        }

        public int InputDepth
        {
            get { return this.depth; }
        }

        public int InputWidth
        {
            get { return this.width; }
        }

        public int InputHeight
        {
            get { return this.height; }
        }

        public int OutputDepth
        {
            get { return this.depth; }
        }

        public int OutputWidth
        {
            get { return this.width; }
        }

        public int OutputHeight
        {
            get { return this.height; }
        }
        
        public int Depth
        {
            get { return this.depth; }
        }

        public int Width
        {
            get { return this.width; }
        }

        public int Height
        {
            get { return this.height; }
        }

        public double DropChance
        {
            get { return this.dropChance; }
        }
        
        public int Parameters
        {
            get { return 0; }
        }

        [OnDeserialized]
        private void SetValuesOnDeserialized(StreamingContext context)
        {
            this.dropped = Backbone.CreateArray<bool>(this.Depth * this.Width * this.Height);
        }

        public void Feed(bool learning = false)
        {
            Backbone.ApplyImageDropout(this.Input.Raw, this.Output.Raw, this.Depth, this.Width, this.Height, this.dropped, this.DropChance, learning);
        }

        public void BackPropagate(Image outputError, Image inputError, bool learning)
        {
            Backbone.BackpropagateImageDropout(this.Input.Raw, this.Output.Raw, this.Depth, this.Width, this.Height, this.dropped, this.DropChance, learning, outputError.Raw, outputError.Depth, outputError.Width, outputError.Height, inputError.Raw, inputError.Depth, inputError.Width, inputError.Height);
        }

        public void UpdateWeights(double rate, double momentum = 0.0) { }
        
        public void SetInputAndOutput(Image input, Image output)
        {
            this.input = input;
            this.output = output;
        }

        public Image SetInputGetOutput(Image input)
        {
            this.input = input;
            return this.output = new Image(this.OutputDepth, this.OutputWidth, this.OutputHeight);
        }

        public ILayer<Image, Image> CreateSiamese(bool createIO = false)
        {
            return new ImageDropoutLayer(this.Depth, this.Width, this.Height, this.DropChance, createIO);
        }

        public virtual IUntypedLayer CreateSiamese()
        {
            return new ImageDropoutLayer(this, true);
        }

        public IUntypedLayer Clone()
        {
            return new ImageDropoutLayer(this, false);
        }
    }
}
