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
using System.Text;
using System.Threading.Tasks;

namespace NeuralSharp
{
    public class ImageAutoencoder : Autoencoder<Image>
    {
        private int depth;
        private int width;
        private int height;

        protected ImageAutoencoder(ImageAutoencoder original, bool siamese) : base(original, siamese)
        {
            this.depth = original.depth;
            this.width = original.width;
            this.height = original.height;
        }

        public ImageAutoencoder(ConvolutionalNN encoder, DeConvolutionalNN decoder, int codeSize) : base(encoder, decoder, codeSize)
        {
            this.depth = encoder.InputDepth;
            this.width = encoder.InputWidth;
            this.height = encoder.InputHeight;
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

        public override void BackPropagate(Image outputError, bool learning = true)
        {
            
        }
        
        public override IUntypedLayer Clone()
        {
            return new ImageAutoencoder(this, false);
        }

        public override IUntypedLayer CreateSiamese()
        {
            return new ImageAutoencoder(this, true);
        }

        public override void Feed(Image input, Image output, bool learning = false)
        {
            this.Encoder.Input.FromImage(input);
            this.Encoder.Feed(learning);
            this.Encoder.Feed(learning);
            output.FromImage(this.Decoder.Output);
        }

        public override double FeedAndGetError(Image input, Image expectedOutput, Image error, bool learning)
        {
            this.Input.FromImage(input);
            this.Feed(learning);
            return this.GetError(this.Output, expectedOutput, error);
        }

        public override double GetError(Image output, Image expectedOutput, Image error)
        {
            double retVal = 0.0;
            for (int i = 0; i < this.Output.Depth; i++)
            {
                for (int j = 0; j < this.Output.Width; j++)
                {
                    for (int k = 0; k < this.Output.Height; k++)
                    {
                        double e = expectedOutput.GetValue(i, j, k) - output.GetValue(i, j, k);
                        retVal += e * e;
                        error.SetValue(i, j, k, (float)e);
                    }
                }
            }
            return retVal;
        }

        protected override Image NewError()
        {
            return new Image(this.Depth, this.Width, this.Height);
        }
    }
}
