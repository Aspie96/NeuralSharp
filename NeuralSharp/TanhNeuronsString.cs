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
using System.Text;
using System.Threading.Tasks;

namespace NeuralSharp
{
    public class TanhNeuronsString : NeuronsString
    {
        protected TanhNeuronsString(TanhNeuronsString original, bool siamese) : base(original, siamese) { }

        public TanhNeuronsString(int length, bool createIO = false) : base(length, createIO) { }
        
        protected override double Activation(double input)
        {
            return Math.Tanh(input);
        }

        protected override double ActivationDerivative(double input, double output)
        {
            return 1.0 - output * output;
        }

        public override IUntypedLayer CreateSiamese()
        {
            return new TanhNeuronsString(this, true);
        }

        public override IUntypedLayer Clone()
        {
            return new TanhNeuronsString(this, false);
        }
    }
}
