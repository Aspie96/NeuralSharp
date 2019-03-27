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

namespace NeuralSharp.Convolutional
{
    /// <summary>Represents a layer whose input is an image and whose output is an image.</summary>
    public interface IImagesLayer : ILayer<Image, Image>
    {
        /// <summary>The depth of the input of the layer.</summary>
        int InputDepth { get; }

        /// <summary>The width of the input of the layer.</summary>
        int InputWidth { get; }

        /// <summary>The height of the input of the layer.</summary>
        int InputHeight { get; }

        /// <summary>The depth of the output of the layer.</summary>
        int OutputDepth { get; }

        /// <summary>The width of the output of the layer.</summary>
        int OutputWidth { get; }

        /// <summary>The height of the output of the layer.</summary>
        int OutputHeight { get; }
    }
}
