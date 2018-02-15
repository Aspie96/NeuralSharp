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

namespace NeuralNetwork.Convolutional
{
    /// <summary>Represents a transformation of an image in a convolutional neural network.</summary>
    public interface IImageTransformation
    {
        /// <summary>The input image to be transformed.</summary>
        Image Input { get; }

        /// <summary>The output image.</summary>
        Image Output { get; }

        /// <summary>The number of parameters in this transformation.</summary>
        int Params { get; }

        /// <summary>Information about this transformation.</summary>
        ITransofrmationInfo Info { get; }

        /// <summary>Sets the input and output image of this transformation. Only to be used when strictly necessary.</summary>
        /// <param name="input">The input image to be set.</param>
        /// <param name="output">The output image to be set.</param>
        void SetLayers(Image input, Image output);

        /// <summary>Transforms the input image, writing the results to the second.</summary>
        void Feed();

        /// <summary>Backpropagates the given error trough this transformation, updating its parameters.</summary>
        /// <param name="error2">The error of the output image, to be backpropagated. It must refer to the latest feeding process.</param>
        /// <param name="error1">The image to be written the error of the input image into.</param>
        /// <param name="rate">The learning rate at which the parameters are to be updated.</param>
        void BackPropagate(Image error2, Image error1, double rate);

        /// <summary>Creates a copy of this transformation.</summary>
        /// <param name="input">The input image to be set for the copy.</param>
        /// <param name="output">The output image to be set for the copy.</param>
        /// <returns>The generated instance.</returns>
        IImageTransformation Clone(Image input, Image output);
    }
}
