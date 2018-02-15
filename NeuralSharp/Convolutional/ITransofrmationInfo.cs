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
    /// <summary>Represents information about the transformation of an image.</summary>
    public interface ITransofrmationInfo
    {
        /// <summary>Gets the size of the output image of a transformation represented by this info given the size of the input image.</summary>
        /// <param name="depthBefore">The depth of the input image.</param>
        /// <param name="widthBefore">The width of the input image.</param>
        /// <param name="heightBefore">The height of the input image.</param>
        /// <param name="depth">The depth of the otuput image.</param>
        /// <param name="width">The width of the output image.</param>
        /// <param name="height">The height of the otuput image.</param>
        void SizeAfter(int depthBefore, int widthBefore, int heightBefore, out int depth, out int width, out int height);

        /// <summary>Creates a transformation corresponding to this info.</summary>
        /// <param name="image1">The input image of the transformation.</param>
        /// <param name="image2">The output image of the transformation.</param>
        /// <returns>The generated transformation.</returns>
        IImageTransformation GetTransformation(Image image1, Image image2);
    }
}
