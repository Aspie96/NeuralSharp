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
    /// <summary>Represents the error function of an array.</summary>
    public interface IArrayError : IError<float[]>
    {
        /// <summary>Gets the error given the output array and the expected output array.</summary>
        /// <param name="outputArray">The output.</param>
        /// <param name="outputSkip">The index of the first entry of the output array to be used.</param>
        /// <param name="expectedOutputArray">The expected output.</param>
        /// <param name="expectedOutputSkip">The index of the first entry of the output array to be used.</param>
        /// <param name="errorArray">The array to be written the error into.</param>
        /// <param name="errorSkip">The index of the first entry of the error array to be used.</param>
        /// <param name="length">The lenght of the output.</param>
        /// <returns>The error.</returns>
        float GetError(float[] outputArray, int outputSkip, float[] expectedOutputArray, int expectedOutputSkip, float[] errorArray, int errorSkip, int length);

        /// <summary>Gets the error given the output array and the expected output array.</summary>
        /// <param name="outputArray">The output.</param>
        /// <param name="expectedOutputArray">The expected output.</param>
        /// <param name="errorArray">The array to be written the error into.</param>
        /// <param name="length">The lenght of the output.</param>
        /// <returns>The error.</returns>
        float GetError(float[] outputArray, float[] expectedOutputArray, float[] errorArray, int length);
    }
}
