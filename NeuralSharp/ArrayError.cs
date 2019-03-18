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
    /// <summary>Represents an error function on an array.</summary>
    public class ArrayError : Error<double[]>, IArrayError
    {
        /// <summary>Represents an error function on an array.</summary>
        /// <param name="outputArray">The output.</param>
        /// <param name="outputSkip">The index of the first entry of the output array to be used.</param>
        /// <param name="expectedOutputArray">The expected output.</param>
        /// <param name="expectedOutputSkip">The index of the first entry of the expected output array to be used.</param>
        /// <param name="errorArray">The array to be written the error into.</param>
        /// <param name="errorSkip">The index of the first entry of the error array to be used.</param>
        /// <param name="length">The lenght of the output.</param>
        /// <returns>The error.</returns>
        public delegate double ArrayErrorFunction(double[] outputArray, int outputSkip, double[] expectedOutputArray, int expectedOutputSkip, double[] errorArray, int errorSkip, int length);

        private ArrayErrorFunction arrayErrorFunction;
        
        /// <summary>Creates an instance of the <code>ArrayError</code> class.</summary>
        /// <param name="arrayErrorFunction">The error function to be used.</param>
        public ArrayError(ArrayErrorFunction arrayErrorFunction) : base(delegate(double[] output, double[] expected, double[] error)
        {
            return arrayErrorFunction(output, 0, expected, 0, error, 0, Math.Min(output.Length, Math.Min(expected.Length, error.Length)));
        })
        {
            this.arrayErrorFunction = arrayErrorFunction;
        }

        /// <summary>Gets the error, given the actual output and the expected output.</summary>
        /// <param name="outputArray">The output.</param>
        /// <param name="outputSkip">The index of the first entry of the output to be used.</param>
        /// <param name="expectedOutputArray">The expected output.</param>
        /// <param name="expectedOutputSkip">The index of the first entry of the expected output array to be used.</param>
        /// <param name="errorArray">The array to be written the error into.</param>
        /// <param name="errorSkip">The index of the first entry of the error array to be used.</param>
        /// <param name="length">The lenght of the output.</param>
        /// <returns>The error.</returns>
        public double GetError(double[] outputArray, int outputSkip, double[] expectedOutputArray, int expectedOutputSkip, double[] errorArray, int errorSkip, int length)
        {
            return this.arrayErrorFunction(outputArray, outputSkip, expectedOutputArray, expectedOutputSkip, errorArray, errorSkip, length);
        }

        /// <summary>Gets the error, given the actual output and the expected output.</summary>
        /// <param name="outputArray">The output.</param>
        /// <param name="expectedOutputArray">The expected output.</param>
        /// <param name="errorArray">The array to be written the errro into.</param>
        /// <param name="length">The lenght of the output.</param>
        /// <returns>The error.</returns>
        public double GetError(double[] outputArray, double[] expectedOutputArray, double[] errorArray, int length)
        {
            return this.arrayErrorFunction(outputArray, 0, expectedOutputArray, 0, errorArray, 0, length);
        }
    }
}
