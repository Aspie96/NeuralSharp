using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralSharp
{
    /// <summary>Represents an error function.</summary>
    /// <typeparam name="T">The type the error function can be applied to.</typeparam>
    public class Error<T> : IError<T> where T : class
    {
        /// <summary>Represents an error function.</summary>
        /// <param name="output">The output.</param>
        /// <param name="expectedOutput">The expected output.</param>
        /// <param name="error">The object to be written the error into.</param>
        /// <returns>The error.</returns>
        public delegate float ErrorFunction(T output, T expectedOutput, T error);

        private ErrorFunction errorFunction;
        
        /// <summary>Creates an instance of the <code>Error</code> class.</summary>
        /// <param name="errorFunction">The error function to be used.</param>
        public Error(ErrorFunction errorFunction)
        {
            this.errorFunction = errorFunction;
        }

        /// <summary>Gets the error given the actual output and the expected output.</summary>
        /// <param name="output">The actual output.</param>
        /// <param name="expectedOutput">The expected output.</param>
        /// <param name="error">The object to be written the error into.</param>
        /// <returns>The error.</returns>
        public float GetError(T output, T expectedOutput, T error)
        {
            return this.errorFunction(output, expectedOutput, error);
        }
    }
}
