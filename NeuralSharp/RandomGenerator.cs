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
    internal static class RandomGenerator
    {
        private static Random r = new Random(220);

        public static double GetDouble()
        {
            return RandomGenerator.r.NextDouble();
        }

        public static float GetFloat()
        {
            return (float)RandomGenerator.GetDouble();
        }

        public static void ShuffleArray<T>(T[] arr)
        {
            for (int i = 0; i < arr.Length; i++)
            {
                int j = RandomGenerator.r.Next(arr.Length);
                T aux = arr[i];
                arr[i] = arr[j];
                arr[j] = aux;
            }
        }

        public static double GetNormalNumber(double variance)
        {
            //return RandomGenerator.r.NextDouble() * Math.Sqrt(variance * 12) - Math.Sqrt(variance * 3);
            return Math.Sqrt(variance) * Math.Sqrt(-2.0 * Math.Log(RandomGenerator.GetDouble())) * Math.Sin(2.0 * Math.PI * RandomGenerator.GetDouble());
        }

        public static int GetInt(int max)
        {
            return RandomGenerator.r.Next(max);
        }
    }
}
