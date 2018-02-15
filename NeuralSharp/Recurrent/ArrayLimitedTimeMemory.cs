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

namespace NeuralNetwork.Recurrent
{
    internal class ArrayLimitedTimeMemory : LimitedTimeMemory<double[]>, IArrayTimeMemory
    {
        private int size;

        public ArrayLimitedTimeMemory(int capacity, int size) : base(capacity)
        {
            this.size = size;
        }

        public int Size
        {
            get { return this.size; }
        }
        
        public void Add(double[] element, int skip)
        {
            int index = this.NormalizeIndex(this.Length);
            if (this.Array[index] == null)
            {
                this.Array[index] = new double[this.size];
            }
            System.Array.Copy(element, skip, this.Array[index], 0, this.size);
            this.IncrementLength();
        }

        public override void Add(double[] element)
        {
            this.Add(element, 0);
        }

        public double[] Add(bool zero = false)
        {
            int index = this.NormalizeIndex(this.Length);
            if (this.Array[index] == null)
            {
                this.Array[index] = new double[this.size];
            }
            this.IncrementLength();
            if (zero)
            {
                System.Array.Clear(this.Array[index], 0, this.Size);
            }
            return this.Array[index];
        }

        public void GetLast(double[] output, int skip = 0)
        {
            System.Array.Copy(this.Last, output, skip);
        }

        public void Get(int index, double[] output, int outputSkip = 0)
        {
            System.Array.Copy(this[index], 0, output, outputSkip, this.size);
        }
    }
}
