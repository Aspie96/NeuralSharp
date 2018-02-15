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

using System;

namespace NeuralNetwork.Recurrent
{
    internal class ArrayInfiniteTimeMemory : InfiniteTimeMemory<double[]>, IArrayTimeMemory
    {
        private int length;
        private int size;

        public ArrayInfiniteTimeMemory(int size)
        {
            this.length = 0;
            this.size = size;
        }
        
        public override int Length
        {
            get { return this.length; }
        }

        public int Size
        {
            get { return this.size; }
        }
        
        public void Add(double[] element, int skip, bool alwaysCopy)
        {
            if (this.List.Count > this.Length)
            {
                Array.Copy(element, this.List[this.Length], this.size);
            }
            else if (alwaysCopy || skip != 0 || element.Length > this.size)
            {
                double[] copy = new double[this.size];
                Array.Copy(element, skip, copy, 0, this.size);
                this.List.Add(copy);
            }
            else
            {
                this.List.Add(element);
            }
            this.length++;
        }

        public void GetLast(double[] output, int skip = 0)
        {
            Array.Copy(this.Last, output, skip);
        }

        public void Add(double[] element, int skip = 0)
        {
            this.Add(element, skip, false);
        }

        public override void Add(double[] element)
        {
            this.Add(element, 0, true);
        }

        public double[] Add(bool zero = false)
        {
            double[] retVal;
            if (this.Length < this.List.Count)
            {
                retVal = this.List[this.Length];
            }
            else
            {
                retVal = new double[this.Size];
                this.List.Add(retVal);
            }
            this.length++;
            if (zero)
            {
                Array.Clear(retVal, 0, this.Size);
            }
            return retVal;
        }
        
        public override void Clear()
        {
            this.length = 0;
        }

        public void Get(int index, double[] output, int outputSkip = 0)
        {
            Array.Copy(this[index], 0, output, outputSkip, this.size);
        }
    }
}
