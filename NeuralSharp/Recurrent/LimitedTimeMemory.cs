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
    internal class LimitedTimeMemory<T> : ITimeMemory<T>
    {
        private T[] array;
        private int firstIndex;
        private int length;
        private bool cropped;

        public LimitedTimeMemory(int capacity)
        {
            this.array = new T[capacity];
            this.firstIndex = 0;
            this.length = 0;
            this.cropped = false;
        }
        
        public T Last
        {
            get { return this[this.Length - 1]; }
        }

        protected T[] Array
        {
            get { return this.array; }
        }

        public T this[int index]
        {
            get{ return this.Get(index); }
        }

        public int Length
        {
            get{return this.length;}
        }

        public int Capacity
        {
            get { return this.array.Length; }
        }

        public bool Cropped
        {
            get{return this.cropped;}
        }

        protected int NormalizeIndex(int index)
        {
            return (this.firstIndex + index) % this.Capacity;
        }

        protected void IncrementLength()
        {
            if (this.Length < this.Capacity)
            {
                this.length++;
            }
            else
            {
                this.firstIndex = this.NormalizeIndex(1);
                this.cropped = true;
            }
        }

        public virtual void Add(T element)
        {
            int index = this.NormalizeIndex(this.Length);
            this.array[index] = element;
            this.IncrementLength();
        }

        public T Get(int index)
        {
            return this.array[(this.firstIndex + index) % this.Length];
        }

        public void Clear()
        {
            this.firstIndex = 0;
            this.length = 0;
            this.cropped = false;
        }
    }
}
