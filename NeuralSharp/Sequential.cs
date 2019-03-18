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
    /// <summary>Represents a sequential learner.</summary>
    /// <typeparam name="TData">The data type.</typeparam>
    /// <typeparam name="TLayer">The layer type.</typeparam>
    /// <typeparam name="TErrFunc">The error function type.</typeparam>
    public abstract class Sequential<TData, TLayer, TErrFunc> : ForwardLearner<TData, TData, TErrFunc>, ILayer<TData, TData> where TData : class where TLayer : ILayer<TData, TData> where TErrFunc : IError<TData>
    {
        private List<TLayer> layers;
        private object siameseID;

        /// <summary>Either creates a siamese of the given <code>Sequential</code> instance or clones is.</summary>
        /// <param name="original">The original instance to be created a siamese or cloned.</param>
        /// <param name="siamese"><code>true</code> if a siamese is to be created, <code>false</code> if a clone is.</param>
        public Sequential(Sequential<TData, TLayer, TErrFunc> original, bool siamese)
        {
            this.layers = new List<TLayer>();
            this.layers = original.layers.ConvertAll(delegate (TLayer layer)
            {
                if (siamese)
                {
                    return (TLayer)layer.CreateSiamese();
                }
                else
                {
                    return (TLayer)layer.Clone();
                }
            });
            if (siamese)
            {
                this.siameseID = original.SiameseID;
            }
            else
            {
                this.siameseID = new object();
            }
        }

        /// <summary>Creates an instance of the <code>Sequential</code> class.</summary>
        /// <param name="layers">The layers of the learner.</param>
        public Sequential(ICollection<TLayer> layers)
        {
            this.layers = layers.ToList();
            this.siameseID = new object();
        }

        /// <summary>The layers of the learner.</summary>
        protected ICollection<TLayer> Layers
        {
            get { return this.layers.AsReadOnly(); }
        }

        /// <summary>The first layer of the learner.</summary>
        public TLayer FirstLayer
        {
            get { return this.layers[0]; }
        }

        /// <summary>The last layer of the learner.</summary>
        public TLayer LastLayer
        {
            get { return this.Layers.Last(); }
        }

        /// <summary>The input object of the learner.</summary>
        public TData Input
        {
            get { return this.layers[0].Input; }
        }

        /// <summary>The output object of the learner.</summary>
        public TData Output
        {
            get { return this.layers.Last().Output; }
        }

        /// <summary>The siamese identifier of the learner.</summary>
        public object SiameseID
        {
            get { return this.siameseID; }
        }

        /// <summary>The amount of parameters.</summary>
        public int Parameters
        {
            get
            {
                return this.Layers.Sum(delegate (TLayer layer)
                {
                    return layer.Parameters;
                });
            }
        }

        /// <summary>Sets the input object and the output object of the network.</summary>
        /// <param name="input">The input object to be set.</param>
        /// <param name="output">The output object to be set.</param>
        public abstract void SetInputAndOutput(TData input, TData output);

        /// <summary>Sets the input object and creates and sets the output object.</summary>
        /// <param name="input">The input object to be set.</param>
        /// <returns>The created output.</returns>
        public abstract TData SetInputGetOutput(TData input);

        /// <summary>Creates a siamese of the layer.</summary>
        /// <returns>The created siamese.</returns>
        public abstract ILayer<TData, TData> CreateSiamese();

        /// <summary>Creates a clone of the layer.</summary>
        /// <returns>The created <code>Sequential</code> instance.</returns>
        public abstract ILayer<TData, TData> Clone();

        /// <summary>Adds a top layer.</summary>
        /// <param name="layer">The layer to be added.</param>
        protected virtual void AddTopLayer(TLayer layer)
        {
            this.layers.Insert(0, layer);
        }

        /// <summary>Adds a layer to the bottom.</summary>
        /// <param name="layer">The layer to be added.</param>
        protected virtual void AddBottomLayer(TLayer layer)
        {
            this.layers.Add(layer);
        }

        /// <summary>Removes a top layer.</summary>
        protected virtual void RemoveTopLayer()
        {
            this.layers.RemoveAt(0);
        }

        /// <summary>Removes a bottom layer.</summary>
        protected virtual void RemoveBottomLayer()
        {
            this.layers.RemoveAt(this.layers.Count - 1);
        }

        /// <summary>Feeds the layer forward.</summary>
        /// <param name="learning">Whether the layer is being used in a training session.</param>
        public void Feed(bool learning = false)
        {
            foreach (IArraysLayer layer in this.Layers)
            {
                layer.Feed(learning);
            }
        }
        
        /// <summary>Updates the weights of the learner.</summary>
        /// <param name="rate">The learning rate to be used.</param>
        /// <param name="momentum">The momentum to be used.</param>
        public override void UpdateWeights(double rate, double momentum = 0.0)
        {
            foreach (IArraysLayer layer in this.Layers)
            {
                layer.UpdateWeights(rate, momentum);
            }
        }

        /// <summary>Counts the amount of parameters of the learner.</summary>
        /// <param name="siameseIDs">The siamese identifiers to be excluded. The siamese identifiers of the learner will be aded to the list.</param>
        /// <returns>The amount of parameters.</returns>
        public int CountParameters(List<object> siameseIDs)
        {
            if (siameseIDs.Contains(this.SiameseID))
            {
                return 0;
            }
            siameseIDs.Add(this.SiameseID);
            return this.Layers.Sum(delegate (TLayer layer)
            {
                return layer.CountParameters(siameseIDs);
            });
        }
    }
}
