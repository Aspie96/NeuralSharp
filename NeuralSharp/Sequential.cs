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
    /// <typeparam name="TIn">The input type.</typeparam>
    /// <typeparam name="TOut">The output type.</typeparam>
    public abstract class Sequential<TIn, TOut> : ForwardLearner<TIn, TOut>, ILayer<TIn, TOut> where TIn : class where TOut : class
    {
        private List<IUntypedLayer> layers;
        private ILayerFrom<TIn> firstLayer;
        private ILayerTo<TOut> lastLayer;

        /// <summary>Either creates a siamese of the given <code>Sequential</code> instance or clones is.</summary>
        /// <param name="original">The original instance to be created a siamese or cloned.</param>
        /// <param name="siamese"><code>true</code> if a siamese is to be created, <code>false</code> if a clone is.</param>
        protected Sequential(Sequential<TIn, TOut> original, bool siamese)
        {
            this.layers = new List<IUntypedLayer>();
            if (siamese)
            {
                foreach (IUntypedLayer layer in original.Layers)
                {
                    this.layers.Add(layer.CreateSiamese());
                }
            }
            else
            {
                foreach (IUntypedLayer layer in original.Layers)
                {
                    this.layers.Add(layer.Clone());
                }
            }
            this.firstLayer = (ILayerFrom<TIn>)this.layers.First();
            this.lastLayer = (ILayerTo<TOut>)this.layers.Last();
        }

        /// <summary>Creates an instance of the <code>Sequential</code> class.</summary>
        /// <param name="layers">The layers of the learner.</param>
        public Sequential(ICollection<IUntypedLayer> layers)
        {
            this.layers = layers.ToList();
            this.firstLayer = (ILayerFrom<TIn>)layers.First();
            this.lastLayer = (ILayerTo<TOut>)layers.Last();
        }
        
        /// <summary>The layers of the learner.</summary>
        protected List<IUntypedLayer> Layers
        {
            get { return this.layers; }
        }
        
        /// <summary>The input object of the learner.</summary>
        public TIn Input
        {
            get { return this.firstLayer.Input; }
        }

        /// <summary>The output object of the learner.</summary>
        public TOut Output
        {
            get { return this.lastLayer.Output; }
        }
        
        /// <summary>The amount of parameters.</summary>
        public override int Parameters
        {
            get
            {
                return this.Layers.Sum(delegate (IUntypedLayer layer)
                {
                    return layer.Parameters;
                });
            }
        }

        /// <summary>Sets the input object and the output object of the network.</summary>
        /// <param name="input">The input object to be set.</param>
        /// <param name="output">The output object to be set.</param>
        public abstract void SetInputAndOutput(TIn input, TOut output);

        /// <summary>Sets the input object and creates and sets the output object.</summary>
        /// <param name="input">The input object to be set.</param>
        /// <returns>The created output.</returns>
        public abstract TOut SetInputGetOutput(TIn input);
        
        /// <summary>Creates a siamese of the layer.</summary>
        /// <returns>The created siamese.</returns>
        public abstract IUntypedLayer CreateSiamese();

        /// <summary>Creates a clone of the layer.</summary>
        /// <returns>The created <code>Sequential</code> instance.</returns>
        public abstract IUntypedLayer Clone();

        /// <summary>Adds a top layer.</summary>
        /// <param name="layer">The layer to be added.</param>
        protected abstract void AddTopLayer(ILayer<TIn, TIn> layer);

        /// <summary>Adds a layer to the bottom.</summary>
        /// <param name="layer">The layer to be added.</param>
        protected abstract void AddBottomLayer(ILayer<TOut, TOut> layer);

        /// <summary>Removes a top layer.</summary>
        protected abstract void RemoveTopLayer();

        /// <summary>Removes a bottom layer.</summary>
        protected abstract void RemoveBottomLayer();

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
    }
}
