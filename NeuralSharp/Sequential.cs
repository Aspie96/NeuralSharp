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
    public abstract class Sequential<TIn, TOut> : ForwardLearner<TIn, TOut>, ILayer<TIn, TOut> where TIn : class where TOut : class
    {
        private List<IUntypedLayer> layers;
        private ILayerFrom<TIn> firstLayer;
        private ILayerTo<TOut> lastLayer;

        public Sequential(Sequential<TIn, TOut> original, bool siamese)
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

        public Sequential(ICollection<IUntypedLayer> layers)
        {
            this.layers = layers.ToList();
            this.firstLayer = (ILayerFrom<TIn>)layers.First();
            this.lastLayer = (ILayerTo<TOut>)layers.Last();
        }
        
        protected List<IUntypedLayer> Layers
        {
            get { return this.layers; }
        }
        
        public TIn Input
        {
            get { return this.firstLayer.Input; }
        }

        public TOut Output
        {
            get { return this.lastLayer.Output; }
        }
        
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

        public abstract void SetInputAndOutput(TIn input, TOut output);
        public abstract TOut SetInputGetOutput(TIn input);
        public abstract IUntypedLayer CreateSiamese();
        public abstract IUntypedLayer Clone();
        protected abstract void AddTopLayer(ILayer<TIn, TIn> layer);
        protected abstract void AddBottomLayer(ILayer<TOut, TOut> layer);
        protected abstract void RemoveTopLayer();
        protected abstract void RemoveBottomLayer();

        public void Feed(bool learning = false)
        {
            foreach (IArraysLayer layer in this.Layers)
            {
                layer.Feed(learning);
            }
        }
        
        public override void UpdateWeights(double rate, double momentum = 0.0)
        {
            foreach (IArraysLayer layer in this.Layers)
            {
                layer.UpdateWeights(rate, momentum);
            }
        }
    }
}
