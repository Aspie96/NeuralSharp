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
using System.Drawing;
using System.IO;
using System.Drawing.Imaging;
using System.Runtime.Serialization;

namespace NeuralSharp
{
    [DataContract]
    public class Image
    {
        private int depth;
        private int width;
        private int height;
        private float[] data;   // [depth, width, height]
        
        public Image(int depth, int width, int height, bool c = false)
        {
            this.depth = depth;
            this.width = width;
            this.height = height;
            if (c)
            {
                this.data = new float[depth * width * height];
            }
            else
            {
                this.data = Backbone.CreateArray<float>(depth * width * height);
            }
        }
        
        public int Depth
        {
            get { return this.depth; }
        }
        
        public int Width
        {
            get { return this.width; }
        }
        
        public int Height
        {
            get { return this.height; }
        }
        
        public float[] Raw
        {
            get { return this.data; }
        }
        
        public float GetValue(int w, int x, int y)
        {
            return this.Raw[w * this.width * this.height + x * this.height + y];
        }
        
        public void SetValue(int w, int x, int y, float value)
        {
            this.Raw[w * this.width * this.height + x * this.height + y] = value;
        }
        
        public void ToArray(int depth, int width, int height, double[] array, int arraySkip = 0)
        {
            int index = 0;
            Backbone.ImageToArray(this.Raw, this.Depth, this.Width, this.Height, depth, width, height, array, arraySkip);
        }

        public void ToArray(double[] array, int arraySkip = 0)
        {
            this.ToArray(this.Depth, this.Width, this.Height, array, arraySkip);
        }

        public void FromImage(Image image)
        {
            this.FromImage(image, Math.Min(this.Depth, image.Depth), Math.Min(this.Width, image.Width), Math.Min(this.Height, image.Height));
        }
        
        public void FromImage(Image image, int sourceW, int sourceX, int sourceY, int thisW, int thisX, int thisY, int depth, int width, int height)
        {
            Backbone.ImageToImage(this.Raw, this.Depth, this.Width, this.Height, image.Raw, image.Depth, image.Width, image.Height, sourceW, sourceX, sourceY, thisW, thisX, thisY, depth, width, height);
        }
        
        public void FromArray(double[] array, int skip, int thisW, int thisX, int thisY, int depth, int width, int height)
        {
            Backbone.ArrayToImage(this.Raw, this.Depth, this.Width, this.Height, array, skip, thisW, thisX, thisY, depth, width, height);
        }

        public void FromArray(double[] array, int skip = 0)
        {
            this.FromArray(array, skip, 0, 0, 0, this.Depth, this.Width, this.Height);
        }

        public void FromImage(Image image, int depth, int width, int height)
        {
            this.FromImage(image, 0, 0, 0, 0, 0, 0, depth, width, height);
        }
        
        public void FromImageScaled(Image image, int sourceW, int sourceX, int sourceY, int thisW, int thisX, int thisY, int depth, int sourceWidth, int sourceHeight, int thisWidth, int thisHeight)
        {
            float scaleX = (thisWidth > sourceWidth) ? ((thisWidth - 1.0F) / (sourceWidth - 1.0F)) : ((float)thisWidth / sourceWidth);
            float scaleY = (thisHeight > sourceHeight) ? ((thisHeight - 1.0F) / (sourceHeight - 1.0F)) : ((float)thisHeight / sourceHeight);
            float stepX = Math.Max(1.0F / scaleX, 1.0F);
            float stepY = Math.Max(1.0F / scaleY, 1.0F);
            float globalWeight = 1.0F / (stepX * stepY);
            for (int i = 0; i < thisWidth; i++)
            {
                float fromXNum = i / scaleX;
                int fromX = (int)fromXNum;
                float fromXWeight = fromX + 1 - fromXNum;
                float toXNum = fromXNum + stepX;
                float toX = Math.Min((int)Math.Ceiling(toXNum), sourceWidth);
                float toXWeight = toXNum - (int)toXNum;
                if (thisWidth < sourceWidth && toXWeight == 0.0F)
                {
                    toXWeight = 1.0F;
                }
                for (int j = 0; j < thisHeight; j++)
                {
                    float fromYNum = j / scaleY;
                    int fromY = (int)fromYNum;
                    float fromYWeight = fromY + 1 - fromYNum;
                    float toYNum = fromYNum + stepY;
                    float toY = Math.Min((int)Math.Ceiling(toYNum), sourceHeight);
                    float toYWeight = toYNum - (int)toYNum;
                    if (thisHeight < sourceHeight && toYWeight == 0.0F)
                    {
                        toYWeight = 1.0F;
                    }

                    for (int k = 0; k < this.Depth; k++)
                    {
                        this.SetValue(k + thisW, i + thisX, j + thisY, 0.0F);
                    }
                    double totalWeight = 0.0;
                    for (int k = fromX; k < toX; k++)
                    {
                        for (int l = fromY; l < toY; l++)
                        {
                            float weight = globalWeight;
                            if (k == fromX)
                            {
                                weight *= fromXWeight;
                            }
                            else if (k == toX - 1)
                            {
                                weight *= toXWeight;
                            }
                            if (l == fromY)
                            {
                                weight *= fromYWeight;
                            }
                            else if (l == toY - 1)
                            {
                                weight *= toYWeight;
                            }
                            for (int m = 0; m < depth; m++)
                            {
                                this.SetValue(m + thisW, i + thisX, j + thisY, this.GetValue(m + thisW, i + thisX, j + thisY) + image.GetValue(m + sourceW, k + sourceX, l + sourceY) * weight);
                                totalWeight += weight;
                            }
                        }
                    }
                }
            }
        }
        
        public void FromImageScaled(Image image, int sourceX, int sourceY, int thisX, int thisY, int sourceWidth, int sourceHeight, int thisWidth, int thisHeight)
        {
            this.FromImageScaled(image, 0, sourceX, sourceY, 0, thisX, thisY, Math.Min(this.Depth, image.Depth), sourceWidth, sourceHeight, thisWidth, thisHeight);
        }
        
        public void FromImageScaled(Image image)
        {
            this.FromImageScaled(image, 0, 0, 0, 0, image.Width, image.height, this.Width, this.Height);
        }
        
        public void FromBitmap(Bitmap image)
        {
            if (this.Depth == 3)
            {
                for (int i = 0; i < this.Width; i++)
                {
                    for (int j = 0; j < this.Height; j++)
                    {
                        Color pixel = image.GetPixel(i, j);
                        this.Raw[i * this.Height + j] = pixel.R / 255.0F;
                        this.Raw[this.Width * this.Height + i * this.Height + j] = pixel.G / 255.0F;
                        this.Raw[2 * this.Width * this.Height + i * this.Height + j] = pixel.B / 255.0F;
                    }
                }
            }
            else if (this.Depth == 1)
            {
                for (int i = 0; i < this.Width; i++)
                {
                    for (int j = 0; j < this.Height; j++)
                    {
                        Color pixel = image.GetPixel(i, j);
                        this.Raw[i * this.Height + j] = (float)Math.Sqrt(pixel.R * pixel.R * 0.241F + pixel.G * pixel.G * 0.691F + pixel.B * pixel.B * 0.068F) / 255.0F;
                    }
                }
            }
        }
        
        public void FromBitmap(string path)
        {
            this.FromBitmap(new Bitmap(path));
        }

        public static Image AdaptFromBitmap(string path, int width, int height)
        {
            Bitmap image = new Bitmap(path);
            double ratio = (double)width / height;
            double originalRatio = (double)image.Width / (double)image.Height;
            double scale;
            double skipX;
            double skipY;
            if (originalRatio > ratio)
            {
                skipX = image.Width * (1 - ratio / originalRatio) / 2.0F;
                skipY = 0.0;
                scale = (double)image.Height / height;
            }
            else
            {
                skipY = image.Height * (1 - originalRatio / ratio) / 2.0F;
                skipX = 0.0;
                scale = (double)image.Width / width;
            }
            Image retVal = new Image(3, width, height);
            if (scale <= 1.0 || true)
            {
                for (int i = 0; i < width; i++)
                {
                    for (int j = 0; j < height; j++)
                    {
                        double x = skipX + scale * i;
                        double y = skipY + scale * j;
                        int apprX = (int)x;
                        int apprY = (int)y;
                        double a = x - apprX;
                        double b = y - apprY;
                        double weight0 = a * b;
                        double weight1 = a * (1 - b);
                        double weight2 = (1 - a) * b;
                        double weight3 = (1 - a) * (1 - b);
                        int apprX1;
                        if (apprX + 1 < image.Width)
                        {
                            apprX1 = apprX + 1;
                        }
                        else
                        {
                            apprX1 = apprX;
                        }
                        int apprY1;
                        if (apprY + 1 < image.Height)
                        {
                            apprY1 = apprY + 1;
                        }
                        else
                        {
                            apprY1 = apprY;
                        }
                        Color color0 = image.GetPixel(apprX1, apprY1);
                        Color color1 = image.GetPixel(apprX1, apprY);
                        Color color2 = image.GetPixel(apprX, apprY1);
                        Color color3 = image.GetPixel(apprX, apprY);
                        retVal.Raw[i * height + j] = (float)Math.Sqrt(color0.R * color0.R * weight0 + color1.R * color1.R * weight1 + color2.R * color2.R * weight2 + color3.R * color3.R * weight3) / 255.0F;
                        retVal.Raw[1 * width * height + i * height + j] = (float)Math.Sqrt(color0.G * color0.G * weight0 + color1.G * color1.G * weight1 + color2.G * color2.G * weight2 + color3.G * color3.G * weight3) / 255.0F;
                        retVal.Raw[2 * width * height + i * height + j] = (float)Math.Sqrt(color0.B * color0.B * weight0 + color1.B * color1.B * weight1 + color2.B * color2.B * weight2 + color3.B * color3.B * weight3) / 255.0F;
                    }
                }
            }
            else
            {
                for (int i = 0; i < width; i++)
                {
                    for (int j = 0; j < height; j++)
                    {
                        double startX = skipX + scale * i;
                        double startY = skipY + scale * j;
                        double endX = skipX + scale * (i + 1);
                        double endY = skipY + scale * (j + 1);
                        int intStartX = (int)startX;
                        int intStartY = (int)startY;
                        int intEndX = (int)Math.Ceiling(endX) - 1;
                        int intEndY = (int)Math.Ceiling(endY) - 1;
                        double red = 0.0;
                        double green = 0.0;
                        double blue = 0.0;
                        double totWeight = 0.0;
                        for (int k = (int)startX; k <= intEndX; k++)
                        {
                            double weightX = 0.0;
                            if (intStartX < k && k < intEndX)
                            {
                                weightX += 1.0;
                            }
                            if (k == intStartX)
                            {
                                weightX += 1.0 - (startX - intStartX);
                            }
                            if (k == intEndX)
                            {
                                weightX += endX - (int)endX;
                            }
                            for (int l = (int)startY; l <= intEndY; l++)
                            {
                                double weightY = 0.0;
                                if (intStartY < l && l < intEndY)
                                {
                                    weightY += 1.0;
                                }
                                if (l == intStartY)
                                {
                                    weightY = 1.0 - (startY - intStartY);
                                }
                                if (l == intEndY)
                                {
                                    weightY += endY - (int)endY;
                                }
                                if (k < image.Width && l < image.Height)
                                {
                                    double weight = weightX * weightY;
                                    Color color = image.GetPixel(k, l);
                                    red += color.R * color.R * weight;
                                    green += color.G * color.G * weight;
                                    blue += color.B * color.B * weight;
                                    totWeight += weight;
                                }

                            }
                        }
                        if (totWeight != scale * scale)
                        {

                        }
                        retVal.Raw[i * height + j] = (float)(Math.Sqrt(red / totWeight) / 255.0);
                        retVal.Raw[1 * width * height + i * height + j] = (float)(Math.Sqrt(green / totWeight) / 255.0);
                        retVal.Raw[2 * width * height + i * height + j] = (float)(Math.Sqrt(blue / totWeight) / 255.0);
                    }
                }
            }
            return retVal;
        }
        
        public Bitmap ToGreenBitmap()
        {
            Bitmap retVal = new Bitmap(this.Width, this.Height);
            for (int i = 0; i < this.Width; i++)
            {
                for (int j = 0; j < this.Height; j++)
                {
                    Color color;
                    if (this.Depth == 1)
                    {
                        int bright = (int)Math.Round(this.Raw[i * this.Height + j] * 255.0F);
                        color = Color.FromArgb(0, bright, 0);
                    }
                    else
                    {
                        color = Color.FromArgb((int)Math.Round(this.Raw[i * this.Height + j] * 255.0F), (int)Math.Round(this.Raw[this.Width * this.Height * 1 + i * this.Height + j] * 255.0F), (int)Math.Round(this.Raw[2 * this.Width * this.Height + i * this.Height + j] * 255.0F));
                    }
                    retVal.SetPixel(i, j, color);
                }
            }
            return retVal;
        }
        
        public void Export(string path, ImageFormat format)
        {
            Bitmap bmp = this.ToBitmap();
            bmp.Save(path, format);
            bmp.Dispose();
        }
        
        public void Export(string path)
        {
            if (Path.GetExtension(path) == "")
            {
                path += ".png";
            }
            this.Export(path, ImageFormat.Png);
        }
        
        public static void SaveAllLayers(string path, Image[] images, bool smart = false, bool createHtml = true)
        {
            Directory.CreateDirectory(path);
            for (int i = 0; i < images.Length; i++)
            {
                images[i].ExportLayers(path + "\\" + i, smart, false);
            }
            if (createHtml)
            {
                StreamWriter sw = new StreamWriter(path + "\\doc.html");
                sw.WriteLine("<body style=\"background-color:aqua\">");
                for (int i = 0; i < images.Length; i++)
                {
                    sw.WriteLine("<h2>" + i + "</h2>");
                    for (int j = 0; j < images[i].Depth; j++)
                    {
                        sw.WriteLine("<img src=\"" + i + "/" + j + ".png\">");
                    }
                }
                sw.WriteLine("</body>");
                sw.Close();
            }
        }
        
        public void ExportLayers(string path, bool smart = false, bool createHtml = true)
        {
            Directory.CreateDirectory(path);
            for (int i = 0; i < this.Depth; i++)
            {
                this.ToBitmap(i, smart).Save(path + "\\" + i + ".png", ImageFormat.Png);
            }
            if (createHtml)
            {
                StreamWriter sw = new StreamWriter(path + "\\doc.html");
                for (int i = 0; i < this.Depth; i++)
                {
                    sw.WriteLine("<img src=\"" + i + ".png\">");
                }
                sw.Close();
            }
        }
        
        public Bitmap ToBitmap(int layer, bool smart = false)
        {
            Bitmap retVal = new Bitmap(this.Width, this.Height);
            for (int i = 0; i < this.Width; i++)
            {
                for (int j = 0; j < this.Height; j++)
                {
                    Color color;
                    if (smart)
                    {
                        int bright = Math.Min((int)Math.Round(Math.Sqrt(this.Raw[layer * this.Width * this.Height + i * this.Height + j]) * 255.0F * 3), 255 * 3);
                        color = Color.FromArgb(Math.Min(bright, 255), Math.Max(0, Math.Min(bright - 255, 255)), Math.Max(0, Math.Min(bright - 510, 255)));
                    }
                    else
                    {
                        int bright = Math.Min((int)Math.Round(this.Raw[layer * this.Width * this.Height + i * this.Height + j] * 255.0F), 255);
                        color = Color.FromArgb(bright, bright, bright);
                    }
                    retVal.SetPixel(i, j, color);
                }
            }
            return retVal;
        }
        
        public Bitmap ToBitmap()
        {
            Bitmap retVal = new Bitmap(this.Width, this.Height);
            for (int i = 0; i < this.Width; i++)
            {
                for (int j = 0; j < this.Height; j++)
                {
                    Color color;
                    if (this.Depth == 1)
                    {
                        int bright = (int)Math.Round(this.Raw[i * this.height + j] * 255.0F);
                        color = Color.FromArgb(bright, bright, bright);
                    }
                    else
                    {
                        color = Color.FromArgb((int)Math.Round(this.Raw[i * this.Height + j] * 255.0F), (int)Math.Round(this.Raw[this.Width * this.Height + i * this.Height + j] * 255.0F), (int)Math.Round(this.Raw[2 * this.Width * this.Height + i * this.Height + j] * 255.0F));
                    }
                    retVal.SetPixel(i, j, color);
                }
            }
            return retVal;
        }
        
        public void Normalize()
        {
            for (int i = 0; i < this.Depth; i++)
            {
                float min = float.PositiveInfinity;
                float max = float.NegativeInfinity;
                for (int j = 0; j < this.Width; j++)
                {
                    for (int k = 0; k < this.Height; k++)
                    {
                        min = Math.Min(this.Raw[i * this.Width * this.Height + j * this.height + k], min);
                        max = Math.Max(this.Raw[i * this.Width * this.Height + j * this.Height + k], max);
                    }
                }
                if (max > min)
                {
                    for (int j = 0; j < this.Width; j++)
                    {
                        for (int k = 0; k < this.Height; k++)
                        {
                            this.Raw[i * this.Width * this.Height + j * this.Height + k] = (this.Raw[i * this.Width * this.Height + j * this.Height + k] - min) / (max - min);
                        }
                    }
                }
                else
                {
                    Array.Clear(this.Raw, 0, this.Raw.Length);
                }
            }
        }
        
        public void GetMinMax(out float min, out float max)
        {
            min = float.PositiveInfinity;
            max = float.NegativeInfinity;
            for (int i = 0; i < this.Depth; i++)
            {
                for (int j = 0; j < this.Width; j++)
                {
                    for (int k = 0; k < this.Height; k++)
                    {
                        min = Math.Min(this.Raw[i * this.Width * this.Height + j * this.Height + k], min);
                        max = Math.Max(this.Raw[i * this.Width * this.Height + j * this.Height + k], max);
                    }
                }
            }
        }
        
        public static void NormalizeAll(params Image[] images)
        {
            float min = float.PositiveInfinity;
            float max = float.NegativeInfinity;
            for (int i = 0; i < images.Length; i++)
            {
                images[i].GetMinMax(out float imgMin, out float imgMax);
                min = Math.Min(min, imgMin);
                max = Math.Max(max, imgMax);
            }
            for (int i = 0; i < images.Length; i++)
            {
                images[i].Normalize(min, max);
            }
        }
        
        public void Normalize(float min, float max)
        {
            for (int i = 0; i < this.Depth; i++)
            {
                if (max > min)
                {
                    for (int j = 0; j < this.Width; j++)
                    {
                        for (int k = 0; k < this.Height; k++)
                        {
                            this.Raw[i * this.Width * this.Height + j * this.Height + k] = (this.Raw[i * this.Width * this.Height + j * this.Height + k] - min) / (max - min);
                        }
                    }
                }
                else
                {
                    Array.Clear(this.Raw, 0, this.Raw.Length);
                }
            }
        }
        
        /*public void Randomize()
        {
            for (int i = 0; i < this.Depth; i++)
            {
                for (int j = 0; j < this.Width; j++)
                {
                    for (int k = 0; k < this.Height; k++)
                    {
                        this.Raw[i * this.Width * this.Height + j * this.Height + k] = RandomGenerator.GetFloat();
                    }
                }
            }
        }*/
        
        public void Clear()
        {
            Array.Clear(this.Raw, 0, this.Raw.Length);
        }
        
        public void Clear(int w, int x, int z, int depth, int width, int height)
        {
            /*for (int i = w; i < w + depth; i++)
            {
                for (int j = x; j < x + width; j++)
                {
                    for (int k = z; k < z + height; k++)
                    {
                        this.Raw[i, j, k] = 0.0F;
                    }
                }
            }*/
        }
        
        /*public void Add(Image image, float alpha = 1.0F)
        {
            int depth = Math.Min(this.Depth, image.Depth);
            int width = Math.Min(this.Width, image.Width);
            int height = Math.Min(this.Height, image.Height);
            for (int i = 0; i < depth; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    for (int k = 0; k < height; k++)
                    {
                        this.Raw[i, j, k] += image.Raw[i, j, k] * alpha;
                    }
                }
            }
        }
        
        public float MaxAt(int x, int y)
        {
            float retVal = this.Raw[0, x, y];
            for (int i = 1; i < this.Depth; i++)
            {
                retVal = Math.Max(retVal, this.Raw[0, x, y]);
            }
            return retVal;
        }
        
        public Image MnistAdapt(int outputSize = 28, int outputPad = 4, int w = 0)
        {
            Image retVal = new Image(1, outputSize, outputSize);

            float meanX = 0;
            float meanY = 0;
            float sumPixels = 0.0F;
            for (var i = 0; i < this.Width; i++)
            {
                for (var j = 0; j < this.Height; j++)
                {
                    sumPixels += this.Raw[w, i, j];
                    meanY += j * this.Raw[w, i, j];
                    meanX += i * this.Raw[w, i, j];
                }
            }
            meanX /= sumPixels;
            meanY /= sumPixels;
            int centerX = (int)Math.Round(meanX);
            int centerY = (int)Math.Round(meanY);

            var threshold = 0.01F;
            int xMin = this.Width;
            int xMax = -1;
            int yMin = this.Height;
            int yMax = -1;
            for (int i = 0; i < this.Width; i++)
            {
                for (int j = 0; j < this.Height; j++)
                {
                    if (this.Raw[w, i, j] > threshold)
                    {
                        xMin = Math.Min(i, xMin);
                        xMax = Math.Max(i, xMax);
                        yMin = Math.Min(j, yMin);
                        yMax = Math.Max(j, yMax);
                    }
                }
            }

            int radius = Math.Max(Math.Max(centerX - xMin, centerY - yMin), Math.Max(xMax - centerX, yMax - centerY));
            int x = Math.Max(centerX - radius, 0);
            int y = Math.Max(centerY - radius, 0);
            int contentSize = outputSize - outputPad * 2;

            retVal.FromImageScaled(this, w, x, y, 0, outputPad, outputPad, 1, Math.Min(this.Width - x, radius * 2), Math.Min(this.Height - y, radius * 2), contentSize, contentSize);
            retVal.Normalize();
            return retVal;
        }*/

        private static void FromMnistFunc(Stream stream, Image[] array, bool normalize, int count, int skip, bool emnist = false)
        {
            int width = (stream.ReadByte() << 24) + (stream.ReadByte() << 16) + (stream.ReadByte() << 8) + stream.ReadByte();
            int height = (stream.ReadByte() << 24) + (stream.ReadByte() << 16) + (stream.ReadByte() << 8) + stream.ReadByte();
            stream.Position += skip * width * height;
            for (int i = 0; i < count; i++)
            {
                array[i] = new Image(1, width, height, true);
                if (emnist)
                {
                    for (int j = 0; j < width; j++)
                    {
                        for (int k = 0; k < height; k++)
                        {
                            array[i].SetValue(0, j, k, stream.ReadByte() / 255.0F);
                        }
                    }
                }
                else
                {
                    for (int j = 0; j < height; j++)
                    {
                        for (int k = 0; k < width; k++)
                        {
                            array[i].SetValue(0, k, j, stream.ReadByte() / 255.0F);
                        }
                    }
                }
                if (normalize)
                {
                    array[i].Normalize();
                }
            }
        }
        
        public static Image[] FromMnist(Stream stream, bool normalize = true, int maxCount = 0, int skip = 0, bool emnist = false)
        {
            stream.Position = 4;
            int count = (stream.ReadByte() << 24) + (stream.ReadByte() << 16) + (stream.ReadByte() << 8) + stream.ReadByte();
            if (maxCount > 0 && count > maxCount)
            {
                count = maxCount;
            }
            Image[] retVal = new Image[count];
            FromMnistFunc(stream, retVal, normalize, count, skip, emnist);
            return retVal;
        }
        
        public static void FromMnist(Stream stream, Image[] array, bool normalize = true, int maxCount = 0, int skip = 0, bool emnist = false)
        {
            stream.Position = 4;
            int count = (stream.ReadByte() << 24) + (stream.ReadByte() << 16) + (stream.ReadByte() << 8) + stream.ReadByte();
            if (maxCount > 0 && count > maxCount || count > array.Length)
            {
                count = Math.Min(maxCount, array.Length);
            }
            FromMnistFunc(stream, array, normalize, count, skip, emnist);
        }

        private static void FromMnistLabelsFunc(Stream stream, double[][] array, bool smart, int count, int skip, int labels)
        {
            stream.Position += skip;
            if (smart)
            {
                double[][] vectorLables = new double[labels][];
                for (int i = 0; i < labels; i++)
                {
                    vectorLables[i] = new double[labels];
                    vectorLables[i][i] = 1.0;
                }
                for (int i = 0; i < count; i++)
                {
                    array[i] = vectorLables[stream.ReadByte()];
                }
            }
            else
            {
                for (int i = 0; i < count; i++)
                {
                    array[i] = new double[labels];
                    array[i][stream.ReadByte()] = 1.0;
                }
            }
        }
        
        public static double[][] FromMnistLabels(Stream stream, bool smart = true, int maxCount = 0, int skip = 0, int labels = 10)
        {
            stream.Position = 4;
            int count = (stream.ReadByte() << 24) + (stream.ReadByte() << 16) + (stream.ReadByte() << 8) + stream.ReadByte();
            if (maxCount > 0 && count > maxCount)
            {
                count = maxCount;
            }
            double[][] retVal = new double[count][];
            FromMnistLabelsFunc(stream, retVal, smart, count, skip, labels);
            return retVal;
        }
        
        public static void FromMnistLabels(Stream stream, double[][] array, bool smart = true, int maxCount = 0, int skip = 0, int labels = 10)
        {
            stream.Position = 4;
            int count = (stream.ReadByte() << 24) + (stream.ReadByte() << 16) + (stream.ReadByte() << 8) + stream.ReadByte();
            if (maxCount > 0 && count > maxCount || count > array.Length)
            {
                count = Math.Min(array.Length, maxCount);
            }
            stream.Position += skip;
            FromMnistLabelsFunc(stream, array, smart, count, skip, labels);
        }
        
        public static Image[] FromMnist(string fileName, bool normalize = true, int maxCount = 0, int skip = 0, bool emnist = false)
        {
            FileStream fs = new FileStream(fileName, FileMode.Open);
            Image[] retVal = Image.FromMnist(fs, normalize, maxCount, skip, emnist);
            fs.Close();
            return retVal;
        }
        
        public static double[][] FromMnistLabels(string fileName, bool smart = true, int maxCount = 0, int skip = 0, int labels = 10)
        {
            FileStream fs = new FileStream(fileName, FileMode.Open);
            double[][] retVal = Image.FromMnistLabels(fs, smart, maxCount, skip, labels);
            fs.Close();
            return retVal;
        }
        
        public static void FromMnist(string fileName, Image[] array, bool normalize = true, int maxCount = 0, int skip = 0, bool emnist = false)
        {
            FileStream fs = new FileStream(fileName, FileMode.Open);
            Image.FromMnist(fs, array, normalize, maxCount, skip, emnist);
            fs.Close();
        }
        
        public static void FromMnistLabels(string fileName, double[][] array, bool smart = true, int maxCount = 0, int skip = 0, int labels = 10)
        {
            FileStream fs = new FileStream(fileName, FileMode.Open);
            Image.FromMnistLabels(fs, array, smart, maxCount, skip, labels);
            fs.Close();
        }
    }
}
