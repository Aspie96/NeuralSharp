﻿/*
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

namespace NeuralSharp.Convolutional
{
    /// <summary>Represents an image.</summary>
    public class Image
    {
        private int depth;
        private int width;
        private int height;
        private float[] data;   // [depth, width, height]
        
        /// <summary>Creates an instance of the <code>Image</code> class.</summary>
        /// <param name="depth">The depth of the image.</param>
        /// <param name="width">The width of the image.</param>
        /// <param name="height">The height of the image.</param>
        /// <param name="c">Wether to use an array to encode the image.</param>
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
        
        /// <summary>The depth of the image.</summary>
        public int Depth
        {
            get { return this.depth; }
        }
        
        /// <summary>The width of the image.</summary>
        public int Width
        {
            get { return this.width; }
        }
        
        /// <summary>The height of the image.</summary>
        public int Height
        {
            get { return this.height; }
        }
        
        /// <summary>The data contained by the image.</summary>
        public float[] Raw
        {
            get { return this.data; }
        }
        
        /// <summary>Gets a value contained in the image, given its coordinates.</summary>
        /// <param name="w">The W coordinate of the value.</param>
        /// <param name="x">The X coordinate of the value.</param>
        /// <param name="y">The Y coordinate of the value.</param>
        /// <returns></returns>
        public float GetValue(int w, int x, int y)
        {
            return this.Raw[w * this.width * this.height + x * this.height + y];
        }
        
        /// <summary>Sets a value in the image.</summary>
        /// <param name="w">The W coordinate of the value.</param>
        /// <param name="x">The X coordinate of the value.</param>
        /// <param name="y">The Y coordinate of the value.</param>
        /// <param name="value">The value.</param>
        public void SetValue(int w, int x, int y, float value)
        {
            this.Raw[w * this.width * this.height + x * this.height + y] = value;
        }
        
        /// <summary>Copies the content of the image into an array.</summary>
        /// <param name="depth">The depth of the portion of the image to be copied.</param>
        /// <param name="width">The width of the portion of the image to be copied.</param>
        /// <param name="height">The height of the portion of the image to be copied.</param>
        /// <param name="array">The array to be copied the image into.</param>
        /// <param name="arraySkip">The index of the first position of the array to be used.</param>
        public void ToArray(int depth, int width, int height, float[] array, int arraySkip = 0)
        {
            Backbone.ImageToArray(this.Raw, this.Depth, this.Width, this.Height, depth, width, height, array, arraySkip);
        }

        /// <summary>Copies the content of the image into an array.</summary>
        /// <param name="array">The array to be copied into.</param>
        /// <param name="arraySkip">The index of the first position of the array to be used.</param>
        public void ToArray(float[] array, int arraySkip = 0)
        {
            this.ToArray(this.Depth, this.Width, this.Height, array, arraySkip);
        }

        /// <summary>Copies the data from the given image.</summary>
        /// <param name="image">The image to be copied from.</param>
        public void FromImage(Image image)
        {
            this.FromImage(image, Math.Min(this.Depth, image.Depth), Math.Min(this.Width, image.Width), Math.Min(this.Height, image.Height));
        }
        
        /// <summary>Copies the data from the given image.</summary>
        /// <param name="image">The image to be copied from.</param>
        /// <param name="sourceW">The W coordinate of the first value of the given image to be used.</param>
        /// <param name="sourceX">The X coordinate of the first value of the given image to be used.</param>
        /// <param name="sourceY">The Y coordinate of the first value of the given image to be used.</param>
        /// <param name="thisW">The W coordinate of the first value to be copied into.</param>
        /// <param name="thisX">The X coordinate of the first value to be copied into.</param>
        /// <param name="thisY">The Y coordinate of the first value to be copied into.</param>
        /// <param name="depth">The depth of the portion to be copied.</param>
        /// <param name="width">The width of the portion to be copied.</param>
        /// <param name="height">The height of the portion to be copied.</param>
        public void FromImage(Image image, int sourceW, int sourceX, int sourceY, int thisW, int thisX, int thisY, int depth, int width, int height)
        {
            Backbone.ImageToImage(this.Raw, this.Depth, this.Width, this.Height, image.Raw, image.Depth, image.Width, image.Height, sourceW, sourceX, sourceY, thisW, thisX, thisY, depth, width, height);
        }
        
        /// <summary>Copies data from an array.</summary>
        /// <param name="array">The array to be copied from.</param>
        /// <param name="skip">The index of the first entry of the array to be used.</param>
        /// <param name="w">The W coordinate of the first value to be copied into.</param>
        /// <param name="x">The X coordinate of the first value to be copied into.</param>
        /// <param name="y">The Y coordinate of the first value to be copied into.</param>
        /// <param name="depth">The depth of the portion to be copied into.</param>
        /// <param name="width">The width of the portion to be copied into.</param>
        /// <param name="height">The height of the portion to be copied into.</param>
        public void FromArray(float[] array, int skip, int w, int x, int y, int depth, int width, int height)
        {
            Backbone.ArrayToImage(this.Raw, this.Depth, this.Width, this.Height, array, skip, w, x, y, depth, width, height);
        }

        /// <summary>Copies data from an array.</summary>
        /// <param name="array">The array to be copied from.</param>
        /// <param name="skip">The index of the first entry of the array to be used.</param>
        public void FromArray(float[] array, int skip = 0)
        {
            this.FromArray(array, skip, 0, 0, 0, this.Depth, this.Width, this.Height);
        }

        /// <summary>Copies the data from the given image.</summary>
        /// <param name="image">The image to be copied from.</param>
        /// <param name="depth">The depth of the portion to be copied.</param>
        /// <param name="width">The width of the portion to be copied.</param>
        /// <param name="height">The height of the portion to be copied.</param>
        public void FromImage(Image image, int depth, int width, int height)
        {
            this.FromImage(image, 0, 0, 0, 0, 0, 0, depth, width, height);
        }
        
        /// <summary>Copies data from the given image, scaling it.</summary>
        /// <param name="image">The image to be copied from.</param>
        /// <param name="sourceW">The W coordinate of the first value of the given image to be used.</param>
        /// <param name="sourceX">The X coordinate of the first value of the given image to be used.</param>
        /// <param name="sourceY">The Y coordinate of the first value of the given image to be used.</param>
        /// <param name="thisW">The W coordinate of the first value to be copied into.</param>
        /// <param name="thisX">The X coordinate of the first value to be copied into.</param>
        /// <param name="thisY">The Y coordinate of the first value to be copied into.</param>
        /// <param name="depth">The depth of the portion to be copied.</param>
        /// <param name="sourceWidth">The width of the portion to be copied.</param>
        /// <param name="sourceHeight">The height of the portion to be copied.</param>
        /// <param name="thisWidth">The resulting width of the portion to be copied.</param>
        /// <param name="thisHeight">The resulting height of the portion to be copied.</param>
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
                    float totalWeight = 0.0F;
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
        
        /// <summary>Copies data from the given image, scaling it.</summary>
        /// <param name="image">The image to be copied from.</param>
        /// <param name="sourceX">The X coordinate of the first value to be copied from.</param>
        /// <param name="sourceY">The X coordinate of the first value to be copied from.</param>
        /// <param name="thisX">The X coordinate of the firt value to be copied into.</param>
        /// <param name="thisY">The Y coordinate of the first value to be copied into.</param>
        /// <param name="sourceWidth">The width of the portion to be copied.</param>
        /// <param name="sourceHeight">The height of the portion to be copied.</param>
        /// <param name="thisWidth">The resulting width of the portion to be copied.</param>
        /// <param name="thisHeight">The resulting height of the portion to be copied.</param>
        public void FromImageScaled(Image image, int sourceX, int sourceY, int thisX, int thisY, int sourceWidth, int sourceHeight, int thisWidth, int thisHeight)
        {
            this.FromImageScaled(image, 0, sourceX, sourceY, 0, thisX, thisY, Math.Min(this.Depth, image.Depth), sourceWidth, sourceHeight, thisWidth, thisHeight);
        }
        
        /// <summary>Copies the data from the given image, scaling it.</summary>
        /// <param name="image">The image to be copied from.</param>
        public void FromImageScaled(Image image)
        {
            this.FromImageScaled(image, 0, 0, 0, 0, image.Width, image.height, this.Width, this.Height);
        }
        
        /// <summary>Copies data from the given image.</summary>
        /// <param name="image">The image to be copied from.</param>
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
                        this.Raw[i * this.Height + j] = (float)(float)Math.Sqrt(pixel.R * pixel.R * 0.241F + pixel.G * pixel.G * 0.691F + pixel.B * pixel.B * 0.068F) / 255.0F;
                    }
                }
            }
        }
        
        /// <summary>Copies data from the given image.</summary>
        /// <param name="path">The image to be copied from.</param>
        public void FromBitmap(string path)
        {
            this.FromBitmap(new Bitmap(path));
        }

        /// <summary>Returns the central scaled portion from the given image.</summary>
        /// <param name="path">The path of the image to be copied from.</param>
        /// <param name="width">The width of the image.</param>
        /// <param name="height">The height of the image.</param>
        /// <returns>The created image.</returns>
        public static Image AdaptFromBitmap(string path, int width, int height)
        {
            Bitmap image = new Bitmap(path);
            float ratio = (float)width / height;
            float originalRatio = (float)image.Width / (float)image.Height;
            float scale;
            float skipX;
            float skipY;
            if (originalRatio > ratio)
            {
                skipX = image.Width * (1 - ratio / originalRatio) / 2.0F;
                skipY = 0.0F;
                scale = (float)image.Height / height;
            }
            else
            {
                skipY = image.Height * (1 - originalRatio / ratio) / 2.0F;
                skipX = 0.0F;
                scale = (float)image.Width / width;
            }
            Image retVal = new Image(3, width, height);
            if (scale <= 1.0F || true)
            {
                for (int i = 0; i < width; i++)
                {
                    for (int j = 0; j < height; j++)
                    {
                        float x = skipX + scale * i;
                        float y = skipY + scale * j;
                        int apprX = (int)x;
                        int apprY = (int)y;
                        float a = x - apprX;
                        float b = y - apprY;
                        float weight0 = a * b;
                        float weight1 = a * (1 - b);
                        float weight2 = (1 - a) * b;
                        float weight3 = (1 - a) * (1 - b);
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
                        retVal.Raw[i * height + j] = (float)(float)Math.Sqrt(color0.R * color0.R * weight0 + color1.R * color1.R * weight1 + color2.R * color2.R * weight2 + color3.R * color3.R * weight3) / 255.0F;
                        retVal.Raw[1 * width * height + i * height + j] = (float)(float)Math.Sqrt(color0.G * color0.G * weight0 + color1.G * color1.G * weight1 + color2.G * color2.G * weight2 + color3.G * color3.G * weight3) / 255.0F;
                        retVal.Raw[2 * width * height + i * height + j] = (float)(float)Math.Sqrt(color0.B * color0.B * weight0 + color1.B * color1.B * weight1 + color2.B * color2.B * weight2 + color3.B * color3.B * weight3) / 255.0F;
                    }
                }
            }
            else
            {
                for (int i = 0; i < width; i++)
                {
                    for (int j = 0; j < height; j++)
                    {
                        float startX = skipX + scale * i;
                        float startY = skipY + scale * j;
                        float endX = skipX + scale * (i + 1);
                        float endY = skipY + scale * (j + 1);
                        int intStartX = (int)startX;
                        int intStartY = (int)startY;
                        int intEndX = (int)Math.Ceiling(endX) - 1;
                        int intEndY = (int)Math.Ceiling(endY) - 1;
                        float red = 0.0F;
                        float green = 0.0F;
                        float blue = 0.0F;
                        float totWeight = 0.0F;
                        for (int k = (int)startX; k <= intEndX; k++)
                        {
                            float weightX = 0.0F;
                            if (intStartX < k && k < intEndX)
                            {
                                weightX += 1.0F;
                            }
                            if (k == intStartX)
                            {
                                weightX += 1.0F - (startX - intStartX);
                            }
                            if (k == intEndX)
                            {
                                weightX += endX - (int)endX;
                            }
                            for (int l = (int)startY; l <= intEndY; l++)
                            {
                                float weightY = 0.0F;
                                if (intStartY < l && l < intEndY)
                                {
                                    weightY += 1.0F;
                                }
                                if (l == intStartY)
                                {
                                    weightY = 1.0F - (startY - intStartY);
                                }
                                if (l == intEndY)
                                {
                                    weightY += endY - (int)endY;
                                }
                                if (k < image.Width && l < image.Height)
                                {
                                    float weight = weightX * weightY;
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
                        retVal.Raw[i * height + j] = (float)((float)Math.Sqrt(red / totWeight) / 255.0);
                        retVal.Raw[1 * width * height + i * height + j] = (float)((float)Math.Sqrt(green / totWeight) / 255.0);
                        retVal.Raw[2 * width * height + i * height + j] = (float)((float)Math.Sqrt(blue / totWeight) / 255.0);
                    }
                }
            }
            return retVal;
        }
        
        /// <summary>Creates a green and black version of the image.</summary>
        /// <returns>The created image.</returns>
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
        
        /// <summary>Saves the image to a file.</summary>
        /// <param name="path">The file to be exported into.</param>
        /// <param name="format">The format of the output image.</param>
        public void Export(string path, ImageFormat format)
        {
            Bitmap bmp = this.ToBitmap();
            bmp.Save(path, format);
            bmp.Dispose();
        }
        
        /// <summary>Saves the image to a file.</summary>
        /// <param name="path">The file to be exported into.</param>
        public void Export(string path)
        {
            if (Path.GetExtension(path) == "")
            {
                path += ".png";
            }
            this.Export(path, ImageFormat.Png);
        }
        
        /// <summary>Exports the image.</summary>
        /// <param name="w">The W coordinate of the portion to be exported.</param>
        /// <param name="smart"><code>true</code> if the resulting image is not to be black and white, <code>true</code> otherwise.</param>
        /// <returns>The created image.</returns>
        public Bitmap ToBitmap(int w, bool smart = false)
        {
            Bitmap retVal = new Bitmap(this.Width, this.Height);
            for (int i = 0; i < this.Width; i++)
            {
                for (int j = 0; j < this.Height; j++)
                {
                    Color color;
                    if (smart)
                    {
                        int bright = Math.Min((int)Math.Round((float)Math.Sqrt(this.Raw[w * this.Width * this.Height + i * this.Height + j]) * 255.0F * 3), 255 * 3);
                        color = Color.FromArgb(Math.Min(bright, 255), Math.Max(0, Math.Min(bright - 255, 255)), Math.Max(0, Math.Min(bright - 510, 255)));
                    }
                    else
                    {
                        int bright = Math.Min((int)Math.Round(this.Raw[w * this.Width * this.Height + i * this.Height + j] * 255.0F), 255);
                        color = Color.FromArgb(bright, bright, bright);
                    }
                    retVal.SetPixel(i, j, color);
                }
            }
            return retVal;
        }
        
        /// <summary>Exports the image.</summary>
        /// <returns>The created image.</returns>
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
        
        /// <summary>Normalizes the values in the image.</summary>
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

        /// <summary>Gets images from the MNIST format.</summary>
        /// <param name="stream">The stream to be read.</param>
        /// <param name="normalize">Whether to normalize the images.</param>
        /// <param name="maxCount">The maximum amount of images to be read.</param>
        /// <param name="skip">The index of the first entry of the dataset to be read.</param>
        /// <param name="emnist">Whether it is the EMNIST format.</param>
        /// <returns>The read images.</returns>
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

        /// <summary>Gets images from the MNIST dataset.</summary>
        /// <param name="stream">The stream to be read.</param>
        /// <param name="array">The array to be written the images into.</param>
        /// <param name="normalize">Whether to normalize the images.</param>
        /// <param name="maxCount">The maximum amount of images to be read.</param>
        /// <param name="skip">The index of the first image of the dataset to be read.</param>
        /// <param name="emnist">Whether it is the EMNIST format.</param>
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

        private static void FromMnistLabelsFunc(Stream stream, float[][] array, bool smart, int count, int skip, int labels)
        {
            stream.Position += skip;
            if (smart)
            {
                float[][] vectorLables = new float[labels][];
                for (int i = 0; i < labels; i++)
                {
                    vectorLables[i] = new float[labels];
                    vectorLables[i][i] = 1.0F;
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
                    array[i] = new float[labels];
                    array[i][stream.ReadByte()] = 1.0F;
                }
            }
        }

        /// <summary>Reads labels form the MNIST format.</summary>
        /// <param name="stream">The stream to be read the labels from.</param>
        /// <param name="smart">Whether to save memory.</param>
        /// <param name="maxCount">The maximum amount of labels to be read.</param>
        /// <param name="skip">The index of the first label to read.</param>
        /// <param name="labels">The amount of labels.</param>
        /// <returns>The read labels.</returns>
        public static float[][] FromMnistLabels(Stream stream, bool smart = true, int maxCount = 0, int skip = 0, int labels = 10)
        {
            stream.Position = 4;
            int count = (stream.ReadByte() << 24) + (stream.ReadByte() << 16) + (stream.ReadByte() << 8) + stream.ReadByte();
            if (maxCount > 0 && count > maxCount)
            {
                count = maxCount;
            }
            float[][] retVal = new float[count][];
            FromMnistLabelsFunc(stream, retVal, smart, count, skip, labels);
            return retVal;
        }

        /// <summary>Reads labels from the MNIST format.</summary>
        /// <param name="stream">The stream to be read the labels from.</param>
        /// <param name="array">The array to be written the labels into.</param>
        /// <param name="smart">Whether to save space.</param>
        /// <param name="maxCount">The maximum amount of read labels.</param>
        /// <param name="skip">The index of the first label to be read.</param>
        /// <param name="labels">The amount of labels.</param>
        public static void FromMnistLabels(Stream stream, float[][] array, bool smart = true, int maxCount = 0, int skip = 0, int labels = 10)
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

        /// <summary>Reads images from the MNIST dataset.</summary>
        /// <param name="fileName">The path of the dataset.</param>
        /// <param name="normalize">Whether to normalize the images.</param>
        /// <param name="maxCount">The maximum amount of images to be read.</param>
        /// <param name="skip">The index of the first entry to be read.</param>
        /// <param name="emnist">Whether it's the EMNIST dataset.</param>
        /// <returns>The read images.</returns>
        public static Image[] FromMnist(string fileName, bool normalize = true, int maxCount = 0, int skip = 0, bool emnist = false)
        {
            FileStream fs = new FileStream(fileName, FileMode.Open);
            Image[] retVal = Image.FromMnist(fs, normalize, maxCount, skip, emnist);
            fs.Close();
            return retVal;
        }

        /// <summary>Reads labels from the MNIST dataset.</summary>
        /// <param name="fileName">The path of the dataset.</param>
        /// <param name="smart">Whether to save space.</param>
        /// <param name="maxCount">The maximum amount of images to be read.</param>
        /// <param name="skip">The index of the first label to be read.</param>
        /// <param name="labels">The amount of labels.</param>
        /// <returns>The read labels.</returns>
        public static float[][] FromMnistLabels(string fileName, bool smart = true, int maxCount = 0, int skip = 0, int labels = 10)
        {
            FileStream fs = new FileStream(fileName, FileMode.Open);
            float[][] retVal = Image.FromMnistLabels(fs, smart, maxCount, skip, labels);
            fs.Close();
            return retVal;
        }

        /// <summary>Reads images from the MNIST dataset.</summary>
        /// <param name="fileName">The path of the dataset.</param>
        /// <param name="array">The array to be written the images into.</param>
        /// <param name="normalize">Whether to normalize the images.</param>
        /// <param name="maxCount">The maximum amount of images to be read.</param>
        /// <param name="skip">The index of the first image to read.</param>
        /// <param name="emnist">Whether it's the EMNIST dataset.</param>
        public static void FromMnist(string fileName, Image[] array, bool normalize = true, int maxCount = 0, int skip = 0, bool emnist = false)
        {
            FileStream fs = new FileStream(fileName, FileMode.Open);
            Image.FromMnist(fs, array, normalize, maxCount, skip, emnist);
            fs.Close();
        }
    }
}
