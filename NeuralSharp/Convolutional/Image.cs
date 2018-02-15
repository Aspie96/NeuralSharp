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
using System.Drawing;
using System.IO;
using System.Drawing.Imaging;

namespace NeuralNetwork.Convolutional
{
    /// <summary>Represents an image.</summary>
    public class Image
    {
        private int depth;
        private int width;
        private int height;
        private float[,,] data;   // [depth, width, height]

        /// <summary>Creates a new instance of the <code>Imge</code> class.</summary>
        /// <param name="depth">The depth of the image.</param>
        /// <param name="width">The width of the image.</param>
        /// <param name="height">The height of the image.</param>
        public Image(int depth, int width, int height)
        {
            this.depth = depth;
            this.width = width;
            this.height = height;
            this.data = new float[depth, width, height];
        }

        /// <summary>The depth of this image.</summary>
        public int Depth
        {
            get { return this.depth; }
        }

        /// <summary>The width of this image.</summary>
        public int Width
        {
            get { return this.width; }
        }

        /// <summary>The height of this image.</summary>
        public int Height
        {
            get { return this.height; }
        }

        /// <summary>The data contained within this image.</summary>
        public float[,,] Raw
        {
            get { return this.data; }
        }

        /// <summary>Returns a value within this image.</summary>
        /// <param name="w">The W coordinate of the value (depth).</param>
        /// <param name="x">The X coordinate of the value (width).</param>
        /// <param name="y">The Y coordinate of the value (height).</param>
        /// <returns>The value at the given position.</returns>
        public float GetValue(int w, int x, int y)
        {
            return this.Raw[w, x, y];
        }

        /// <summary>Sets a value in this image.</summary>
        /// <param name="w">The W coordinate of the value (depth).</param>
        /// <param name="x">The X coordinate of the value (width).</param>
        /// <param name="y">The Y coordinate of the value (height).</param>
        /// <param name="value">Value to be set.</param>
        public void SetValue(int w, int x, int y, float value)
        {
            this.Raw[w, x, y] = value;
        }

        /// <summary>Copies the content of an array into this image.</summary>
        /// <param name="array">The array to be copied from.</param>
        /// <param name="depth">The depth of the section of the image to be copied into.</param>
        /// <param name="width">The width of the section of the image to be copied into.</param>
        /// <param name="height">The height of the section of the image to be copied into.</param>
        public void FromArray(double[] array, int depth, int width, int height)
        {
            int index = 0;
            for (int i = 0; i < depth; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    for (int k = 0; k < height; k++)
                    {
                        this.Raw[i, j, k] = (float)array[index++];
                    }
                }
            }
        }

        /// <summary>Copies the content of this image into an array.</summary>
        /// <param name="array">The array to be copied into.</param>
        /// <param name="depth">The depth of the section of the image to be copied.</param>
        /// <param name="width">The width of the section of the image to be copied.</param>
        /// <param name="height">The height of the section of the image to be copied.</param>
        public void ToArray(double[] array, int depth, int width, int height)
        {
            int index = 0;
            for (int i = 0; i < depth; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    for (int k = 0; k < height; k++)
                    {
                        array[index++] = this.Raw[i, j, k];
                    }
                }
            }
        }

        /// <summary>Copies another image into this.</summary>
        /// <param name="image">The image to be copied from.</param>
        public void FromImage(Image image)
        {
            this.FromImage(image, Math.Min(this.Depth, image.Depth), Math.Min(this.Width, image.Width), Math.Min(this.Height, image.Height));
        }

        /// <summary>Copies another image into this.</summary>
        /// <param name="image">The image to be copied from.</param>
        /// <param name="sourceW">The lowest W coordinate of the section of the image to be copied from.</param>
        /// <param name="sourceX">The lowest X coordinate of the section of the image to be copied from.</param>
        /// <param name="sourceY">The lowest Y coordinate of the section of the image to be copied from.</param>
        /// <param name="thisW">The lowest W coordinate of the section of this image to be copied into.</param>
        /// <param name="thisX">The lowest X coordinate of the section of this image to be copied into.</param>
        /// <param name="thisY">The lowest Y coordinate of the section of this image to be copied into.</param>
        /// <param name="depth">The depth of the section of image to be copied.</param>
        /// <param name="width">The width of the section of image to be copied.</param>
        /// <param name="height">The height of the section of image to be copied.</param>
        public void FromImage(Image image, int sourceW, int sourceX, int sourceY, int thisW, int thisX, int thisY, int depth, int width, int height)
        {
            for (int i = 0; i < depth; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    for (int k = 0; k < height; k++)
                    {
                        this.Raw[thisW + i, thisX + j, thisY + k] = image.Raw[sourceW + i, sourceX + j, sourceY + k];
                    }
                }
            }
        }

        /// <summary>Copies the content of an array into this image.</summary>
        /// <param name="array">The array to be copied from.</param>
        /// <param name="sourceI">The first index of the array to be copied from.</param>
        /// <param name="thisW">The lowest W coordinate of the section of this image to be copied into.</param>
        /// <param name="thisX">The lowest X coordinate of the section of this image ot be copied into.</param>
        /// <param name="thisY">The lowest Y coordiante of the section of this image to be copied into.</param>
        /// <param name="depth">The depth of the section of this image to be copied into.</param>
        /// <param name="width">The width of the section of this image to be copied into.</param>
        /// <param name="height">The height of the section of this image to be copied into.</param>
        public void FromArray(float[] array, int sourceI, int thisW, int thisX, int thisY, int depth, int width, int height)
        {
            int index = sourceI;
            for (int i = 0; i < depth; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    for (int k = 0; k < height; k++)
                    {
                        this.Raw[thisW + i, thisX + j, thisY + k] = array[index++];
                    }
                }
            }
        }

        /// <summary>Copies the content of another image into this.</summary>
        /// <param name="image">The image to be copied from.</param>
        /// <param name="depth">The depth of the section of image to be copied.</param>
        /// <param name="width">The widht of the section of image to be copied.</param>
        /// <param name="height">The height of the section of image to be copied.</param>
        public void FromImage(Image image, int depth, int width, int height)
        {
            this.FromImage(image, 0, 0, 0, 0, 0, 0, depth, width, height);
        }

        /// <summary>Copies a scaled version (along the X and Y axis) of another image into this.</summary>
        /// <param name="image">The image to be scaled and copied from</param>
        /// <param name="sourceW">The lowest W coordinate of the section of image to be copied from.</param>
        /// <param name="sourceX">The lowest X coordinate of the section of image to be copied from.</param>
        /// <param name="sourceY">The lowest Y coordinate of the section of image ot be copied from.</param>
        /// <param name="thisW">The lowest X coordinate of this image to be copied into.</param>
        /// <param name="thisX">The lowest X coordinate of this image to be copied into.</param>
        /// <param name="thisY">The lowest Y coordinate of this image to be copied into.</param>
        /// <param name="depth">The lepth of the section of image to be copied.</param>
        /// <param name="sourceWidth">The initial width of the section of image to be copied.</param>
        /// <param name="sourceHeight">The initial height of the section of image to be copied.</param>
        /// <param name="thisWidth">The scaled width of the copied and scaled section of the image.</param>
        /// <param name="thisHeight">The scaled height of the copied and scaled section of the image.</param>
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
                        this.Raw[k + thisW, i + thisX, j + thisY] = 0.0F;
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
                                this.Raw[m + thisW, i + thisX, j + thisY] += image.Raw[m + sourceW, k + sourceX, l + sourceY] * weight;
                                totalWeight += weight;
                            }
                        }
                    }
                }
            }
        }

        /// <summary>Scales and copies the content of another image into this.</summary>
        /// <param name="image">The image to be scaled and copied from.</param>
        /// <param name="sourceX">The lowest X coordinate of the section of the image to be copied from.</param>
        /// <param name="sourceY">The lowest Y coordinate of the section of the image to be copied from.</param>
        /// <param name="thisX">The lowest X coordiante of the section of this image to be copied into.</param>
        /// <param name="thisY">The lowest X coordinate of the section of this image to be copied into.</param>
        /// <param name="sourceWidth">The original widht of the section of image to be copied from.</param>
        /// <param name="sourceHeight">The original height of the section of image to be copied from.</param>
        /// <param name="thisWidth">The width of the copied and scaled section of the image.</param>
        /// <param name="thisHeight">The height of the copied and scaled secthion of the image.</param>
        public void FromImageScaled(Image image, int sourceX, int sourceY, int thisX, int thisY, int sourceWidth, int sourceHeight, int thisWidth, int thisHeight)
        {
            this.FromImageScaled(image, 0, sourceX, sourceY, 0, thisX, thisY, Math.Min(this.Depth, image.Depth), sourceWidth, sourceHeight, thisWidth, thisHeight);
        }

        /// <summary>Copies the content of another image into this, scaling it to adapt it to the size of this image.</summary>
        /// <param name="image">The image to be copied from.</param>
        public void FromImageScaled(Image image)
        {
            this.FromImageScaled(image, 0, 0, 0, 0, image.Width, image.height, this.Width, this.Height);
        }

        /// <summary>Copies the content of a bitmap into this.</summary>
        /// <param name="image">The bitmap to be copied from.</param>
        /// <param name="normalize"><code>true</code> if each value is to be normalized between <code>0</code> and <code>1</code>, <code>false</code> otherwise.</param>
        public void FromBitmap(Bitmap image, bool normalize = false)
        {
            if (this.Depth == 3)
            {
                for (int i = 0; i < this.Width; i++)
                {
                    for (int j = 0; j < this.Height; j++)
                    {
                        Color pixel = image.GetPixel(i, j);
                        this.Raw[0, i, j] = pixel.R / 255.0F;
                        this.Raw[1, i, j] = pixel.G / 255.0F;
                        this.Raw[2, i, j] = pixel.B / 255.0F;
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
                        this.Raw[0, i, j] = (float)Math.Sqrt(pixel.R * pixel.R * 0.241F + pixel.G * pixel.G * 0.691F + pixel.B * pixel.B * 0.068F) / 255.0F;
                    }
                }
            }
            if (normalize)
            {
                this.Normalize();
            }
        }

        /// <summary>Imports the content of a saved image into this.</summary>
        /// <param name="path">The name of the file to be imported.</param>
        public void FromBitmap(string path)
        {
            this.FromBitmap(new Bitmap(path));
        }

        /// <summary>Converts this image into a bitmap, using the color green.</summary>
        /// <returns>The generated bitmap.</returns>
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
                        int bright = (int)Math.Round(this.Raw[0, i, j] * 255.0F);
                        color = Color.FromArgb(0, bright, 0);
                    }
                    else
                    {
                        color = Color.FromArgb((int)Math.Round(this.Raw[0, i, j] * 255.0F), (int)Math.Round(this.Raw[1, i, j] * 255.0F), (int)Math.Round(this.Raw[2, i, j] * 255.0F));
                    }
                    retVal.SetPixel(i, j, color);
                }
            }
            return retVal;
        }

        /// <summary>Exports this image to a file.</summary>
        /// <param name="path">The name of the file to be created.</param>
        /// <param name="format">The file format to be used.</param>
        public void Export(string path, ImageFormat format)
        {
            Bitmap bmp = this.ToBitmap();
            bmp.Save(path, format);
            bmp.Dispose();
        }

        /// <summary>Exports this image as a PNG file.</summary>
        /// <param name="path">The name of the file to be created.</param>
        public void Export(string path)
        {
            if (Path.GetExtension(path) == "")
            {
                path += ".png";
            }
            this.Export(path, ImageFormat.Png);
        }

        /// <summary>Exports each layer of the given images as a PNG file.</summary>
        /// <param name="path">The path of the folder to be exported the files into.</param>
        /// <param name="images">The images to be exported.</param>
        /// <param name="smart">If <code>true</code>, allows for an encoding involving all three RGB colors.</param>
        /// <param name="createHtml"><code>true</code> if a HTML files allowing for easy access to the images is to be created, <code>false</code> otherwise.</param>
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

        /// <summary>Exports each layer of this image into a.PNG file.</summary>
        /// <param name="path">The path of the folder to be exported the files into.</param>
        /// <param name="smart">If <code>true</code>, allows for an encoding involving all three RGB colors.</param>
        /// <param name="createHtml"><code>true</code> if a HTML file for easy access to the images is to be created, <code>false</code> otherwise.</param>
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

        /// <summary>Exports a layer of this image as a bitmap.</summary>
        /// <param name="layer">The index of the layer to be exported.</param>
        /// <param name="smart">If <code>true</code>, allows for an encoding involving all three RGB colors.</param>
        /// <returns>The generated bitmap.</returns>
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
                        int bright = Math.Min((int)Math.Round(Math.Sqrt(this.Raw[layer, i, j]) * 255.0F * 3), 255 * 3);
                        color = Color.FromArgb(Math.Min(bright, 255), Math.Max(0, Math.Min(bright - 255, 255)), Math.Max(0, Math.Min(bright - 510, 255)));
                    }
                    else
                    {
                        int bright = Math.Min((int)Math.Round(this.Raw[layer, i, j] * 255.0F), 255);
                        color = Color.FromArgb(bright, bright, bright);
                    }
                    retVal.SetPixel(i, j, color);
                }
            }
            return retVal;
        }

        /// <summary>Exports this image as a bitmap.</summary>
        /// <returns>The generated bitmap.</returns>
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
                        int bright = (int)Math.Round(this.Raw[0, i, j] * 255.0F);
                        color = Color.FromArgb(bright, bright, bright);
                    }
                    else
                    {
                        color = Color.FromArgb((int)Math.Round(this.Raw[0, i, j] * 255.0F), (int)Math.Round(this.Raw[1, i, j] * 255.0F), (int)Math.Round(this.Raw[2, i, j] * 255.0F));
                    }
                    retVal.SetPixel(i, j, color);
                }
            }
            return retVal;
        }

        /// <summary>Normalizes each value of this image between <code>0</code> and <code>1</code>.</summary>
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
                        min = Math.Min(this.Raw[i, j, k], min);
                        max = Math.Max(this.Raw[i, j, k], max);
                    }
                }
                if (max > min)
                {
                    for (int j = 0; j < this.Width; j++)
                    {
                        for (int k = 0; k < this.Height; k++)
                        {
                            this.Raw[i, j, k] = (this.Raw[i, j, k] - min) / (max - min);
                        }
                    }
                }
                else
                {
                    Array.Clear(this.Raw, 0, this.Raw.Length);
                }
            }
        }

        /// <summary>Gets the minimum and the maximum value contanined in this image.</summary>
        /// <param name="min">The minimum value.</param>
        /// <param name="max">The maximum value.</param>
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
                        min = Math.Min(this.Raw[i, j, k], min);
                        max = Math.Max(this.Raw[i, j, k], max);
                    }
                }
            }
        }

        /// <summary>Normalizes multiple images together, using the minimum and maximum value from all of them.</summary>
        /// <param name="images">The images to be normalized.</param>
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

        /// <summary>Normalizes every value of an image between <code>0</code> and <code>1</code>.</summary>
        /// <param name="min">The value to be normalized to <code>0</code>.</param>
        /// <param name="max">The value to be normalized to <code>1</code>.</param>
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
                            this.Raw[i, j, k] = (this.Raw[i, j, k] - min) / (max - min);
                        }
                    }
                }
                else
                {
                    Array.Clear(this.Raw, 0, this.Raw.Length);
                }
            }
        }

        /// <summary>Sets random values between <code>0</code> and <code>1</code> for this image.</summary>
        public void Randomize()
        {
            for (int i = 0; i < this.Depth; i++)
            {
                for (int j = 0; j < this.Width; j++)
                {
                    for (int k = 0; k < this.Height; k++)
                    {
                        this.Raw[i, j, k] = RandomGenerator.GetFloat();
                    }
                }
            }
        }

        /// <summary>Sets every value of this image as <code>0</code>: turns the image black.</summary>
        public void Clear()
        {
            Array.Clear(this.Raw, 0, this.Raw.Length);
        }

        /// <summary>Turns black a portion of this image, setting every value in that portion as <code>0</code>.</summary>
        /// <param name="w">The lowest W coordinate of the section of this image to be cleared.</param>
        /// <param name="x">The lowest X coordinate of the section of this image to be cleared.</param>
        /// <param name="z">The lowest Z coordiante of the section of this image to be cleared.</param>
        /// <param name="depth">The depth of the section of this image to be cleared.</param>
        /// <param name="width">The width of the section of this image to be cleared.</param>
        /// <param name="height">The height of the section of this image to be cleared.</param>
        public void Clear(int w, int x, int z, int depth, int width, int height)
        {
            for (int i = w; i < w + depth; i++)
            {
                for (int j = x; j < x + width; j++)
                {
                    for (int k = z; k < z + height; k++)
                    {
                        this.Raw[w, j, k] = 0.0F;
                    }
                }
            }
        }

        /// <summary>Adds another image to this.</summary>
        /// <param name="image">The image to be added.</param>
        /// <param name="alpha">The multiplier for each value of the image to be added.</param>
        public void Add(Image image, float alpha = 1.0F)
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

        /// <summary>Returns the maximum value of this image in the given position.</summary>
        /// <param name="x">The X position to be searched.</param>
        /// <param name="y">The Y position to be searched.</param>
        /// <returns>The maximum value, among those with (X,Y) coordinates, regardless of its W coordinate.</returns>
        public float MaxAt(int x, int y)
        {
            float retVal = this.Raw[0, x, y];
            for (int i = 1; i < this.Depth; i++)
            {
                retVal = Math.Max(retVal, this.Raw[0, x, y]);
            }
            return retVal;
        }

        /// <summary>Preprocess a layer of this image for an MNIST-trained network.</summary>
        /// <param name="outputSize">The size of the adapted image.</param>
        /// <param name="outputPad">The padding of the adapted image.</param>
        /// <param name="w">The index of the layer to be adapted.</param>
        /// <returns>The preprocessed layer.</returns>
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
        }

        private static void FromMnistFunc(Stream stream, Image[] array, bool normalize, int count, int skip, bool emnist = false)
        {
            int width = (stream.ReadByte() << 24) + (stream.ReadByte() << 16) + (stream.ReadByte() << 8) + stream.ReadByte();
            int height = (stream.ReadByte() << 24) + (stream.ReadByte() << 16) + (stream.ReadByte() << 8) + stream.ReadByte();
            stream.Position += skip * width * height;
            for (int i = 0; i < count; i++)
            {
                array[i] = new Image(1, width, height);
                if (emnist)
                {
                    for (int j = 0; j < width; j++)
                    {
                        for (int k = 0; k < height; k++)
                        {
                            array[i].Raw[0, j, k] = stream.ReadByte() / 255.0F;
                        }
                    }
                }
                else
                {
                    for (int j = 0; j < height; j++)
                    {
                        for (int k = 0; k < width; k++)
                        {
                            array[i].Raw[0, k, j] = stream.ReadByte() / 255.0F;
                        }
                    }
                }
                if (normalize)
                {
                    array[i].Normalize();
                }
            }
        }

        /// <summary>Reads images from the MNIST format.</summary>
        /// <param name="stream">The stream to be read.</param>
        /// <param name="normalize"><code>true</code> if the images are to be normalized, <code>false</code> otherwise.</param>
        /// <param name="maxCount">The maximum amount of images to be read. If <code>0</code>, all available images will be read.</param>
        /// <param name="skip">The amount of images to be skipped.</param>
        /// <param name="emnist"><code>true</code> if the indices within the images are to be inverted, <code>false</code> otherwise.</param>
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

        /// <summary>Reads images from the MNIST format.</summary>
        /// <param name="stream">The stream to be read.</param>
        /// <param name="array">The array to be written the read images into.</param>
        /// <param name="normalize"><code>true</code> if the read images are to be normalized, <code>false</code> otherwise.</param>
        /// <param name="maxCount">The maximum amount of images to be read. If <code>0</code>, all available images are read.</param>
        /// <param name="skip">The amount of images to be skipped.</param>
        /// <param name="emnist"><code>true</code> if the indices have to be inverted.</param>
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

        /// <summary>Reads labels from the MNIST format.</summary>
        /// <param name="stream">The stream to be read.</param>
        /// <param name="smart">If <code>true</code>, allows the same label array to be used twice.</param>
        /// <param name="maxCount">The maximum amount of labels to be read. If <code>true</code>, all available labels are read.</param>
        /// <param name="skip">The amount of labels to be skipped.</param>
        /// <param name="labels">The maximum value for a label, plus one.</param>
        /// <returns>The read labels.</returns>
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

        /// <summary>Reads labels from the MNIST format.</summary>
        /// <param name="stream">The stream to be read.</param>
        /// <param name="array">The array to be written the labels into.</param>
        /// <param name="smart">If <code>true</code>, allow the same label array to be used multiple times.</param>
        /// <param name="maxCount">The maximum amount of labels to be read. If <code>0</code>, all available labels are read.</param>
        /// <param name="skip">The amount of labels to be skipped.</param>
        /// <param name="labels">The maximum value for a label, plus one.</param>
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

        /// <summary>Reads images from the MNIST format.</summary>
        /// <param name="fileName">The name of the file to be read.</param>
        /// <param name="normalize"><code>true</code> if the images must be normalized, <code>false</code> otherwise.</param>
        /// <param name="maxCount">The maximum amount of images to be read. If <code>true</code>, all images are read.</param>
        /// <param name="skip">The amount of images to be skipped.</param>
        /// <param name="emnist">If <code>true</code> switches the indices of the images.</param>
        /// <returns>The read images.</returns>
        public static Image[] FromMnist(string fileName, bool normalize = true, int maxCount = 0, int skip = 0, bool emnist = false)
        {
            FileStream fs = new FileStream(fileName, FileMode.Open);
            Image[] retVal = Image.FromMnist(fs, normalize, maxCount, skip, emnist);
            fs.Close();
            return retVal;
        }

        /// <summary>Reads labels from the MNIST format.</summary>
        /// <param name="fileName">The name of the file to be read.</param>
        /// <param name="smart">If <code>true</code>, allow the same label array to be used twice.</param>
        /// <param name="maxCount">The maximum amount of labels to be read. If <code>0</code>, all labels are read.</param>
        /// <param name="skip">The amount of images to be skipped.</param>
        /// <param name="labels">The maximum value for a label, plus one.</param>
        /// <returns>The read labels.</returns>
        public static double[][] FromMnistLabels(string fileName, bool smart = true, int maxCount = 0, int skip = 0, int labels = 10)
        {
            FileStream fs = new FileStream(fileName, FileMode.Open);
            double[][] retVal = Image.FromMnistLabels(fs, smart, maxCount, skip, labels);
            fs.Close();
            return retVal;
        }

        /// <summary>Read images from the MNIST format.</summary>
        /// <param name="fileName">The name of the file to be read.</param>
        /// <param name="array">The array to be written the read images into.</param>
        /// <param name="normalize">If <code>true</code>, normalizes the read images.</param>
        /// <param name="maxCount">The maximum amount of images to be read. If <code>0</code>, all available images are read.</param>
        /// <param name="skip">The amount of images to be skipped.</param>
        /// <param name="emnist">If <code>true</code> switches the indices.</param>
        public static void FromMnist(string fileName, Image[] array, bool normalize = true, int maxCount = 0, int skip = 0, bool emnist = false)
        {
            FileStream fs = new FileStream(fileName, FileMode.Open);
            Image.FromMnist(fs, array, normalize, maxCount, skip, emnist);
            fs.Close();
        }

        /// <summary>Read labels from the MNIST format.</summary>
        /// <param name="fileName">The name of the file to be read.</param>
        /// <param name="array">The array to be written the labels into.</param>
        /// <param name="smart">If <code>true</code>, allows for the same labels array to be used multiple times.</param>
        /// <param name="maxCount">The maximum amount of lables to be read. If <code>0</code>, all available labels are read.</param>
        /// <param name="skip">The amount of labels to be skipped.</param>
        /// <param name="labels">The maximum number for a label, plus one.</param>
        public static void FromMnistLabels(string fileName, double[][] array, bool smart = true, int maxCount = 0, int skip = 0, int labels = 10)
        {
            FileStream fs = new FileStream(fileName, FileMode.Open);
            Image.FromMnistLabels(fs, array, smart, maxCount, skip, labels);
            fs.Close();
        }
    }
}
