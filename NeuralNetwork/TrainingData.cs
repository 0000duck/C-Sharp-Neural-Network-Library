using System;
using System.Linq;
using static Salty.Maths;

namespace Salty.AI {

    /// <summary>Represents a set of training inputs and their corresponding 
    /// outputs.</summary>
    public class TrainingData
    {
        /// <summary>A matrix with rows representing training example inputs.
        /// </summary>
        private Matrix inputs;
            
        /// <summary>A matrix with rows representing training example expected 
        /// outputs.</summary>
        private Matrix expectedOutputs;

        /// <summary>The number of training examples in this training data set.
        /// </summary>
        public readonly int SampleSize;

        /// <summary>The size of each input in this training data set.
        /// </summary>
        public readonly int InputSize;

        /// <summary>The size of each output in this training data set.
        /// </summary>
        public readonly int OutputSize;

        /// <summary>
        /// Initialises a new training data set with the specified inputs and 
        /// expected outputs.</summary>
        /// <param name="inputs">The inputs as rows in a 2D array.</param>
        /// <param name="expectedOutputs">The expected outputs as rows in a 2D 
        /// array.</param>
        public TrainingData(double[,] inputs, double[,] expectedOutputs)
        {
            if (inputs.GetLength(0) != expectedOutputs.GetLength(0))
            {
                throw new ArgumentException("sample size must be equal for " +
                    "both input and expected output");
            }

            this.inputs = new Matrix(inputs);
            this.expectedOutputs = new Matrix(expectedOutputs);
            this.SampleSize = this.inputs.RowCount;
            this.InputSize = this.inputs.ColumnCount;
            this.OutputSize = this.expectedOutputs.ColumnCount;
        }

        /// <summary>Shuffles the inputs and outputs in-place, retaining their 
        /// correspondance.</summary>
        /// <param name="rng">The random objected used to shuffle the current 
        /// training data set.</param>
        public void Shuffle(Random rng)
        {
            // Get an array of indices
            int[] indices = new int[SampleSize];
            for (int i = 0; i < SampleSize; i++) 
            {
                indices[i] = i;
            }

            // Shuffle the indices
            indices = indices.OrderBy(x => rng.Next()).ToArray();

            // Suffle the rows of both matrices
            Matrix inShuffled = new Matrix(SampleSize, InputSize);
            Matrix outShuffled = new Matrix(SampleSize, OutputSize);

            for (int i = 0; i < SampleSize; i++) 
            {
                inShuffled.SetRow(i, inputs.GetRow(indices[i]));
                outShuffled.SetRow(i, expectedOutputs.GetRow(indices[i]));
            }

            inputs = inShuffled;
            expectedOutputs = outShuffled;
        }

        /// <summary>Returns the input at the given index into the current 
        /// training data set.</summary>
        /// <param name="i">The index into the current training data set to get 
        /// the input at.</param>
        /// <returns>The input at the given index into the current training 
        /// data set.</returns>
        public double[] GetInput(int i)
        {
            return inputs.GetRow(i);
        }

        /// <summary>Returns the expected output at the given index into the 
        /// current training data set.</summary>
        /// <param name="i">The index into the current training data set to get 
        /// the expected output at.</param>
        /// <returns>The expected output at the given index into the current 
        /// training data set.</returns>
        public double[] GetExpectedOutput(int i)
        {
            return expectedOutputs.GetRow(i);
        }

        /// <summary>Returns a subset of the current training data set, called 
        /// a mini-batch.</summary>
        /// <param name="startIndex">The starting index into the curren 
        /// training data set to serve as the beginning of the mini-batch.
        /// </param>
        /// <param name="miniBatchSize">The size of the mini-batch to get.
        /// </param>
        /// <returns>The mini-batch starting at the given start index and of 
        /// the given size.</returns>
        public TrainingData GetMiniBatch(int startIndex, int miniBatchSize)
        {
            if (startIndex < 0 || startIndex >= SampleSize)
            {
                throw new ArgumentOutOfRangeException("mini-batch start " +
                    "index is out of bounds");
            }

            if (startIndex + miniBatchSize > SampleSize)
            {
                throw new ArgumentOutOfRangeException("mini-batch size is " +
                    "too large for the given start index");
            }

            double[,] miniBatchInputs = new double[miniBatchSize, InputSize];
            double[,] miniBatchOutputs = new double[miniBatchSize, OutputSize];
            for (int i = 0; i < miniBatchSize; i++)
            {
                for (int j = 0; j < InputSize; j++)
                {
                    miniBatchInputs[i, j] = inputs[i + startIndex, j];
                }

                for (int j = 0; j < OutputSize; j++)
                {
                    miniBatchOutputs[i,j] = expectedOutputs[i + startIndex, j];
                }
            }

            return new TrainingData(miniBatchInputs, miniBatchOutputs);
        }

        /// <summary>Returns a <see cref="System.String" /> that represents 
        /// the current training data set.</summary>
        /// <returns>A <see cref="System.String" /> that represents the 
        /// current training data set.</returns>
        public override string ToString()
        {
            return "INPUTS\n" + inputs.ToString() + 
                "\nEXPECTED OUTPUTS\n" + expectedOutputs.ToString();
        }
    }
}