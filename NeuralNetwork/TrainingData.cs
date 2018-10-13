using System;
using System.Linq;
using static Salty.Maths;

namespace Salty.AI {

    public class TrainingData
    {
        private Matrix inputs, expectedOutputs;

        public readonly int SampleSize;

        public readonly int InputSize;

        public readonly int OutputSize;

        public TrainingData(float[,] inputs, float[,] expectedOutputs)
        {
            /*if (inputs.GetLength(1) != expectedOutputs.GetLength(1))
            {
                throw new ArgumentException("sample size must be equal for both input and expected output");
            }*/

            this.inputs = new Matrix(inputs);
            this.expectedOutputs = new Matrix(expectedOutputs);
            this.SampleSize = this.inputs.RowCount;
            this.InputSize = this.inputs.ColumnCount;
            this.OutputSize = this.expectedOutputs.ColumnCount;
        }

        public void Shuffle(Random rng)
        {
            int[] randIndices = new int[SampleSize];
            for (int i = 0; i < SampleSize - 1; i++)
            {
                int randIndex = rng.Next(SampleSize);
                while (randIndices.Contains(randIndex))
                {
                    randIndex = rng.Next(SampleSize);
                }
                randIndices[i] = randIndex;
            }

            Matrix randInputs = new Matrix(SampleSize, InputSize);
            Matrix randOutputs = new Matrix(SampleSize, OutputSize);
            for (int i = 0; i < SampleSize; i++)
            {
                int randIndex = randIndices[i];
                randInputs.SetRow(i, inputs.GetRow(randIndex));
                randOutputs.SetRow(i, expectedOutputs.GetRow(randIndex));
            }

            inputs = randInputs;
            expectedOutputs = randOutputs;
        }
    
        public float[] GetInput(int i)
        {
            return inputs.GetRow(i);
        }

        public float[] GetExpectedOutput(int i)
        {
            return expectedOutputs.GetRow(i);
        }

        public TrainingData GetMiniBatch(int startIndex, int miniBatchSize)
        {
            float[,] miniBatchInputs = new float[miniBatchSize, InputSize];
            float[,] miniBatchOutputs = new float[miniBatchSize, OutputSize];
            for (int i = 0; i < miniBatchSize; i++)
            {
                for (int j = 0; j < InputSize; j++)
                {
                    miniBatchInputs[i, j] = inputs[i + startIndex, j];
                }

                for (int j = 0; j < OutputSize; j++)
                {
                    miniBatchOutputs[i, j] = expectedOutputs[i + startIndex, j];
                }
            }

            return new TrainingData(miniBatchInputs, miniBatchOutputs);
        }

        public override string ToString()
        {
            return "INPUTS\n" + inputs.ToString() + 
                "\nEXPECTED OUTPUTS\n" + expectedOutputs.ToString();
        }
    }
}