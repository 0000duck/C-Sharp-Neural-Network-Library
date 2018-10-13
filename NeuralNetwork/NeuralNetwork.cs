using System;
using static Salty.Maths;
using System.Linq;

namespace Salty.AI
{
    public class NeuralNetwork
    {
        public readonly int Size;
        public readonly int[] Structure;

        private readonly Random rng;

        private Matrix[] activations, weights, biases;

        public NeuralNetwork(params int[] structure) 
        {
            this.rng = new Random();
            this.Structure = structure;
            Size = structure.Length;

            activations = new Matrix[Size];
            weights = new Matrix[Size - 1];
            biases = new Matrix[Size - 1];

            for (int i = 0; i < Size - 1; i++)
            {
                weights[i] = 10f * Matrix.Random(Structure[i + 1], Structure[i], rng);
                biases[i] = 10f * Matrix.Random(Structure[i + 1], 1, rng);
            }
        }

        private void feedForward()
        {
            for (int i = 0; i < Size - 1; i++)
            {
                activations[i + 1] = Matrix.Apply(weights[i] * activations[i] + biases[i], Sigmoid);
            }
        }

        public void Train(TrainingData trainingData, float learningRate, int epochs, int miniBatchSize)
        {
            miniBatchSize = Math.Min(miniBatchSize, trainingData.SampleSize);
            
            for (int i = 0; i < epochs; i++)
            {
                // Randomise the training data
                trainingData.Shuffle(rng);

                // Pick out as many mini-batches as we can to cover al training data
                int nMiniBatches = trainingData.SampleSize / miniBatchSize;
                TrainingData[] miniBatches = new TrainingData[nMiniBatches];
                for (int j = 0; j < nMiniBatches; j++)
                {
                    miniBatches[j] = trainingData.GetMiniBatch(j * miniBatchSize, miniBatchSize);
                }

                foreach (TrainingData miniBatch in miniBatches)
                {
                    stepGradientDescent(miniBatch, learningRate);
                }
                
            }
        }

        private void stepGradientDescent(TrainingData miniBatch, float learningRate)
        {
            int miniBatchSize = miniBatch.SampleSize;

            

            /*
            // Calculate the average cost over the training examples in this minibatch.
            float averageCost = 0f;
            for (int j = 0; j < miniBatchSize; j++)
            {
                // TODO: Tidy this mess up!
                Matrix expected = Matrix.FromArray(miniBatch.GetExpectedOutput(j));
                Matrix actual = compute(Matrix.FromArray(miniBatch.GetInput(j)));
                Matrix difference = expected - actual;

                float magDiffSquared = 0f;
                foreach (float d in difference.GetColumn(0))
                {
                    magDiffSquared += d * d;
                }

                averageCost += magDiffSquared / 2f;
            }

            averageCost /= (float) miniBatchSize;
            Console.WriteLine("Avg cost over this minibatch: " + averageCost);
            */
        }

        public float[] Compute(params float[] input) 
        {
            activations[0] = Matrix.FromArray(input);
            feedForward();
            return activations[Size - 1].GetColumn(0);
        }

        private Matrix compute(Matrix input) 
        {
            activations[0] = input;
            feedForward();
            return activations[Size - 1];
        }

        public static float Sigmoid(float x)
        {
            return (float)(1 / (1 + Math.Pow(Math.E, -x)));
        }
    }
}
