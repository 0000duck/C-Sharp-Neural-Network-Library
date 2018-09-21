using System;
using static Salty.Maths;

namespace Salty.AI
{
    public class NeuralNetwork
    {
        public readonly int Size;
        public readonly int[] Structure;

        private readonly Random rng;

        private Matrix[] activations, weights, biases;

        /// <summary>
        /// Initialises a neural network with random weights and biases.
        /// </summary>
        /// <param name="structure">The list of numbers representing the quantity of neurons in each layer.
        /// </param>
        public NeuralNetwork(params int[] structure)
        {
            this.rng = new Random();
            this.Structure = structure;
            Size = structure.Length;

            activations = new Matrix[Size];


 
            // Randomly initialise all the weights and biases as column vectors.
            weights = new Matrix[Size - 1];
            biases = new Matrix[Size - 1];

            for (int i = 0; i < Size - 1; i++)
            {
                weights[i] = Matrix.Random(structure[i + 1], structure[i], rng);
                biases[i] = Matrix.Random(structure[i + 1], 1, rng);
            }         
        }

        private float[] feedForward(params float[] input)
        {
            if (input.Length != Structure[0])
            {
                throw new ArgumentException("input size must be compatible with network structure");
            }

            activations[0] = new Matrix(Structure[0], 1);
            activations[0].SetColumn(0, input);

            for (int i = 0; i < Size - 1; i++)
            {
                activations[i + 1] = Matrix.Apply(weights[i] * activations[i] + biases[i], Sigmoid);
            }

            return activations[Size - 1].GetColumn(0);
        }

        private Matrix getErrors(int layerIndex, Matrix expectedOutput)
        {
            Matrix actualOutput = activations[layerIndex];

            if (layerIndex == Size - 1)
            {
                return expectedOutput - actualOutput;
            }

            Matrix incomingWeights = weights[layerIndex];

            return incomingWeights.Transpose() * getErrors(layerIndex + 1, expectedOutput);
        }

        public override string ToString()
        {
            string repr = "";

            repr += "_INPUT_LAYER_\n";
            repr += "Activation:\n" + activations[0] + "\n";

            for (int i = 1; i < Size - 1; i++)
            {
                repr += "\n_HIDDEN_LAYER_#" + i + "_\n";
                repr += "Activation:\n" + activations[i] + "\n";
                repr += "Incoming Weights:\n" + weights[i - 1] + "\n";
                repr += "Biases:\n" + biases[i - 1] + "\n";
            }

            repr += "\n_OUTPUT_LAYER_\n";
            repr += "Activation:\n" + activations[Size - 1] + "\n";
            repr += "Incoming Weights:\n" + weights[Size - 2] + "\n";
            repr += "Biases:\n" + biases[Size - 2] + "\n";


            return repr;
        }

        public static void Main(string[] args)
        {
            Console.WriteLine("Neural networking expecting output of 0.5:\n");
            NeuralNetwork nn = new NeuralNetwork(2, 2, 1);
            nn.feedForward(0f, 1f);
            Console.WriteLine(nn);
            
            Matrix expected = new Matrix(1, 1);
            expected[0, 0] = 0.5f;

            Console.WriteLine("Input errors:\n" + nn.getErrors(0, expected) + "\n");
            Console.WriteLine("Hidden errors:\n" + nn.getErrors(1, expected) + "\n");
            Console.WriteLine("Output errors:\n" + nn.getErrors(2, expected) + "\n");

            Console.ReadLine();
        }

        /// <summary>
        /// A smooth function which takes any float and positions it proportionately in the range from 0 to 1
        /// (inclusive).
        /// </summary>
        /// <param name="x">The float to shrink to the range [0, 1].</param>
        /// <returns>A number proportional to the input contracted to the range [0, 1].</returns>
        public float Sigmoid(float x)
        {
            return (float)(1 / (1 + Math.Pow(Math.E, -x)));
        }

    }
}
