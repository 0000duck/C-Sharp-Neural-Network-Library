using System;
using static Salty.Maths;
using Salty.AI;

class Program
{
    public static void Main(string[] args)
    {
        Random rng = new Random();

        // Establish the neural network
        NeuralNetwork nn = new NeuralNetwork(2, 2, 1);

        // Establish the training data for xor
        float[][] xorInputs = {
            new float[] {0, 0},
            new float[] {0, 1},
            new float[] {1, 0},
            new float[] {1, 1}
        };

        float[][] xorDesiredOutputs = {
            new float[] {0}, 
            new float[] {1},
            new float[] {1},
            new float[] {0}
        };

        // Train
        float error;
        do 
        {
            error = nn.Train(xorInputs, xorDesiredOutputs);
        }
        while (error > 0.01f);
    }

    public static float Xor(float a, float b)
    {
            return Math.Abs(a - b);
    }
}
