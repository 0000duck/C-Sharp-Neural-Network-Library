using System;
using static Salty.Maths;
using Salty.AI;

class Program
{
    public static void Main(string[] args)
    {
        Random rng = new Random();

        float[,] inputs = new float[,] {
            {0f, 0f},
            {0f, 1f},
            {1f, 0f},
            {1f, 1f}
        };

        float[,] expectedOutputs = new float[,] {
            {0},
            {1},
            {1},
            {0}
        };

        TrainingData td = new TrainingData(inputs, expectedOutputs);
        
        NeuralNetwork nn = new NeuralNetwork(2, 2, 1);
        nn.Train(td, 0.01f, 10, 2);
    }
}
