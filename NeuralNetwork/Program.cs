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
        do
        {
            nn.Train(td, 0.1f, 5, 3);
            Console.WriteLine(nn.Cost);
        } 
        while (nn.Cost > 0.001);

        float[] test = nn.Compute(0, 0);
        Console.WriteLine("Guess for 0 XOR 0: " + test[0]);

        test = nn.Compute(0, 1);
        Console.WriteLine("Guess for 0 XOR 1: " + test[0]);

        test = nn.Compute(1, 0);
        Console.WriteLine("Guess for 1 XOR 0: " + test[0]);

        test = nn.Compute(1, 1);
        Console.WriteLine("Guess for 1 XOR 1: " + test[0]);

        Console.WriteLine(nn);

    }
}
