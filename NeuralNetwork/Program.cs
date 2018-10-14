using System;
using static Salty.Maths;
using Salty.AI;
using Mnist;

class Program
{
    public static void Main(string[] args)
    {
        Random rng = new Random();

        float[,] inputs = new float[60000, 784];

        float[,] expectedOutputs = new float[60000, 10];

        Console.WriteLine("Loading training data 0%");

        int exampleIndex = 0;
        foreach (Image image in Mnist.Reader.ReadTrainingData())
        {
            
            if (exampleIndex % 10000 == 0)
            {
                Console.WriteLine("Loading training data " + exampleIndex / 600 + "%");
            }

            float[] input = new float[784];
            for (int i = 0; i < 28; i++)
            {
                String row = "";
                for (int j = 0; j < 28; j++)
                {
                    int pixel = (int) image.Data[i, j];
                    inputs[exampleIndex, 28 * i + j] = ((float) pixel) / 255f;
                    row += (pixel > 0.5) ? "x" : " ";
                }
            }

            for (int i = 0; i < 10; i++)
            {
                expectedOutputs[exampleIndex, i] = ((int) image.Label == i) ? 1f : 0f;
                //Console.Write((((int) image.Label == i) ? 1f : 0f) + " ");
            }

            exampleIndex++;
        }

        Console.Write("Training data arrays initialised.");

        TrainingData td = new TrainingData(inputs, expectedOutputs);

        Console.Write("Training data filled into stucture.");
        
        NeuralNetwork nn = new NeuralNetwork(784, 30, 10);

        Console.WriteLine("Beginning training.");
        do
        {
            nn.Train(td, 5f, 1, 32);
            Console.WriteLine("Cost this iteration: " + nn.Cost);
        } 
        while (nn.Cost > 0.2);
    }
}
