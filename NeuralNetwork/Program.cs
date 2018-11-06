using System;
using Salty.AI;
using Mnist;
using System.IO;

class Program
{
    public static void Main(string[] args)
    {
        /*FileStream filestream = new FileStream("out.txt", FileMode.Create);
        StreamWriter streamwriter = new StreamWriter(filestream);
        streamwriter.AutoFlush = true;
        //Console.SetOut(streamwriter);
        //Console.SetError(streamwriter);

        Random rng = new Random();

        double[,] inputs = new double[60000, 784];

        double[,] expectedOutputs = new double[60000, 10];

        int exampleIndex = 0;
        foreach (Image image in Mnist.Reader.ReadTrainingData())
        {
            
            if (exampleIndex % 10000 == 0)
            {
                Console.WriteLine("Loading training data " + exampleIndex / 600 + "%");
            }

            double[] input = new double[784];
            for (int i = 0; i < 28; i++)
            {
                String row = "";
                for (int j = 0; j < 28; j++)
                {
                    int pixel = (int) image.Data[i, j];
                    inputs[exampleIndex, 28 * i + j] = ((double) pixel) / 255f;
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

        NeuralNetwork nn = new NeuralNetwork(784, 10);*/

        /*double[,] xorIn = {
            {0f, 0f}, {0f, 1f}, {1f, 0f}, {1f, 1f}
        };

        double[,] xorOut = {
            {0f}, {1f}, {1f}, {0f}
        };

        TrainingData xorTrain = new TrainingData(xorIn, xorOut);

        NeuralNetwork nn = new NeuralNetwork(2, 3, 1);
        do {        
            nn.Train(xorTrain, 0.1f, 5, 3);
            Console.WriteLine("Cost: " + nn.Cost);
        } while (nn.Cost > 0.1);

        //NeuralNetwork nn = new NeuralNetwork("save.xml");

        Console.WriteLine(nn.Compute(0f, 0f)[0]);
        Console.WriteLine(nn.Compute(0f, 1f)[0]);
        Console.WriteLine(nn.Compute(1f, 0f)[0]);
        Console.WriteLine(nn.Compute(1f, 1f)[0]);

        Console.WriteLine(nn);
        nn.Save("save.xml");*/
/*
        Console.Write("Training data arrays initialised.");

        TrainingData td = new TrainingData(inputs, expectedOutputs);

        Console.Write("Training data filled into stucture.");

        Console.WriteLine("Beginning training.");
        do
        {
            nn.Train(td, 3f, 1, 128);
            Console.WriteLine("Cost this iteration: " + nn.Cost);
        } 
        while (false);

        Console.WriteLine(nn);
*/
        NeuralNetwork nn2 = new NeuralNetwork("save.xml");
        Console.WriteLine(nn2);
    }
}
