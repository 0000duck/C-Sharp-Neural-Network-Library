using System;
using static Salty.Maths;

namespace Salty.AI
{
    public class NeuralNetwork
    {
        public readonly int LayerCount;
        public readonly int[] Structure;

        public float Cost;

        private readonly Random rng;

        private Matrix[] activations, weights, biases;

        public NeuralNetwork(params int[] structure) 
        {
            this.rng = new Random();
            this.Structure = structure;
            LayerCount = structure.Length;

            activations = new Matrix[LayerCount];
            weights = new Matrix[LayerCount - 1];
            biases = new Matrix[LayerCount - 1];

            for (int i = 0; i < LayerCount - 1; i++)
            {
                weights[i] = Matrix.Random(Structure[i + 1], Structure[i], rng);
                biases[i] = Matrix.Random(Structure[i + 1], 1, rng);
            }
        }

        private void feedForward(Matrix input)
        {
            activations[0] = input;
            for (int i = 0; i < LayerCount - 1; i++)
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

            Matrix[] weightsDirection = new Matrix[LayerCount - 1];
            Matrix[] biasesDirection = new Matrix[LayerCount - 1];

            // Initialise the direction matrices to be of the same dimensions as the weights and biases
            for (int i = 0; i < LayerCount - 1; i++)
            {
                weightsDirection[i] = new Matrix(weights[i].RowCount, weights[i].ColumnCount);
                biasesDirection[i] = new Matrix(biases[i].RowCount, 1);
            }
            
            // Sum the steps of weights and biases for each training example in the mini-batch
            for (int i = 0; i < miniBatchSize; i++)
            {
                Matrix input = Matrix.FromArray(miniBatch.GetInput(i));
                Matrix expectedOutput = Matrix.FromArray(miniBatch.GetExpectedOutput(i));
                
                Tuple<Matrix[], Matrix[]> direction = getCostGradient(input, expectedOutput);
                
                for (int l = 0; l < LayerCount - 1; l++)
                {
                    weightsDirection[l] += direction.Item1[l];
                    biasesDirection[l] += direction.Item2[l];
                }
            }

            for (int l = 0; l < LayerCount - 1; l++)
            {
                weights[l] -= (learningRate / (float) miniBatchSize) * weightsDirection[l];
                biases[l] -= (learningRate / (float) miniBatchSize) * biasesDirection[l];
            }
        }

        private Tuple<Matrix[], Matrix[]> getCostGradient(Matrix input, Matrix expectedOutput)
        {
            Matrix[] weightsDirection = new Matrix[LayerCount - 1];
            Matrix[] biasesDirection = new Matrix[LayerCount - 1];

            Matrix[] errors = new Matrix[LayerCount - 1];

            feedForward(input);

            // Update the cost
            Matrix squaredDifferences = Matrix.Apply(expectedOutput - activations[LayerCount - 1], x => x * x);
            Cost = Maths.Sum(squaredDifferences.GetColumn(0)) / 2f;

            // Calculate the output error
            errors[errors.Length - 1] = Matrix.Hadamard(
                activations[LayerCount - 1] - expectedOutput, 
                Matrix.Apply(weights[LayerCount - 2] * activations[LayerCount - 2] + biases[LayerCount - 2], DerivativeOfSigmoid)
            );

            // Calculate the rest of the errors, moving backwards through the network
            for (int l = LayerCount - 3; l >= 0; l--)
            {
                errors[l] = Matrix.Hadamard(
                    weights[l + 1].Transpose() * errors[l + 1],
                    Matrix.Apply(weights[l] * activations[l] + biases[l], DerivativeOfSigmoid)
                );
            }

            /* The rate of change of the cost with respect a neuron's bias is exactly 
             * equal to the error from the neuron */
            biasesDirection = errors;

            // Calculate the rate of change of the cost with respect to all the weights
            for (int l = 0; l < LayerCount - 1; l++)
            {
                // The weight directions matrix from layer l to layer (l + 1)
                Matrix weightDirection = new Matrix(weights[l].RowCount, weights[l].ColumnCount);

                for (int i = 0; i < Structure[l + 1]; i++)
                {
                    for (int j = 0; j < Structure[l]; j++)
                    {
                        /* Calculate rate of change of the cost with respect to 
                         * the weight from the "j"th neuron in layer l to 
                         * the "i"th neuron in layer (l + 1) */
                        weightDirection[i, j] = activations[l][j, 0] * errors[l][i, 0];
                    }
                }

                weightsDirection[l] = weightDirection;
            }

            return new Tuple<Matrix[], Matrix[]>(weightsDirection, biasesDirection);
        }

        public float[] Compute(params float[] input) 
        {
            activations[0] = Matrix.FromArray(input);
            feedForward(Matrix.FromArray(input));
            return activations[LayerCount - 1].GetColumn(0);
        }

        private Matrix compute(Matrix input) 
        {
            feedForward(input);
            return activations[LayerCount - 1];
        }

        public static float Sigmoid(float x)
        {
            return (float)(1 / (1 + Math.Pow(Math.E, -x)));
        }

        public static float DerivativeOfSigmoid(float x)
        {
            float s = Sigmoid(x);
            return s * (1 - s);
        }

        public override string ToString()
        {
            string repr = "";
            repr += "_INPUT_LAYER_\n";
            repr += "Activation:\n" + activations[0] + "\n";
            for (int i = 1; i < LayerCount - 1; i++)
            {
                repr += "\n_HIDDEN_LAYER_#" + i + "_\n";
                repr += "Activation:\n" + activations[i] + "\n";
                repr += "Incoming Weights:\n" + weights[i - 1] + "\n";
                repr += "Biases:\n" + biases[i - 1] + "\n";
            }
            repr += "\n_OUTPUT_LAYER_\n";
            repr += "Activation:\n" + activations[LayerCount - 1] + "\n";
            repr += "Incoming Weights:\n" + weights[LayerCount - 2] + "\n";
            repr += "Biases:\n" + biases[LayerCount - 2] + "\n";
            return repr;
        }
    }
}
