using System;
using static Salty.Maths;
using static System.IO.File;
using System.Collections.Generic;

namespace Salty.AI
{
    /// <summary>Represents a feed-forward multi-layer perception, or neural 
    /// network, with no loops. The network trains with Stochastic gradient 
    /// descent using back propagation. The activation function for neurons is 
    /// the sigmoid/logistic function. The inputs and outputs are floating 
    /// point numbers. The network uses the quadratic cost function. </summary>
    public class NeuralNetwork
    {
        /// <summary>The number of layers in the current network.</summary>
        public readonly int LayerCount;

        /// <summary>The structure of the current network. That is, a list of 
        /// numbers representing the number of neurons in each layer, starting 
        /// with the number of inputs and ending with the number of outputs.
        /// </summary>
        public readonly int[] Structure;

        /// <summary>The cost of the current network. This is calculated during 
        /// training using the quadratic cost function.</summary>
        public float Cost
        {
            get;
            private set;
        }

        /// <summary>The random number generator for the current network.
        /// </summary>
        private readonly Random rng;

        /// <summary>The activations of the neurons in each layer. That is, an 
        /// array of column vectors representing the activation value of the 
        /// neurons in each layer.</summary>
        private Matrix[] activations;

        /// <summary>The weight matrices for each layer.</summary>
        private Matrix[] weights;

        /// <summary>The biases of the neurons in each layer.</summary>
        private Matrix[] biases;

        /// <summary>Initialises the current network with the given structure. 
        /// Sets all weights and biases to random values between -1 (inclusive) 
        /// and 1 (exclusive).</summary>
        /// <param name="structure"></param>
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
                weights[i]= Matrix.Random(Structure[i + 1], Structure[i], rng);
                biases[i] = Matrix.Random(Structure[i + 1], 1, rng);
            }
        }

        /// <summary>Feeds the input forward through the network to produce the 
        /// corresponding output.</summary>
        /// <param name="input">The input to feed forward through the network.
        /// </param>
        private void feedForward(Matrix input)
        {
            activations[0] = input;
            for (int i = 0; i < LayerCount - 1; i++)
            {
                activations[i + 1] = Matrix.Apply(weights[i] * activations[i] + 
                    biases[i], sigmoid);
            }
        }

        /// <summary>Trains the current network on the given training data, 
        /// with the given learning rate using Stochastic gradient descent. 
        /// Covers the training data in mini-batches of the given size for each 
        /// epoch.</summary>
        /// <param name="trainingData">The training data to train on.</param>
        /// <param name="learningRate">The constant rate of learning overal all 
        /// epochs.</param>
        /// <param name="nEpochs">The number of epochs to train over.</param>
        /// <param name="miniBatchSize">The size of each mini-batch.</param>
        public void Train(TrainingData trainingData, 
            float learningRate, int nEpochs, int miniBatchSize)
        {
            miniBatchSize = Math.Min(miniBatchSize, trainingData.SampleSize);
            
            for (int i = 0; i < nEpochs; i++)
            {
                // Randomise the training data
                trainingData.Shuffle(rng);

                /* Pick out as many mini-batches as we can to cover al training 
                 * data */
                int nMiniBatches = trainingData.SampleSize / miniBatchSize;
                TrainingData[] miniBatches = new TrainingData[nMiniBatches];

                for (int j = 0; j < nMiniBatches; j++)
                {
                    miniBatches[j] = trainingData.GetMiniBatch(
                        j * miniBatchSize, miniBatchSize);
                }

                for (int j = 0; j < nMiniBatches; j++)
                {
                    TrainingData miniBatch = miniBatches[j];
                    Console.WriteLine(
                        "Epoch " + i + " of " + nEpochs + ": Processing mini" +
                        "-batch " + j + " of " + nMiniBatches);
                    stepGradientDescent(miniBatch, learningRate);
                }
            }
        }

        /// <summary>Performs one step of gradient descent according to the 
        /// current network's cost after training on the given mini-batch of 
        /// training data. Updates all weights and biases according to the 
        /// gradient of the cost function in the network's current 
        /// configuration.</summary>
        /// <param name="miniBatch">The mini-batch of training data to use for 
        /// determining the cost of the current network.</param>
        /// <param name="learningRate">The gradient descent step size.</param>
        private void stepGradientDescent(TrainingData miniBatch, 
            float learningRate)
        {
            int miniBatchSize = miniBatch.SampleSize;

            Matrix[] weightsDirection = new Matrix[LayerCount - 1];
            Matrix[] biasesDirection = new Matrix[LayerCount - 1];

            /* Initialise the direction matrices to be of the same dimensions 
             * as the weights and biases */
            for (int i = 0; i < LayerCount - 1; i++)
            {
                weightsDirection[i] = new Matrix(weights[i].RowCount, 
                    weights[i].ColumnCount);
                biasesDirection[i] = new Matrix(biases[i].RowCount, 1);
            }

            /* Sum the steps of weights and biases for each training example in 
             * the mini-batch */
            for (int i = 0; i < miniBatchSize; i++)
            {
                Matrix input = Matrix.FromArray(miniBatch.GetInput(i));
                Matrix expectedOutput = Matrix.FromArray(
                    miniBatch.GetExpectedOutput(i));
            
                Tuple<Matrix[], Matrix[]> direction = getCostGradient(
                    input, expectedOutput);
                
                for (int l = 0; l < LayerCount - 1; l++)
                {
                    weightsDirection[l] += direction.Item1[l];
                    biasesDirection[l] += direction.Item2[l];
                }
            }

            for (int l = 0; l < LayerCount - 1; l++)
            {
                weights[l] -= 
                    (learningRate / miniBatchSize) * weightsDirection[l];
                biases[l] -= 
                    (learningRate / miniBatchSize) * biasesDirection[l];
            }
        }

        /// <summary>Returns the gradient of the cost function with respect to 
        /// each weight and bias in the current network for a given training 
        /// example. This determines the direction of a gradient descent step.
        /// </summary>
        /// <param name="input">The input for a training example to use in 
        /// determining the gradient of the cost.</param>
        /// <param name="expectedOutput">The expected output for a training 
        /// example to use in determining the gradient of the cost.</param>
        /// <returns>the gradient of the cost function with respect to each 
        /// weight and bias in the current network.</returns>
        private Tuple<Matrix[], Matrix[]> getCostGradient(Matrix input, 
            Matrix expectedOutput)
        {
            Matrix[] weightsDirection = new Matrix[LayerCount - 1];
            Matrix[] biasesDirection = new Matrix[LayerCount - 1];

            Matrix[] errors = new Matrix[LayerCount - 1];

            feedForward(input);

            // Update the cost
            Matrix squaredDifferences = Matrix.Apply(
                expectedOutput - activations[LayerCount - 1], 
                x => x * x);
            Cost = Maths.Sum(squaredDifferences.GetColumn(0)) / 2f;

            // Calculate the output error
            errors[errors.Length - 1] = Matrix.Hadamard(
                activations[LayerCount - 1] - expectedOutput, 
                Matrix.Apply(weights[LayerCount - 2] * 
                activations[LayerCount - 2] + biases[LayerCount - 2], 
                derivativeOfSigmoid)
            );

            /* Calculate the rest of the errors, moving backwards through the 
             * network */
            for (int l = LayerCount - 3; l >= 0; l--)
            {
                errors[l] = Matrix.Hadamard(
                    weights[l + 1].Transpose() * errors[l + 1],
                    Matrix.Apply(weights[l] * activations[l] + biases[l], 
                    derivativeOfSigmoid)
                );
            }

            /* The rate of change of the cost with respect a neuron's bias is 
             * exactly equal to the error from the neuron */
            biasesDirection = errors;

            /* Calculate the rate of change of the cost with respect to all the 
             * weights */
            for (int l = 0; l < LayerCount - 1; l++)
            {
                // The weight directions matrix from layer l to layer (l + 1)
                Matrix weightDirection = 
                    new Matrix(weights[l].RowCount, weights[l].ColumnCount);

                for (int i = 0; i < Structure[l + 1]; i++)
                {
                    for (int j = 0; j < Structure[l]; j++)
                    {
                        /* Calculate rate of change of the cost with respect to 
                         * the weight from the "j"th neuron in layer l to 
                         * the "i"th neuron in layer (l + 1) */
                        weightDirection[i, j] = 
                            activations[l][j, 0] * errors[l][i, 0];
                    }
                }

                weightsDirection[l] = weightDirection;
            }

            return new Tuple<Matrix[], Matrix[]>(weightsDirection, 
                biasesDirection);
        }

        /// <summary>Returns the current network's output for the given input.
        /// </summary>
        /// <param name="input">The input to feed forward through the current 
        /// network.</param>
        /// <returns>The current network's output for the given input.
        /// </returns>
        public float[] Compute(params float[] input) 
        {
            activations[0] = Matrix.FromArray(input);
            feedForward(Matrix.FromArray(input));
            return activations[LayerCount - 1].GetColumn(0);
        }

        public void Save(string filePath)
        {
            List<string> lines = new List<string>();
            for (int l = 0; l < LayerCount; l++)
            {
                string line = String.Format("<Layer Name=\"L{0}\">\n", l);
                for (int i = 0; i < Structure[l]; i++)
                {
                    line += String.Format("\t<Neuron Name=\"N{1}\" " +
                        "Bias=\"{2}\">\n", l, i, biases[l][i, 0]);
                    if (l > 0)
                    {
                        line += "\t<IncomingWeights>\n";
                        for (int j = 0; j < Structure[l - 1]; j++)
                        {
                            line += String.Format(
                                "\t\t<Weight Layer=\"L{0}\" Neuron=\"{1}\" " +
                                "Strength=\"{2}\" />\n", 
                                l - 1, j, weights[l - 1][i, j]);
                        }
                        line += "\t</IncomingWeights>\n";
                    }
                    line += "\t</Neuron>\n";
                }
                line += "</Layer>\n";
                lines.Add(line);
            }
            System.IO.File.WriteAllLines(filePath, lines);
        }

        /// <summary>The sigmoid/logistic function which serves as the 
        /// activation function for the current network.</summary>
        /// <param name="x">The input to this function.</param>
        /// <returns>The output to the sigmoid/logistic function for the given 
        /// input.</returns>
        private static float sigmoid(float x)
        {
            return (1f / (1f + (float) Math.Pow(Math.E, -x)));
        }

        /// <summary>The derivative of the sigmoid/logistic function.</summary>
        /// <param name="x">The input to evaluate the derivative of the 
        /// sigmoid/logistic function at.</param>
        /// <returns>The value of the derivative of the sigmoid/logistic 
        /// function evaluated at the given input.</returns>
        private static float derivativeOfSigmoid(float x)
        {
            float s = sigmoid(x);
            return s * (1 - s);
        }

        /// <summary>Returns a <see cref="System.String" /> that represents 
        /// the current network.</summary>
        /// <returns>A <see cref="System.String" /> that represents the 
        /// current matrix.</returns>
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
