import java.util.List;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

public class NeuralNet {
        /*
        This function fills a list with values to initialize weights. If setWeightsToZero parameter is true
        fills with zeros, else fills with random values in range -0.5 to 0.5. This would be
        applied to lists such as the weights of each node, or the weights of the b node.

        Parameters:
        - boolean setWeightsToZero: if the weights will be set to zero or not
        - int[] weights: set of weights to be set to 0
        */
    public static int train(TrainingSettings netTrainingSettings){
        // Get dataset
        List<DataSample> dataset = netTrainingSettings.dataset;

        // Create net architecture from first data sample in dataset
        DataSample firstSample = dataset.get(0);
        int numInputNodes = firstSample.getRowDimension() * firstSample.getColumnDimension();
        int numOutputNodes = firstSample.getOutputDimension();

        // Weight Matrices initialized with zero values by default
        double[][] weightMatrix = new double[numInputNodes][numOutputNodes];
        double[] biasWeights = new double[numOutputNodes];

        // Create training variables
        double learningRate = netTrainingSettings.learningRate;
        double thetaThreshold = netTrainingSettings.thetaThreshold;

        // Set weights to random values if selected by user
        if (!netTrainingSettings.setWeightsToZero){
            // Initialize bias weights
            initializeWeightsRandomValues(biasWeights);

            // Initialize node weights
            for (int i = 0; i < numInputNodes; i++){
                initializeWeightsRandomValues(biasWeights);
            }
        }

        // Perform training algorithm
        double[] yIn = new double[numOutputNodes];
        double[] yOut = new double[numOutputNodes];
        boolean converged = false;
        int epochNum = 0;
        while (!converged && epochNum < netTrainingSettings.maxEpochs){
            epochNum++;
            boolean weightChanged = false;
            for (DataSample sample : dataset){
                int[] inputSignals = sample.getPixelArray();
                int[] targetOutputs = sample.getOutputVector();
                for (int outputNode = 0; outputNode < numOutputNodes; outputNode++) {
                    yIn[outputNode] = calculateYIn(weightMatrix, biasWeights, inputSignals, outputNode);
                    yOut[outputNode] = applyActivationFunction(yIn[outputNode], thetaThreshold);

                    if (yOut[outputNode] != targetOutputs[outputNode]){
                        updateWeights(weightMatrix, biasWeights, inputSignals, targetOutputs, learningRate, outputNode);
                        weightChanged = true;
                    }
                }
            }
            if (weightChanged == false){
                converged = true;
            }
        }
        // If epochNum stopped while loop
        if (!converged){
            System.out.println("Training reached max epochs: " + netTrainingSettings.maxEpochs + "  before converging");
        }
        saveWeightsToFile(weightMatrix, biasWeights, netTrainingSettings.trainedWeightsFile);
        return epochNum;
    }

    public static void initializeWeightsRandomValues(double[] weights) {
        /*
        This function fills a list with random values in range -0.5 to 0.5 to initialize weights. This would be
        applied to lists such as the weights of each node, or the weights of the b node.

        Parameters:
        - int[] weights: set of weights to be set to 0
        */
        for (int i = 0; i < weights.length; i++) {
            weights[i] = (double) (Math.random() - 0.5);
        }
    }

    public static double calculateYIn(double[][] weightMatrix, double[] biasWeights, int[] inputSignals, int outputNode) {
        /*
        This method calculates the y in value for the corresponding pattern.


        Parameters:
        - double[][] weightMatrix: Matrix of current weight values
        - double[] biasWeights: Array of current bias weight values
        - int[] inputSignals: pixels of the current sample
        - int outputNode: current outputNode being trained for
        */
        double computedYIn = biasWeights[outputNode];
        for (int i = 0; i < inputSignals.length; i++) {
            computedYIn += inputSignals[i] * weightMatrix[i][outputNode];
        }
        return computedYIn;
    }


    public static int applyActivationFunction(double yIn, double thetaThreshold) {
        /*
        This method takes in the yIn value and runs it into a bipolar activation function


        Parameters:
        - int yIn: value to be taken in and converted
        */
        if (yIn > thetaThreshold) {
            return 1;
        } else if(yIn < thetaThreshold){
            return -1;
        } else {
            return 0;
        }
    }


    public static void updateWeights(double[][] weightMatrix, double[] biasWeights, int[] inputSignals, int[] targetOutputs, double learningRate, int outputNode ) {
        /*
        This method changes the weights if the yOut value does not match the target value for
        the corresponding pattern. It then updates the weights to get closer to converging


        Parameters:
        - int[][] _net: overarching net to access weights
        - int[] _weightB: overaching b node weights
        - int[] _x: pixels of the current sample
        - int[] _t: target values of the current sample
        - int currentPattern: current pattern being trained for (Ex. A, B, C)
        */
        for (int i = 0; i < inputSignals.length; i++) {
            weightMatrix[i][outputNode] += (learningRate * targetOutputs[outputNode] * inputSignals[i]);
        }
        biasWeights[outputNode] += (learningRate * targetOutputs[outputNode]);
    }

    public static void saveWeightsToFile(double[][] weightMatrix, double[]biasWeights, String trainedWeightsFileName){
        // Save Node weights
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(trainedWeightsFileName))) {
            writer.write(weightMatrix.length + "\t\t// Number of input nodes\n");
            writer.write(weightMatrix[0].length + "\t\t//Number of output nodes\n\n");

            for (double[] row : weightMatrix){
                for (int j = 0; j < row.length; j++){
                    writer.write(Double.toString(row[j]));
                    if (j < row.length - 1) writer.write(" ");
                }
                writer.newLine();
            }
            writer.newLine();

            // Save bias weights
            for (int j = 0; j< biasWeights.length; j++){
                writer.write(Double.toString(biasWeights[j]));
                if (j < biasWeights.length - 1) writer.write(" ");
            }
            System.out.println("Weights saved successfully to " + trainedWeightsFileName);
        } catch (IOException e){
            e.printStackTrace();
        }
    }

    public static void test(TestingSettings netTestingSettings){
        // Load trained weight matrices from file
    }


}





