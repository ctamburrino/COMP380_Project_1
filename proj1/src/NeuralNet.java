import java.util.List;

public class NeuralNet {
    
    public static int train(TrainingSettings netTrainingSettings){
        // Get dataset
        List<DataSample> dataset = FileParser.parseTrainingFile(netTrainingSettings.trainingDataFilePath);

        // Create net architecture from first data sample in dataset
        DataSample firstSample = dataset.get(0);
        int numInputNodes = firstSample.getRowDimension() * firstSample.getColumnDimension();
        int numOutputNodes = firstSample.getOutputDimension();
        double[][] weightMatrix = new double[numInputNodes][numOutputNodes];
        double[] biasWeights = new double[numOutputNodes];

        // Create training variables
        double learningRate = netTrainingSettings.learningRate;
        double thetaThreshold = netTrainingSettings.thetaThreshold;

        // Initialize bias weights
        initializeWeights(biasWeights, netTrainingSettings.setWeightsToZero);

        // Initialize node weights
        for (int i = 0; i < numInputNodes; i++){
            initializeWeights(weightMatrix[i], netTrainingSettings.setWeightsToZero);
        }

        // Perform training algorithm
        double[] yIn = new double[numOutputNodes];
        double[] yOut = new double[numOutputNodes];
        boolean converged = false;
        int epochNum = 0;
        while (!converged){
            epochNum += 1;
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
        return epochNum;
    }

    public static void initializeWeights(double[] weights, boolean setWeightsToZero) {
        /*
        This function fills a list with values to initialize weights. If setWeightsToZero parameter is true
        fills with zeros, else fills with random values in range -0.5 to 0.5. This would be
        applied to lists such as the weights of each node, or the weights of the b node.

        Parameters:
        - boolean setWeightsToZero: if the weights will be set to zero or not
        - int[] weights: set of weights to be set to 0
        */
        if (setWeightsToZero) {
            for (int i = 0; i < weights.length; i++) {
                weights[i] = 0;
            }
        }
        else {
            for (int i = 0; i < weights.length; i++) {
                weights[i] = (double) (Math.random() - 0.5);
            }
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
}





