public class NeuralNet {
    //push test

    public static void weightInitializer(int[] weights) {
        /*
        This function fills a list with 7 zeroes to initialize it's values. This would be
        applied to lists such as the weights of each node, or the weights of the b node.


        Parameters:
        - int[] weights: set of weights to be set to 0
        */
        for (int i = 0; i < weights.length; i++) {
            weights[i] = 0;
        }
    }


    public static int yInCalculation(int[][] _net, int[] _weightB, int[] _x, int currentPattern) {
        /*
        This method calculates the y in value for the corresponding pattern.


        Parameters:
        - int[][] _net: overarching net to access weights
        - int[] _weightB: overaching b node weights
        - int[] _x: pixels of the current sample
        - int currentPattern: current pattern being trained for (Ex. A, B, C)
        */
        int computedYIn = _weightB[currentPattern];
        for (int i = 0; i < 63; i++) {
            computedYIn += _x[i] * _net[i][currentPattern];
        }


        return computedYIn;


    }


    public static int activationFunction(int _yIn) {
        /*
        This method takes in the yIn value and runs it into a bipolar activation function


        Parameters:
        - int _yIn: value to be taken in and converted
        */
        if (_yIn >= 0) {
            return 1;
        }
        return -1;
    }


    public static void weightUpdating(int[][] _net, int[] _weightB, int[] _x, int[] _t, int a, int currentPattern ) {
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
        for (int i = 0; i < 63; i++) {
            _net[i][currentPattern] += (a * _t[currentPattern] * _x[i]);
        }
        _weightB[currentPattern] += (a * _t[currentPattern]);
    }






    public static void main(String[] args){
        ///FileParser.parseFile();


        //initialize weights
        int n = 63;
        int patterns = 7;
        int fillerA = 1;
        int[][] net = new int[n][patterns];
        int[] weightB = new int[patterns];
        weightInitializer(weightB);
        for (int i = 0; i < n; i++){
            weightInitializer(net[i]);
        }


        boolean converged = false;
        int[] yIn = new int[patterns];
        int[] yOut = new int[patterns];
        weightInitializer(yIn);
        weightInitializer(yOut);
        while (!converged){
            converged = true;
            for (DataSample sample : FileParser.dataset) {
                int[] x = sample.getPixelArray();
                int[] t = sample.getOutputVector();
                for (int i = 0; i < patterns; i++) {
                    yIn[i] = yInCalculation(net, weightB, x, i);
                    yOut[i] = activationFunction(yIn[i]);


                    if (yOut[i] != t[i]) {
                        converged = false;
                        weightUpdating(net, weightB, x, t, fillerA, i);
                    }
                }
                   
            }
        }
        UserIO.welcomeToPerceptron();
    }


       
}





