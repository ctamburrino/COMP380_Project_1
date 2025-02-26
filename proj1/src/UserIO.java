import java.util.Scanner;
import java.io.FileReader;
import java.io.File;

public class UserIO {

    public static void welcomeToPerceptron(){
        getUserIntSelection();
    }

    static Scanner scanner = new Scanner(System.in);
    static TrainingSettings netTrainingSettings = new TrainingSettings();

    public static void getUserIntSelection(){
        int choice;
        while(true) {
            System.out.println("Welcome to our first neural network - A Perceptron Net!");
            System.out.println("1) Enter 1 to train the net on a data file");
            System.out.println("2) Enter 2 to test the net on a data file");
            System.out.println("3) Enter 3 to quit");
            if (scanner.hasNextInt()){
                choice = scanner.nextInt();
                scanner.nextLine();
                switch(choice){
                case 1:
                    getTrainingSettings();
                    return;
                case 2:
                    getTestingSettings();
                    return;
                case 3:
                    scanner.close();
                    return;
                default:
                    System.out.println("Invalid selection, please try again!");
                }
            } else {
                System.out.println("\nPlease enter a valid number!\n");
                scanner.next();
            }
        }
    }


    public static void getTrainingSettings(){
        // Get training data file name
        File file = new File("");
        String trainingFilePath;
        System.out.println("\nEnter the training file name: ");
        do{
            trainingFilePath = "proj1/" + scanner.nextLine().trim() + ".txt";
            file = new File(trainingFilePath);

            if(!file.exists()){
                System.out.println("\nCould not find file, please try again!\n");
                System.out.println("Enter the training file name: ");
            }
        } while(file.exists() == false);
        netTrainingSettings.trainingDataFilePath = trainingFilePath;

        // Get weight initialization selection
        String weightPrompt = "\nEnter 0 to initialize weights to 0, enter 1 to initialize weights to random values between -0.5 and 0.5:";
        int weightChoice = getIntInput(weightPrompt, 0, 1);
        netTrainingSettings.setWeightsToZero = (weightChoice == 0);

        // Get maximum epochs
        String epochPrompt = "\nEnter the maximum number of training epochs:";
        int epochChoice = getIntInput(epochPrompt, 1, 10000);
        netTrainingSettings.maxEpochs = epochChoice;

        // Get file name to save trained weights to
        String trainedWeightOutputFile = getValidFilename("\nEnter a file name to save the trained weight values:");
        netTrainingSettings.trainedWeightsFile = trainedWeightOutputFile;

        // Get learning rate (alpha)
        String alphaPrompt = "\nEnter the learning rate alpha from 0 to 1 but not including 0:";
        double alphaChoice = getDoubleInput(alphaPrompt, 0.1, 1.0);
        netTrainingSettings.learningRate = alphaChoice;

        // Get threshold (theta)
        String thetaPrompt = "\nEnter the threshold theta:";
        double thetaChoice = getDoubleInput(thetaPrompt, -10000.0, 10000.0);
        netTrainingSettings.thetaThreshold = thetaChoice;

        // Get weight change threshold
        String weightThresholdPrompt = "\nEnter the threshold to be used for measuring weight changes:";
        double weightThresholdChoice = getDoubleInput(weightThresholdPrompt, 0.0000001, 10000000.0);
        netTrainingSettings.weightChangeThreshold = weightThresholdChoice;
    }

    private static int getIntInput(String prompt, int min, int max) {
        int input;
        while (true) {
            System.out.println(prompt);
            if (scanner.hasNextInt()) {
                input = scanner.nextInt();
                scanner.nextLine();  // Consume the newline
                if (input >= min && input <= max) {
                    return input;  // Return the valid input
                } else {
                    System.out.println("Please enter a number between " + min + " and " + max + ".");
                }
            } else {
                System.out.println("Invalid input, please enter a valid integer.");
                scanner.next();  // Consume the invalid input
            }
        }
    }

    private static String getValidFilename(String prompt){
        String filename;
        do {
            System.out.println(prompt);
            filename = scanner.nextLine().trim();
            if (!isValidFilename(filename)) {
                System.out.println("Invalid filename, Please try again!");
            }
        } while (!isValidFilename(filename));
        return filename;
    }

    private static boolean isValidFilename(String filename){
        if (filename == null || filename.isEmpty()) return false;
        if (!filename.matches("^[^/]*$")) return false;
        if (filename.equals(".") || filename.equals("..")) return false;
        if (filename.length() > 255) return false;
        return true;
    }

    private static double getDoubleInput(String prompt, double min, double max){
        double input;
        while (true){
            System.out.println(prompt);
            if(scanner.hasNextFloat()){
                input = scanner.nextFloat();
                scanner.nextLine(); // Consume new line
                if(input > min && input <= max){
                    return input;
                } else{
                    System.out.println("Please enter a number between " + min + " and " + max + ".");
                }
            } else {
                System.out.println("Invalid input, please enter a valid input.");
                scanner.next();  // Consume the invalid input
            }
        }
    }


    public static void getTestingSettings(){
        
    }
}
