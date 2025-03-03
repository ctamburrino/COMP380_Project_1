import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class FileParser {
    static String trainingDataFileName = "proj1/Sample Training Dataset.txt";
    static String testingDataFileName = "proj1/Sample Testing Dataset-1.txt";

    static int inputRows;
    static int inputColumns;
    static int outputDimensions;
    static int numSamples;

    static List<DataSample> dataset = new ArrayList<>();

    public static List<DataSample> parseTrainingFile(String trainingDataFileName){

        try (BufferedReader reader = new BufferedReader(new FileReader(trainingDataFileName))){
            String line;
            for (int i = 0;i < 4; i++){
                line = reader.readLine();
                String[] parts = line.trim().split("\\s+");
                int value = Integer.parseInt(parts[0]);

                switch(i) {
                    case 0: inputRows = value;
                    case 1: inputColumns = value;
                    case 2: outputDimensions = value;
                    case 3: numSamples = value;
                }
            }

            for (int i = 0; i < numSamples; i++){
                int[] pixelArray = new int[inputRows * inputColumns];
                
                // Consume blank line
                reader.readLine();
                
                int pixelArrayIndex = 0;
                for (int j = 0; j < inputRows; j++){
                    line = reader.readLine();
                    String[] parts = line.trim().split("\\s+");
                    for (int k = 0; k < inputColumns; k ++){
                        pixelArray[pixelArrayIndex] = (Integer.parseInt(parts[k]));
                        pixelArrayIndex++;
                    }
                }
                // Consume blank line
                reader.readLine();
                String[] parts = reader.readLine().trim().split("\\s+");
                int[] outputVector = new int[outputDimensions];
                int outputVectorIndex = 0;
                for (int k = 0; k < outputDimensions; k++){
                    outputVector[outputVectorIndex] = Integer.parseInt(parts[k]);
                    outputVectorIndex++;
                }
                char label = reader.readLine().charAt(0);
                
            DataSample newDataSample = createDataSample(inputRows, inputColumns, outputDimensions);
            newDataSample.setPixelArray(pixelArray);
            newDataSample.setOutputVector(outputVector);
            newDataSample.setLabel(label);
            dataset.add(newDataSample);
            }
            return dataset;

        }catch (IOException e) {
            System.out.println("Error reading file: " + e.getMessage());
            return null;
        } 
    }

    
    public static DataSample createDataSample(int rows, int columns, int outputDimension){
        DataSample newDataSample = new DataSample (rows, columns, outputDimension);
        return newDataSample;
    }

    public static void parseTrainedWeights(TestingSettings netTestingSettings){
        String trainedWeightsFileName = "proj1/ " + netTestingSettings.trainedWeightsFilePath + ".txt";
        

        try (BufferedReader reader = new BufferedReader(new FileReader(trainedWeightsFileName))){
            String line;
            line = reader.readLine();
            String[] parts = line.trim().split("\\s+");
            int numInputNodes = Integer.parseInt(parts[0]);

            line = reader.readLine();
            parts = line.trim().split("\\s+");
            int numOutputNodes = Integer.parseInt(parts[0]);

            reader.readLine();
            double[][] weightMatrix = new double[numInputNodes][numOutputNodes];
            double[] biasWeights = new double[numOutputNodes];

            for (int rowNum = 0; rowNum < numInputNodes; rowNum++){
                for (int columnNum = 0; columnNum < numOutputNodes; columnNum++){
                    line = reader.readLine();
                    for (int j = 0; j < inputRows; j++){
                        parts = line.trim().split("\\s+");
                        for (int k = 0; k < inputColumns; k ++){
                            weightMatrix[rowNum][columnNum] = Double.parseDouble(parts[columnNum]);
                        }
                    }
                }
            }

            netTestingSettings.weightMatrix = weightMatrix;

            reader.readLine();
            line = reader.readLine();
            parts = line.trim().split("\\s+");
            for (int columnNum = 0; columnNum < numOutputNodes; columnNum++){
                biasWeights[columnNum] = Double.parseDouble(parts[columnNum]);
            }

            netTestingSettings.biasWeights = biasWeights;
        } catch (IOException e) {
            System.out.println("Error reading file: " + e.getMessage());
        } 
    }
}

