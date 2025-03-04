import java.util.List;

public class TestingSettings {
    String trainedWeightsFilePath;
    String testingDataFilePath;
    String testingResultsOutputFilePath;
    double[][] trainedWeightMatrix;
    double[] trainedBiasWeights;
    List<DataSample> dataset;
    double thetaThreshold;
}
