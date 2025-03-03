import java.util.List;

public class TrainingSettings {
    String trainingDataFilePath;
    boolean setWeightsToZero;
    int maxEpochs;
    String trainedWeightsFile;
    double learningRate;
    double thetaThreshold;
    double weightChangeThreshold;
    List<DataSample> dataset;
}
