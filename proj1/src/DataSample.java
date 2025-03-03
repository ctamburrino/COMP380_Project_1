public class DataSample {
    private int rowDimension;
    private int columnDimension;
    private int outputDimension;

    private char label;
    private int[] pixelArray;
    private static int[] outputVector;


    public DataSample(int rows, int columns, int outputDimension){
        this.rowDimension = rows;
        this.columnDimension = columns;
        this.outputDimension = outputDimension;
    }

    public char getLabel(){
        return label;
    }

    public int[] getPixelArray(){
        return pixelArray;
    }

    public int getRowDimension(){
        return rowDimension;
    }

    public int getColumnDimension(){
        return columnDimension;
    }

    public int getOutputDimension(){
        return outputDimension;
    }

    public int[] getOutputVector(){
        return outputVector;
    }

    public void setPixelArray(int[] pixelArray){
        this.pixelArray = pixelArray;
    }

    public void setLabel(char letter){
        this.label = letter;
    }
}
