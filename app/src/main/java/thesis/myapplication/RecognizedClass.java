package thesis.myapplication;

public class RecognizedClass implements Comparable<RecognizedClass>{

    private String label; //recognized label
    private double score; //score of the label

    public RecognizedClass(String label, double score) {
        this.label = label;
        this.score = score;
    }

    public String getLabel() {
        return label;
    }
    public double getScore() {
        return score;
    }

    @Override
    public String toString(){
        return label+" - "+(int)(score*100)+"%";
    }
    @Override
    public int compareTo(RecognizedClass o) {
        return - new Double(score).compareTo(o.score);
    }
}
