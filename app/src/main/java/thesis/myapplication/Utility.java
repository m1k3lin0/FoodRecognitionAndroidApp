package thesis.myapplication;

import android.content.Context;
import android.content.SharedPreferences;
import android.os.Environment;
import android.widget.TextView;

import junit.framework.Assert;

import org.apache.commons.lang3.text.WordUtils;
import org.w3c.dom.Text;

/*
* Class containing all the parameters
* */
public class Utility {

    //Permission parameters
    public static final int PERMISSION_REQUEST = 1995;

    //SharedPreferences parameters
    public static final String MODE = "cnn_mode";
    public static final String CORRECT_CLASSIFICATIONS = "correct_top5";
    public static final String TOTAL_CLASSIFICATIONS = "total";
    public static final String TOTAL_TOP1_CLASSIFICATIONS = "total_top1";
    public static final String CORRECT_TOP1_CLASSIFICATION = "correct_top1";
    public static final String SCORES_SUM = "score_sum";

    //Intent parameters
    public static int PHOTO_CODE = 100;
    public static int CROP_CODE = 101;
    public static String LABEL_EXTRA = "label";
    public static String BROADCAST_ACTION = "broadcast_action";
    public static String BITMAP_EXTRA = "bitmap_extra";
    public static String SCORES_EXTRA = "scores_extra";
    public static String LABELS_EXTRA = "labels_extra";
    public static String TOP1_CALL = "top1_call";

    //Connection parameters
    public static String IP = "foodrecognition.sytes.net";
    public static String URL = "http://"+IP+":8080/FoodRecognitionServices/services/IRServices/recognize";

    //CNN parameters
    public static String[] labels = {"apple pie","baby back ribs","baklava","beef carpaccio","beef tartare","beet salad","beignets","bibimbap","bread pudding","breakfast burrito","bruschetta","caesar salad","cannoli","caprese salad","carrot cake","ceviche","cheese plate","cheesecake","chicken curry","chicken quesadilla","chicken wings","chocolate cake","chocolate mousse","churros","clam chowder","club sandwich","crab cakes","creme brulee","croque madame","cup cakes","deviled eggs","donuts","dumplings","edamame","eggs benedict","escargots","falafel","filet mignon","fish and chips","foie gras","french fries","french onion soup","french toast","fried calamari","fried rice","frozen yogurt","garlic bread","gnocchi","greek salad","grilled cheese sandwich","grilled salmon","guacamole","gyoza","hamburger","hot and sour soup","hot dog","huevos rancheros","hummus","ice cream","lasagna","lobster bisque","lobster roll sandwich","macaroni and cheese","macarons","miso soup","mussels","nachos","omelette","onion rings","oysters","pad thai","paella","pancakes","panna cotta","peking duck","pho","pizza","pork chop","poutine","prime rib","pulled pork sandwich","ramen","ravioli","red velvet cake","risotto","samosa","sashimi","scallops","seaweed salad","shrimp and grits","spaghetti bolognese","spaghetti carbonara","spring rolls","steak","strawberry shortcake","sushi","tacos","takoyaki","tiramisu","tuna tartare","waffles"};

    //Phone storage parameters
    public static String TEMP_DIR_PATH = Environment.getExternalStorageDirectory().getAbsolutePath() + "/FoodRecognition";
    public static String TEMP_IMAGE_NAME = "img.png";
    public static String TEMP_CROP_NAME = "crop.png";

    //Thresholds
    public static double TOP1_THRESHOLD = 0.9785;         //to show the top1
    public static double TOP5_THRESHOLD = 0;              //to show the top5 //

    public static int getDrawable(Context context, String name) {
        Assert.assertNotNull(context);
        Assert.assertNotNull(name);

        return context.getResources().getIdentifier(name, "drawable", context.getPackageName());
    }

    //all res_id
    public static Integer[] initializeImagesResId(Context context){
        Integer[] images = new Integer[101];
        for (int i=0;i<101;i++){
            images[i]=getDrawable(context.getApplicationContext(),labels[i].replace(" ","_"));
        }
        return images;
    }

    public static String[] capitalizeLabels(){
        String[] capLabels = new String[101];

        for (int i=0;i<101;i++){
            capLabels[i] = WordUtils.capitalize(labels[i]);
        }

        return capLabels;
    }

    /*
    * Returns the topK predictions given the array of scores
    * */
    public static RecognizedClass[] getTopKPredictions(double[] result, int k){
        RecognizedClass[] topKPredictions = new RecognizedClass[k];
        for (int j=0;j<k;j++){
            int maxIndex = 0;
            for (int i = 1;i < result.length;i++){
                double newnumber = result[i];
                if ((newnumber> result[maxIndex])){
                    maxIndex = i;
                }
            }
            topKPredictions[j] = new RecognizedClass(Utility.labels[maxIndex], result[maxIndex]);
            result[maxIndex] = 0;
        }

        return topKPredictions;
    }

    /*
    * Return labels array
    * */
    public static String[] getLabels(RecognizedClass[] array){
        String[] labels = new String[array.length];
        for (int i = 0; i<array.length; i++ ){
            labels[i] = array[i].getLabel();
        }
        return labels;
    }

    /*
    * Return labels array
    * */
    public static double[] getScores(RecognizedClass[] array){
        double[] scores = new double[array.length];
        for (int i = 0; i<array.length; i++ ){
            scores[i] = array[i].getScore();
        }
        return scores;
    }

    /*
    * Increment int in SharedPreferences of the specified amount and returns the value
    * */
    public static int incrementSharedPrefInt(SharedPreferences sharedPref, SharedPreferences.Editor editor, String name, int amount){
        int value = sharedPref.getInt(name,0);
        value = value + amount;
        editor.putInt(name,value);
        editor.commit();
        return value;
    }

    /*
    * Increment int in SharedPreferences of the specified amount and returns the value
    * */
    public static double incrementSharedPrefFloat (SharedPreferences sharedPref, SharedPreferences.Editor editor, String name, double amount){
        double value = sharedPref.getFloat(name,0.0f);
        value = value + amount;
        editor.putFloat(name, (float)value);
        editor.commit();
        return value;
    }
}
