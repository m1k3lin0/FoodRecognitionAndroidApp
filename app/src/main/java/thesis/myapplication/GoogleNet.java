package thesis.myapplication;

import android.content.Context;
import android.content.Intent;
import android.content.res.Resources;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Environment;
import android.renderscript.Allocation;
import android.renderscript.Element;
import android.renderscript.RenderScript;
import android.renderscript.Type;
import android.util.Log;

import com.example.mmota.squeezenet_dse.ScriptC_convNet;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;

import static android.graphics.Color.blue;
import static android.graphics.Color.green;
import static android.graphics.Color.red;

public class GoogleNet {

    private static final boolean debug = false;
    private static final boolean result = true;
    private static final boolean logEnable = false;
    private boolean uglyLoop = true;

    private float[] img;

    private float[] meanImg;

    private float[] conv1_w;
    private float[] conv1_b;

    private float[] conv2_3x3_reduce_w;
    private float[] conv2_3x3_reduce_b;

    private float[] conv2_3x3_w;
    private float[] conv2_3x3_b;

    private float[] inception_3a_1x1_w;
    private float[] inception_3a_1x1_b;

    private float[] inception_3a_3x3_reduce_w;
    private float[] inception_3a_3x3_reduce_b;

    private float[] inception_3a_5x5_reduce_w;
    private float[] inception_3a_5x5_reduce_b;

    private float[] inception_3a_3x3_w;
    private float[] inception_3a_3x3_b;

    private float[] inception_3a_5x5_w;
    private float[] inception_3a_5x5_b;

    private float[] inception_3a_pool_proj_w;
    private float[] inception_3a_pool_proj_b;

    private float[] inception_3b_1x1_w;
    private float[] inception_3b_1x1_b;

    private float[] inception_3b_3x3_reduce_w;
    private float[] inception_3b_3x3_reduce_b;

    private float[] inception_3b_5x5_reduce_w;
    private float[] inception_3b_5x5_reduce_b;

    private float[] inception_3b_3x3_w;
    private float[] inception_3b_3x3_b;

    private float[] inception_3b_5x5_w;
    private float[] inception_3b_5x5_b;

    private float[] inception_3b_pool_proj_w;
    private float[] inception_3b_pool_proj_b;

    private float[] inception_4a_1x1_w;
    private float[] inception_4a_1x1_b;

    private float[] inception_4a_3x3_reduce_w;
    private float[] inception_4a_3x3_reduce_b;

    private float[] inception_4a_5x5_reduce_w;
    private float[] inception_4a_5x5_reduce_b;

    private float[] inception_4a_3x3_w;
    private float[] inception_4a_3x3_b;

    private float[] inception_4a_5x5_w;
    private float[] inception_4a_5x5_b;

    private float[] inception_4a_pool_proj_w;
    private float[] inception_4a_pool_proj_b;

    private float[] inception_4b_1x1_w;
    private float[] inception_4b_1x1_b;

    private float[] inception_4b_3x3_reduce_w;
    private float[] inception_4b_3x3_reduce_b;

    private float[] inception_4b_5x5_reduce_w;
    private float[] inception_4b_5x5_reduce_b;

    private float[] inception_4b_3x3_w;
    private float[] inception_4b_3x3_b;

    private float[] inception_4b_5x5_w;
    private float[] inception_4b_5x5_b;

    private float[] inception_4b_pool_proj_w;
    private float[] inception_4b_pool_proj_b;

    private float[] inception_4c_1x1_w;
    private float[] inception_4c_1x1_b;

    private float[] inception_4c_3x3_reduce_w;
    private float[] inception_4c_3x3_reduce_b;

    private float[] inception_4c_5x5_reduce_w;
    private float[] inception_4c_5x5_reduce_b;

    private float[] inception_4c_3x3_w;
    private float[] inception_4c_3x3_b;

    private float[] inception_4c_5x5_w;
    private float[] inception_4c_5x5_b;

    private float[] inception_4c_pool_proj_w;
    private float[] inception_4c_pool_proj_b;

    private float[] inception_4d_1x1_w;
    private float[] inception_4d_1x1_b;

    private float[] inception_4d_3x3_reduce_w;
    private float[] inception_4d_3x3_reduce_b;

    private float[] inception_4d_5x5_reduce_w;
    private float[] inception_4d_5x5_reduce_b;

    private float[] inception_4d_3x3_w;
    private float[] inception_4d_3x3_b;

    private float[] inception_4d_5x5_w;
    private float[] inception_4d_5x5_b;

    private float[] inception_4d_pool_proj_w;
    private float[] inception_4d_pool_proj_b;

    private float[] inception_4e_1x1_w;
    private float[] inception_4e_1x1_b;

    private float[] inception_4e_3x3_reduce_w;
    private float[] inception_4e_3x3_reduce_b;

    private float[] inception_4e_5x5_reduce_w;
    private float[] inception_4e_5x5_reduce_b;

    private float[] inception_4e_3x3_w;
    private float[] inception_4e_3x3_b;

    private float[] inception_4e_5x5_w;
    private float[] inception_4e_5x5_b;

    private float[] inception_4e_pool_proj_w;
    private float[] inception_4e_pool_proj_b;

    private float[] inception_5a_1x1_w;
    private float[] inception_5a_1x1_b;

    private float[] inception_5a_3x3_reduce_w;
    private float[] inception_5a_3x3_reduce_b;

    private float[] inception_5a_5x5_reduce_w;
    private float[] inception_5a_5x5_reduce_b;

    private float[] inception_5a_3x3_w;
    private float[] inception_5a_3x3_b;

    private float[] inception_5a_5x5_w;
    private float[] inception_5a_5x5_b;

    private float[] inception_5a_pool_proj_w;
    private float[] inception_5a_pool_proj_b;

    private float[] inception_5b_1x1_w;
    private float[] inception_5b_1x1_b;

    private float[] inception_5b_3x3_reduce_w;
    private float[] inception_5b_3x3_reduce_b;

    private float[] inception_5b_5x5_reduce_w;
    private float[] inception_5b_5x5_reduce_b;

    private float[] inception_5b_3x3_w;
    private float[] inception_5b_3x3_b;

    private float[] inception_5b_5x5_w;
    private float[] inception_5b_5x5_b;

    private float[] inception_5b_pool_proj_w;
    private float[] inception_5b_pool_proj_b;

    private float[] loss3_classifier_w;
    private float[] loss3_classifier_b;

    private static final String tag = "GoogLeNet";

    private RecognizedClass[] predictions;  //to store the result of the net

    Context context;
    /*
    * Constructor, initializes the float arrays with the weights
    * */
    public GoogleNet(){
        //loadParameters("GoogleNet/Logs", "parallel.txt", "/GoogleNet/Parameters/Vectorized", logEnable);
    }

    /*
    * Load the image into the float array. The image is obtained as follow: Img = Image - meanImg
    * */
    public void loadImage(Bitmap image){
        //convert the Bitmap image into a float array
        float[] inputImg = new float[227*227*3];
        img = new float[227*227*3];
        bitmapToFloatArray(image,inputImg);
        //load into the float array: Img = Image - Mean
        for (int i=0;i<227*227*3;i++){
            img[i]=inputImg[i]-meanImg[i];
        }
    }

    /*
    * Parallel implementation of the GoogLeNet
    * */
    public void parGoogleNet(Context context) throws InterruptedException {
        Log.i(tag, "Start");
        RenderScript rs = RenderScript.create(context);
        ScriptC_convNet convNet = new ScriptC_convNet(rs);

        Type.Builder imgType = new Type.Builder(rs, Element.F32_4(rs)).setX(227 * 227);
        Allocation imgAllocation = Allocation.createTyped(rs, imgType.create());
        paraReshape(rs, convNet, imgAllocation);

        /***************************************************************************/
        /* Conv 1 */
        int Wout = 114;
        int Hout = 114;
        int Win = 227;
        int Hin = 227;
        int numOutLayer = 64;
        int kernelSz = 7;

        Allocation conv1Allocation = Allocation.createSized(rs, Element.F32(rs), Wout * Hout * numOutLayer); //Output size

        int[] conv1TypeSet = {4};//{1, 2, 4, 8, 16};
        for (int convType : conv1TypeSet) {
            int parallelOFMs = numOutLayer / convType;
            paraConvRelu(rs, convNet, /*Weight size F4*/kernelSz * kernelSz * numOutLayer, /*bias size*/numOutLayer, conv1_w,
                    conv1_b, Wout * Hout * parallelOFMs, kernelSz, Wout, Hout, Win, Hin,/*#Out Layers*/ numOutLayer,
                    /*#In Layers*/ 3, /*S*/ 2, /*pad*/3,/*N_new*/ 1, /*offset*/0, imgAllocation,
                    conv1Allocation, convType, parallelOFMs, "GoogleNet/Logs",
                    String.valueOf(convType) + ".txt", "Conv1 (ms): ", logEnable);
            if (debug) {
                float[] tmp = new float[Wout * Hout * numOutLayer];
                conv1Allocation.copyTo(tmp);
                binaryDumper("GoogleNet/Intermediate_rslts/Parallel/" + String.valueOf(convType), "conv1.bin", tmp);
            }
        }
        Log.i(tag, "Conv1 finished.");


        /***************************************************************************/
        /* Pool 1 */
        Wout = 57;
        Hout = 57;
        Win = 114;
        Hin = 114;
        numOutLayer = 64;
        kernelSz = 3;
        Allocation pool1Allocation = Allocation.createSized(rs, Element.F32(rs), Wout * Hout * numOutLayer);
        paraMaxPool(rs, convNet, Wout * Hout * (numOutLayer / 4), kernelSz, Win, Hin, Wout, Hout, 2,
                conv1Allocation, pool1Allocation, "GoogleNet/Logs", "4.txt", "Pool1 (ms): ", logEnable);

        //ToDo: Move LRN to GPU. Then this data transfer will not be required.
        float[] pool1 = new float[Wout * Hout * numOutLayer];
        vectorizedToNormal(pool1Allocation, pool1, Wout, Hout, numOutLayer);
        if (debug) {
            //float[] pool1 = new float[Wout * Hout * numOutLayer];
            //pool1Allocation.copyTo(pool1);
            binaryDumper("GoogleNet/Intermediate_rslts/Parallel/4", "pool1.bin", pool1);
        }
        Log.i(tag, "Pool1 finished.");

        /***************************************************************************/
        /* lrn1 */

        Wout = 57;
        Hout = 57;
        //Win = 57;
        //Hin = 57;
        numOutLayer = 64;

        float[] lrn1 = new float[57 * 57 * 64];
        lrn(pool1, lrn1, 57, 57, 64, 5, 0.0001f, 0.75f, 1, "GoogleNet/Logs", "paraSeq.txt",
                "LRN1 (ms)", logEnable);
//        if (debug)
//            binaryDumper("GoogleNet/Intermediate_rslts/Parallel/Sequential", "lrn1.bin", lrn1);
//

        Type.Builder lrnType = new Type.Builder(rs, Element.F32_4(rs)).setX(Wout * Hout * (numOutLayer / 4));
        Allocation lrnAllocation = Allocation.createTyped(rs, lrnType.create());
        normalToVectorized(rs, convNet, lrn1, lrnAllocation, Wout, Hout, numOutLayer);
        if (debug) {
            lrnAllocation.copyTo(lrn1);
            binaryDumper("GoogleNet/Intermediate_rslts/Parallel/Sequential", "lrn1.bin", lrn1);
        }
        Log.i(tag, "LRN1 finished.");

        /***************************************************************************/
        /* conv2_3x3_reduce */
        Wout = 57;
        Hout = 57;
        Win = 57;
        Hin = 57;
        int numInLayer = 64;
        numOutLayer = 64;
        kernelSz = 1;
        Allocation conv2_3x3_reduceAllocation = Allocation.createSized(rs, Element.F32(rs), Wout * Hout * numOutLayer); //Output size
        int[] conv2_3x3_reduceTypeSet = {4};//{1, 2, 4, 8, 16};
        for (int convType : conv2_3x3_reduceTypeSet) {
            int parallelOFMs = numOutLayer / convType;
            paraConvRelu(rs, convNet, /*Weight size F4*/kernelSz * kernelSz * numInLayer * (numOutLayer / 4), /*bias size*/numOutLayer, conv2_3x3_reduce_w,
                    conv2_3x3_reduce_b, Wout * Hout * parallelOFMs, kernelSz, Wout, Hout, Win, Hin,/*#Out Layers*/ numOutLayer,
                    /*#In Layers*/ 64, /*S*/ 1, /*pad*/0,/*N_new*/ 16, /*offset*/0, lrnAllocation,
                    conv2_3x3_reduceAllocation, convType, parallelOFMs, "GoogleNet/Logs",
                    String.valueOf(convType) + ".txt", "conv2_3x3_reduce (ms): ", logEnable);
            if (debug) {
                float[] tmp = new float[Wout * Hout * numOutLayer];
                conv2_3x3_reduceAllocation.copyTo(tmp);
                binaryDumper("GoogleNet/Intermediate_rslts/Parallel/" + String.valueOf(convType),
                        "conv2_3x3_reduce.bin", tmp);
            }
        }
        Log.i(tag, "conv2_3x3_reduce finished.");

        /***************************************************************************/
        /* conv2_3x3 */
        Wout = 57;
        Hout = 57;
        Win = 57;
        Hin = 57;
        numInLayer = 64;
        numOutLayer = 192;
        kernelSz = 3;
        Allocation conv2_3x3_Allocation = Allocation.createSized(rs, Element.F32(rs), Wout * Hout * numOutLayer); //Output size
        int[] conv2_3x3_TypeSet = {6};//{1, 2, 4, 6, 8, 12, 16, 24, 48};
        for (int convType : conv2_3x3_TypeSet) {
            int parallelOFMs = numOutLayer / convType;
            paraConvRelu(rs, convNet, /*Weight size F4*/kernelSz * kernelSz * numInLayer * (numOutLayer / 4), /*bias size*/numOutLayer, conv2_3x3_w,
                    conv2_3x3_b, Wout * Hout * parallelOFMs, kernelSz, Wout, Hout, Win, Hin,/*#Out Layers*/ numOutLayer,
                    /*#In Layers*/ numInLayer, /*S*/ 1, /*pad*/1,/*N_new*/ numInLayer / 4, /*offset*/0, conv2_3x3_reduceAllocation,
                    conv2_3x3_Allocation, convType, parallelOFMs, "GoogleNet/Logs",
                    String.valueOf(convType) + ".txt", "conv2_3x3 (ms): ", logEnable);
            if (debug) {
                float[] tmp = new float[Wout * Hout * numOutLayer];
                conv2_3x3_Allocation.copyTo(tmp);
                binaryDumper("GoogleNet/Intermediate_rslts/Parallel/" + String.valueOf(convType),
                        "conv2_3x3.bin", tmp);
            }
        }
        Log.i(tag, "conv2_3x3 finished.");

        float[] conv2_3x3 = new float[Wout * Hout * numOutLayer];
        vectorizedToNormal(conv2_3x3_Allocation, conv2_3x3, Wout, Hout, numOutLayer);

        /***************************************************************************/
        /* lrn2 */

        Wout = 57;
        Hout = 57;
        numOutLayer = 192;

        float[] lrn2 = new float[Wout * Hout * numOutLayer];
        lrn(conv2_3x3, lrn2, Wout, Hout, numOutLayer, 5, 0.0001f, 0.75f, 1, "GoogleNet/Logs",
                "paraSeq.txt", "LRN2 (ms)", logEnable);

        Type.Builder lrn2Type = new Type.Builder(rs, Element.F32_4(rs)).setX(Wout * Hout * (numOutLayer / 4));
        Allocation lrn2Allocation = Allocation.createTyped(rs, lrn2Type.create());
        normalToVectorized(rs, convNet, lrn2, lrn2Allocation, Wout, Hout, numOutLayer);
        if (debug) {
            lrn2Allocation.copyTo(lrn2);
            binaryDumper("GoogleNet/Intermediate_rslts/Parallel/Sequential", "lrn2.bin", lrn2);
        }
        Log.i(tag, "LRN2 finished.");

        /***************************************************************************/
        /* Pool 2 */
        Wout = 28;
        Hout = 28;
        Win = 57;
        Hin = 57;
        numOutLayer = 192;
        kernelSz = 3;
        Allocation pool2_Allocation = Allocation.createSized(rs, Element.F32(rs), Wout * Hout * numOutLayer);
        paraMaxPool(rs, convNet, Wout * Hout * (numOutLayer / 4), kernelSz, Win, Hin, Wout, Hout, 2,
                lrn2Allocation, pool2_Allocation, "GoogleNet/Logs", "4.txt", "Pool2 (ms): ", logEnable);

        if (debug) {
            float[] pool2 = new float[Wout * Hout * numOutLayer];
            pool2_Allocation.copyTo(pool2);
            binaryDumper("GoogleNet/Intermediate_rslts/Parallel/4", "pool2.bin", pool2);
        }
        Log.i(tag, "Pool2 finished.");
        /***************************************************************************/


        /* output_3a */
        Allocation output_3a_Allocation = Allocation.createSized(rs, Element.F32(rs), 28 * 28 * 256); //Output size

        /***************************************************************************/
        /* inception_3a_3x3_reduce */
        Wout = 28;
        Hout = 28;
        Win = 28;
        Hin = 28;
        numInLayer = 192;
        numOutLayer = 96;
        kernelSz = 1;
        Allocation inception_3a_3x3_reduce_Allocation = Allocation.createSized(rs, Element.F32(rs), Wout * Hout * numOutLayer); //Output size
        int[] inception_3a_3x3_reduce_TypeSet = {6};//{1, 2, 4, 6, 8, 12, 24};
        for (int convType : inception_3a_3x3_reduce_TypeSet) {
            int parallelOFMs = numOutLayer / convType;
            paraConvRelu(rs, convNet, /*Weight size F4*/kernelSz * kernelSz * numInLayer * (numOutLayer / 4), /*bias size*/numOutLayer, inception_3a_3x3_reduce_w,
                    inception_3a_3x3_reduce_b, Wout * Hout * parallelOFMs, kernelSz, Wout, Hout, Win, Hin,/*#Out Layers*/ numOutLayer,
                    /*#In Layers*/ numInLayer, /*S*/ 1, /*pad*/0,/*N_new*/ numInLayer / 4, /*offset*/0, pool2_Allocation,
                    inception_3a_3x3_reduce_Allocation, convType, parallelOFMs, "GoogleNet/Logs",
                    String.valueOf(convType) + ".txt", "inception_3a_3x3_reduce (ms): ", logEnable);
            if (debug) {
                float[] tmp = new float[Wout * Hout * numOutLayer];
                inception_3a_3x3_reduce_Allocation.copyTo(tmp);
                binaryDumper("GoogleNet/Intermediate_rslts/Parallel/" + String.valueOf(convType),
                        "inception_3a_3x3_reduce.bin", tmp);
            }
        }
        Log.i(tag, "inception_3a_3x3_reduce finished.");

        /***************************************************************************/
        /* inception_3a_5x5_reduce */
        Wout = 28;
        Hout = 28;
        Win = 28;
        Hin = 28;
        numInLayer = 192;
        numOutLayer = 16;
        kernelSz = 1;
        Allocation inception_3a_5x5_reduce_Allocation = Allocation.createSized(rs, Element.F32(rs), Wout * Hout * numOutLayer); //Output size
        int[] inception_3a_5x5_reduce_TypeSet = {4};//{1, 2, 4};
        for (int convType : inception_3a_5x5_reduce_TypeSet) {
            int parallelOFMs = numOutLayer / convType;
            paraConvRelu(rs, convNet, /*Weight size F4*/kernelSz * kernelSz * numInLayer * (numOutLayer / 4), /*bias size*/numOutLayer, inception_3a_5x5_reduce_w,
                    inception_3a_5x5_reduce_b, Wout * Hout * parallelOFMs, kernelSz, Wout, Hout, Win, Hin,/*#Out Layers*/ numOutLayer,
                    /*#In Layers*/ numInLayer, /*S*/ 1, /*pad*/0,/*N_new*/ numInLayer / 4, /*offset*/0, pool2_Allocation,
                    inception_3a_5x5_reduce_Allocation, convType, parallelOFMs, "GoogleNet/Logs",
                    String.valueOf(convType) + ".txt", "inception_3a_5x5_reduce (ms): ", logEnable);
            if (debug) {
                float[] tmp = new float[Wout * Hout * numOutLayer];
                inception_3a_5x5_reduce_Allocation.copyTo(tmp);
                binaryDumper("GoogleNet/Intermediate_rslts/Parallel/" + String.valueOf(convType),
                        "inception_3a_5x5_reduce.bin", tmp);
            }
        }
        Log.i(tag, "inception_3a_5x5_reduce finished.");

        /***************************************************************************/
        /* Pool 3 */
        Wout = 28;
        Hout = 28;
        Win = 28;
        Hin = 28;
        numOutLayer = 192;
        kernelSz = 3;
        Allocation pool3_Allocation = Allocation.createSized(rs, Element.F32(rs), Wout * Hout * numOutLayer);
        paraMaxPoolNew(rs, convNet, Wout * Hout * (numOutLayer / 4), kernelSz, Win, Hin, Wout, Hout, 1,
                pool2_Allocation, pool3_Allocation, "GoogleNet/Logs", "4.txt", "Pool3 (ms): ", logEnable);

        if (debug) {
            float[] tmp = new float[Wout * Hout * numOutLayer];
            pool3_Allocation.copyTo(tmp);
            binaryDumper("GoogleNet/Intermediate_rslts/Parallel/4", "pool3.bin", tmp);
        }
        Log.i(tag, "Pool3 finished.");
        /***************************************************************************/
        int[] inception_3a_TypeSet = {4};//{1, 2, 4, 8, 16, 32};

        int[] inception_3a_1x1_TypeSet = {4};//{1, 2, 4, 8, 16};
        int[] inception_3a_3x3_TypeSet = {4};//{1, 2, 4, 8, 16, 32};
        int[] inception_3a_5x5_TypeSet = {4};//{1, 2, 4, 8};
        int[] inception_3a_pool_proj_TypeSet = {4};//{1, 2, 4, 8};


        for (int genericConvType : inception_3a_TypeSet) { //Very bad idea.We should have separate loops
            /***************************************************************************/
            /* inception_3a_1x1 */
            int convType = findConvType(inception_3a_1x1_TypeSet, genericConvType);
            Wout = 28;
            Hout = 28;
            Win = 28;
            Hin = 28;
            numInLayer = 192;
            numOutLayer = 64;
            kernelSz = 1;
            int parallelOFMs = numOutLayer / convType;
            paraConvRelu(rs, convNet, /*Weight size F4*/kernelSz * kernelSz * numInLayer * (numOutLayer / 4), /*bias size*/numOutLayer, inception_3a_1x1_w,
                    inception_3a_1x1_b, Wout * Hout * parallelOFMs, kernelSz, Wout, Hout, Win, Hin,/*#Out Layers*/ numOutLayer,
                    /*#In Layers*/ numInLayer, /*S*/ 1, /*pad*/0,/*N_new*/ numInLayer / 4, /*offset*/0, pool2_Allocation,
                    output_3a_Allocation, convType, parallelOFMs, "GoogleNet/Logs",
                    String.valueOf(convType) + ".txt", "inception_3a_1x1 (ms): ", logEnable && uglyLoop);
            Log.i(tag, "inception_3a_1x1 finished: " + genericConvType + ", " + convType);

            /***************************************************************************/
             /* inception_3a_3x3 */
            convType = findConvType(inception_3a_3x3_TypeSet, genericConvType);
            Wout = 28;
            Hout = 28;
            Win = 28;
            Hin = 28;
            numInLayer = 96;
            numOutLayer = 128;
            kernelSz = 3;

            parallelOFMs = numOutLayer / convType;
            paraConvRelu(rs, convNet, /*Weight size F4*/kernelSz * kernelSz * numInLayer * (numOutLayer / 4), /*bias size*/numOutLayer, inception_3a_3x3_w,
                    inception_3a_3x3_b, Wout * Hout * parallelOFMs, kernelSz, Wout, Hout, Win, Hin,/*#Out Layers*/ numOutLayer,
                    /*#In Layers*/ numInLayer, /*S*/ 1, /*pad*/1,/*N_new*/ numInLayer / 4, /*offset*/28 * 28 * 64, inception_3a_3x3_reduce_Allocation,
                    output_3a_Allocation, convType, parallelOFMs, "GoogleNet/Logs",
                    String.valueOf(convType) + ".txt", "inception_3a_3x3 (ms): ", logEnable && uglyLoop);
            Log.i(tag, "inception_3a_3x3 finished: " + genericConvType + ", " + convType);

            /***************************************************************************/
            /* inception_3a_5x5 */
            convType = findConvType(inception_3a_5x5_TypeSet, genericConvType);
            Wout = 28;
            Hout = 28;
            Win = 28;
            Hin = 28;
            numInLayer = 16;
            numOutLayer = 32;
            kernelSz = 5;
            parallelOFMs = numOutLayer / convType;
            paraConvRelu(rs, convNet, /*Weight size F4*/kernelSz * kernelSz * numInLayer * (numOutLayer / 4), /*bias size*/numOutLayer, inception_3a_5x5_w,
                    inception_3a_5x5_b, Wout * Hout * parallelOFMs, kernelSz, Wout, Hout, Win, Hin,/*#Out Layers*/ numOutLayer,
                    /*#In Layers*/ numInLayer, /*S*/ 1, /*pad*/2,/*N_new*/ numInLayer / 4, /*offset*/28 * 28 * 64 + 28 * 28 * 128,
                    inception_3a_5x5_reduce_Allocation, output_3a_Allocation, convType, parallelOFMs, "GoogleNet/Logs",
                    String.valueOf(convType) + ".txt", "inception_3a_5x5 (ms): ", logEnable && uglyLoop);
            Log.i(tag, "inception_3a_5x5 finished: " + genericConvType + ", " + convType);

            /***************************************************************************/
            /* inception_3a_pool_proj */
            convType = findConvType(inception_3a_pool_proj_TypeSet, genericConvType);
            Wout = 28;
            Hout = 28;
            Win = 28;
            Hin = 28;
            numInLayer = 192;
            numOutLayer = 32;
            kernelSz = 1;
            parallelOFMs = numOutLayer / convType;
            paraConvRelu(rs, convNet, /*Weight size F4*/kernelSz * kernelSz * numInLayer * (numOutLayer / 4), /*bias size*/numOutLayer, inception_3a_pool_proj_w,
                    inception_3a_pool_proj_b, Wout * Hout * parallelOFMs, kernelSz, Wout, Hout, Win, Hin,/*#Out Layers*/ numOutLayer,
                    /*#In Layers*/ numInLayer, /*S*/ 1, /*pad*/0,/*N_new*/ numInLayer / 4,
                    /*offset*/28 * 28 * 64 + 28 * 28 * 128 + 28 * 28 * 32, pool3_Allocation,
                    output_3a_Allocation, convType, parallelOFMs, "GoogleNet/Logs",
                    String.valueOf(convType) + ".txt", "inception_3a_pool_proj (ms): ", logEnable && uglyLoop);
            Log.i(tag, "inception_3a_pool_proj finished: " + genericConvType + ", " + convType);

            if (debug) {
                float[] tmp = new float[28 * 28 * 256];
                output_3a_Allocation.copyTo(tmp);
                binaryDumper("GoogleNet/Intermediate_rslts/Parallel/" + String.valueOf(genericConvType),
                        "output_3a.bin", tmp);
            }
        }


        /* output_3b */
        Allocation output_3b_Allocation = Allocation.createSized(rs, Element.F32(rs), 28 * 28 * 480); //Output size
        /***************************************************************************/
        /* inception_3b_3x3_reduce */
        Wout = 28;
        Hout = 28;
        Win = 28;
        Hin = 28;
        numInLayer = 256;
        numOutLayer = 128;
        kernelSz = 1;
        Allocation inception_3b_3x3_reduce_Allocation = Allocation.createSized(rs, Element.F32(rs), Wout * Hout * numOutLayer); //Output size
        int[] inception_3b_3x3_reduce_TypeSet = {4};//{1, 2, 4, 8, 16, 32};
        for (int convType : inception_3b_3x3_reduce_TypeSet) {
            int parallelOFMs = numOutLayer / convType;
            paraConvRelu(rs, convNet, /*Weight size F4*/kernelSz * kernelSz * numInLayer * (numOutLayer / 4), /*bias size*/numOutLayer, inception_3b_3x3_reduce_w,
                    inception_3b_3x3_reduce_b, Wout * Hout * parallelOFMs, kernelSz, Wout, Hout, Win, Hin,/*#Out Layers*/ numOutLayer,
                    /*#In Layers*/ numInLayer, /*S*/ 1, /*pad*/0,/*N_new*/ numInLayer / 4, /*offset*/0, output_3a_Allocation,
                    inception_3b_3x3_reduce_Allocation, convType, parallelOFMs, "GoogleNet/Logs",
                    String.valueOf(convType) + ".txt", "inception_3b_3x3_reduce (ms): ", logEnable);
            if (debug) {
                float[] tmp = new float[Wout * Hout * numOutLayer];
                inception_3b_3x3_reduce_Allocation.copyTo(tmp);
                binaryDumper("GoogleNet/Intermediate_rslts/Parallel/" + String.valueOf(convType),
                        "inception_3b_3x3_reduce.bin", tmp);
            }
        }
        Log.i(tag, "inception_3b_3x3_reduce finished.");

        /***************************************************************************/
        /* inception_3b_5x5_reduce */
        Wout = 28;
        Hout = 28;
        Win = 28;
        Hin = 28;
        numInLayer = 256;
        numOutLayer = 32;
        kernelSz = 1;
        Allocation inception_3b_5x5_reduce_Allocation = Allocation.createSized(rs, Element.F32(rs), Wout * Hout * numOutLayer); //Output size
        int[] inception_3b_5x5_reduce_TypeSet = {4};//{1, 2, 4, 8};
        for (int convType : inception_3b_5x5_reduce_TypeSet) {
            int parallelOFMs = numOutLayer / convType;
            paraConvRelu(rs, convNet, /*Weight size F4*/kernelSz * kernelSz * numInLayer * (numOutLayer / 4), /*bias size*/numOutLayer, inception_3b_5x5_reduce_w,
                    inception_3b_5x5_reduce_b, Wout * Hout * parallelOFMs, kernelSz, Wout, Hout, Win, Hin,/*#Out Layers*/ numOutLayer,
                    /*#In Layers*/ numInLayer, /*S*/ 1, /*pad*/0,/*N_new*/ numInLayer / 4, /*offset*/0, output_3a_Allocation,
                    inception_3b_5x5_reduce_Allocation, convType, parallelOFMs, "GoogleNet/Logs",
                    String.valueOf(convType) + ".txt", "inception_3b_5x5_reduce (ms): ", logEnable);
            if (debug) {
                float[] tmp = new float[Wout * Hout * numOutLayer];
                inception_3b_5x5_reduce_Allocation.copyTo(tmp);
                binaryDumper("GoogleNet/Intermediate_rslts/Parallel/" + String.valueOf(convType),
                        "inception_3b_5x5_reduce.bin", tmp);
            }
        }
        Log.i(tag, "inception_3b_5x5_reduce finished.");

        /***************************************************************************/
        /* Pool 4 */
        Wout = 28;
        Hout = 28;
        Win = 28;
        Hin = 28;
        numOutLayer = 256;
        kernelSz = 3;
        Allocation pool4_Allocation = Allocation.createSized(rs, Element.F32(rs), Wout * Hout * numOutLayer);
        paraMaxPoolNew(rs, convNet, Wout * Hout * (numOutLayer / 4), kernelSz, Win, Hin, Wout, Hout, 1,
                output_3a_Allocation, pool4_Allocation, "GoogleNet/Logs", "4.txt", "Pool4 (ms): ", logEnable);

        if (debug) {
            float[] tmp = new float[Wout * Hout * numOutLayer];
            pool4_Allocation.copyTo(tmp);
            binaryDumper("GoogleNet/Intermediate_rslts/Parallel/4", "pool4.bin", tmp);
        }
        Log.i(tag, "Pool4 finished.");
        /***************************************************************************/

        int[] inception_3b_TypeSet = {4};//{1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64};

        int[] inception_3b_1x1_TypeSet = {4};//{1, 2, 4, 8, 16, 32};
        int[] inception_3b_3x3_TypeSet = {4};//{1, 2, 4, 6, 8, 12, 16, 24, 48};
        int[] inception_3b_5x5_TypeSet = {4};//{1, 2, 4, 6, 8, 12, 24};
        int[] inception_3b_pool_proj_TypeSet ={4};//{1, 2, 4, 8, 16};


        for (int genericConvType : inception_3b_TypeSet) {
            /***************************************************************************/
            /* inception_3b_1x1 */
            int convType = findConvType(inception_3b_1x1_TypeSet, genericConvType);
            Wout = 28;
            Hout = 28;
            Win = 28;
            Hin = 28;
            numInLayer = 256;
            numOutLayer = 128;
            kernelSz = 1;
            int parallelOFMs = numOutLayer / convType;
            paraConvRelu(rs, convNet, /*Weight size F4*/kernelSz * kernelSz * numInLayer * (numOutLayer / 4), /*bias size*/numOutLayer, inception_3b_1x1_w,
                    inception_3b_1x1_b, Wout * Hout * parallelOFMs, kernelSz, Wout, Hout, Win, Hin,/*#Out Layers*/ numOutLayer,
                /*#In Layers*/ numInLayer, /*S*/ 1, /*pad*/0,/*N_new*/ numInLayer / 4, /*offset*/0, output_3a_Allocation,
                    output_3b_Allocation, convType, parallelOFMs, "GoogleNet/Logs",
                    String.valueOf(convType) + ".txt", "inception_3b_1x1 (ms): ", logEnable && uglyLoop);

            Log.i(tag, "inception_3b_1x1 finished: " + genericConvType + ", " + convType);

            /***************************************************************************/
            /* inception_3b_3x3 */
            convType = findConvType(inception_3b_3x3_TypeSet, genericConvType);
            Wout = 28;
            Hout = 28;
            Win = 28;
            Hin = 28;
            numInLayer = 128;
            numOutLayer = 192;
            kernelSz = 3;
            parallelOFMs = numOutLayer / convType;
            paraConvRelu(rs, convNet, /*Weight size F4*/kernelSz * kernelSz * numInLayer * (numOutLayer / 4), /*bias size*/numOutLayer, inception_3b_3x3_w,
                    inception_3b_3x3_b, Wout * Hout * parallelOFMs, kernelSz, Wout, Hout, Win, Hin,/*#Out Layers*/ numOutLayer,
                /*#In Layers*/ numInLayer, /*S*/ 1, /*pad*/1,/*N_new*/ numInLayer / 4, /*offset*/28 * 28 * 128, inception_3b_3x3_reduce_Allocation,
                    output_3b_Allocation, convType, parallelOFMs, "GoogleNet/Logs",
                    String.valueOf(convType) + ".txt", "inception_3b_3x3 (ms): ", logEnable && uglyLoop);
            Log.i(tag, "inception_3b_3x3 finished: " + genericConvType + ", " + convType);

            /***************************************************************************/
            /* inception_3b_5x5 */
            convType = findConvType(inception_3b_5x5_TypeSet, genericConvType);
            Wout = 28;
            Hout = 28;
            Win = 28;
            Hin = 28;
            numInLayer = 32;
            numOutLayer = 96;
            kernelSz = 5;
            parallelOFMs = numOutLayer / convType;
            paraConvRelu(rs, convNet, /*Weight size F4*/kernelSz * kernelSz * numInLayer * (numOutLayer / 4), /*bias size*/numOutLayer, inception_3b_5x5_w,
                    inception_3b_5x5_b, Wout * Hout * parallelOFMs, kernelSz, Wout, Hout, Win, Hin,/*#Out Layers*/ numOutLayer,
                /*#In Layers*/ numInLayer, /*S*/ 1, /*pad*/2,/*N_new*/ numInLayer / 4, /*offset*/28 * 28 * 128 + 28 * 28 * 192,
                    inception_3b_5x5_reduce_Allocation, output_3b_Allocation, convType, parallelOFMs, "GoogleNet/Logs",
                    String.valueOf(convType) + ".txt", "inception_3b_5x5 (ms): ", logEnable && uglyLoop);
            Log.i(tag, "inception_3b_5x5 finished: " + genericConvType + ", " + convType);

            /***************************************************************************/
            /* inception_3b_pool_proj */
            convType = findConvType(inception_3b_pool_proj_TypeSet, genericConvType);
            Wout = 28;
            Hout = 28;
            Win = 28;
            Hin = 28;
            numInLayer = 256;
            numOutLayer = 64;
            kernelSz = 1;
            parallelOFMs = numOutLayer / convType;
            paraConvRelu(rs, convNet, /*Weight size F4*/kernelSz * kernelSz * numInLayer * (numOutLayer / 4), /*bias size*/numOutLayer, inception_3b_pool_proj_w,
                    inception_3b_pool_proj_b, Wout * Hout * parallelOFMs, kernelSz, Wout, Hout, Win, Hin,/*#Out Layers*/ numOutLayer,
                /*#In Layers*/ numInLayer, /*S*/ 1, /*pad*/0,/*N_new*/ numInLayer / 4,
                /*offset*/28 * 28 * 128 + 28 * 28 * 192 + 28 * 28 * 96, pool4_Allocation,
                    output_3b_Allocation, convType, parallelOFMs, "GoogleNet/Logs",
                    String.valueOf(convType) + ".txt", "inception_3b_pool_proj (ms): ", logEnable && uglyLoop);
            Log.i(tag, "inception_3b_pool_proj finished: " + genericConvType + ", " + convType);

            if (debug) {
                float[] tmp = new float[28 * 28 * 480];
                output_3b_Allocation.copyTo(tmp);
                binaryDumper("GoogleNet/Intermediate_rslts/Parallel/" + String.valueOf(genericConvType),
                        "output_3b.bin", tmp);
            }

        }

        /***************************************************************************/
        /* Pool 5 */
        Wout = 14;
        Hout = 14;
        Win = 28;
        Hin = 28;
        numOutLayer = 480;
        kernelSz = 3;
        Allocation pool5_Allocation = Allocation.createSized(rs, Element.F32(rs), Wout * Hout * numOutLayer);
        paraMaxPool(rs, convNet, Wout * Hout * (numOutLayer / 4), kernelSz, Win, Hin, Wout, Hout, 2,
                output_3b_Allocation, pool5_Allocation, "GoogleNet/Logs", "4.txt", "Pool5 (ms): ", logEnable);

        if (debug) {
            float[] tmp = new float[Wout * Hout * numOutLayer];
            pool5_Allocation.copyTo(tmp);
            binaryDumper("GoogleNet/Intermediate_rslts/Parallel/4", "pool5.bin", tmp);
        }
        Log.i(tag, "Pool5 finished.");


        /* output_4a */
        Allocation output_4a_Allocation = Allocation.createSized(rs, Element.F32(rs), 14 * 14 * 512); //Output size
        /***************************************************************************/
        /* inception_4a_3x3_reduce */
        Wout = 14;
        Hout = 14;
        Win = 14;
        Hin = 14;
        numInLayer = 480;
        numOutLayer = 96;
        kernelSz = 1;
        Allocation inception_4a_3x3_reduce_Allocation = Allocation.createSized(rs, Element.F32(rs), Wout * Hout * numOutLayer); //Output size
        int[] inception_4a_3x3_reduce_TypeSet = {4};//{1, 2, 4, 6, 8, 12, 24};
        for (int convType : inception_4a_3x3_reduce_TypeSet) {
            int parallelOFMs = numOutLayer / convType;
            paraConvRelu(rs, convNet, /*Weight size F4*/kernelSz * kernelSz * numInLayer * (numOutLayer / 4), /*bias size*/numOutLayer, inception_4a_3x3_reduce_w,
                    inception_4a_3x3_reduce_b, Wout * Hout * parallelOFMs, kernelSz, Wout, Hout, Win, Hin,/*#Out Layers*/ numOutLayer,
                    /*#In Layers*/ numInLayer, /*S*/ 1, /*pad*/0,/*N_new*/ numInLayer / 4, /*offset*/0, pool5_Allocation,
                    inception_4a_3x3_reduce_Allocation, convType, parallelOFMs, "GoogleNet/Logs",
                    String.valueOf(convType) + ".txt", "inception_4a_3x3_reduce (ms): ", logEnable);
            if (debug) {
                float[] tmp = new float[Wout * Hout * numOutLayer];
                inception_4a_3x3_reduce_Allocation.copyTo(tmp);
                binaryDumper("GoogleNet/Intermediate_rslts/Parallel/" + String.valueOf(convType),
                        "inception_4a_3x3_reduce.bin", tmp);
            }
        }
        Log.i(tag, "inception_4a_3x3_reduce finished.");

        /***************************************************************************/
        /* inception_4a_5x5_reduce */
        Wout = 14;
        Hout = 14;
        Win = 14;
        Hin = 14;
        numInLayer = 480;
        numOutLayer = 16;
        kernelSz = 1;
        Allocation inception_4a_5x5_reduce_Allocation = Allocation.createSized(rs, Element.F32(rs), Wout * Hout * numOutLayer); //Output size
        int[] inception_4a_5x5_reduce_TypeSet = {4};//{1, 2, 4};
        for (int convType : inception_4a_5x5_reduce_TypeSet) {
            int parallelOFMs = numOutLayer / convType;
            paraConvRelu(rs, convNet, /*Weight size F4*/kernelSz * kernelSz * numInLayer * (numOutLayer / 4), /*bias size*/numOutLayer, inception_4a_5x5_reduce_w,
                    inception_4a_5x5_reduce_b, Wout * Hout * parallelOFMs, kernelSz, Wout, Hout, Win, Hin,/*#Out Layers*/ numOutLayer,
                    /*#In Layers*/ numInLayer, /*S*/ 1, /*pad*/0,/*N_new*/ numInLayer / 4, /*offset*/0, pool5_Allocation,
                    inception_4a_5x5_reduce_Allocation, convType, parallelOFMs, "GoogleNet/Logs",
                    String.valueOf(convType) + ".txt", "inception_4a_5x5_reduce (ms): ", logEnable);
            if (debug) {
                float[] tmp = new float[Wout * Hout * numOutLayer];
                inception_4a_5x5_reduce_Allocation.copyTo(tmp);
                binaryDumper("GoogleNet/Intermediate_rslts/Parallel/" + String.valueOf(convType),
                        "inception_4a_5x5_reduce.bin", tmp);
            }
        }
        Log.i(tag, "inception_4a_5x5_reduce finished.");

        /***************************************************************************/
        /* Pool 6 */
        Wout = 14;
        Hout = 14;
        Win = 14;
        Hin = 14;
        numOutLayer = 480;
        kernelSz = 3;
        Allocation pool6_Allocation = Allocation.createSized(rs, Element.F32(rs), Wout * Hout * numOutLayer);
        paraMaxPoolNew(rs, convNet, Wout * Hout * (numOutLayer / 4), kernelSz, Win, Hin, Wout, Hout, 1,
                pool5_Allocation, pool6_Allocation, "GoogleNet/Logs", "4.txt", "Pool6 (ms): ", logEnable);

        if (debug) {
            float[] tmp = new float[Wout * Hout * numOutLayer];
            pool6_Allocation.copyTo(tmp);
            binaryDumper("GoogleNet/Intermediate_rslts/Parallel/4", "Pool6.bin", tmp);
        }
        Log.i(tag, "Pool6 finished.");
        /***************************************************************************/
        int[] inception_4a_TypeSet = {4};//{1, 2, 4, 6, 8, 12, 13, 16, 24, 26, 48, 52};

        int[] inception_4a_1x1_TypeSet = {4};//{1, 2, 4, 6, 8, 12, 16, 24, 48};
        int[] inception_4a_3x3_TypeSet = {4};//{1, 2, 4, 13, 26, 52};
        int[] inception_4a_5x5_TypeSet = {4};//{1, 2, 4, 6, 12};
        int[] inception_4a_pool_proj_TypeSet ={4};//{1, 2, 4, 8, 16};

        for (int genericConvType : inception_4a_TypeSet) {
            /***************************************************************************/
            /* inception_4a_1x1 */
            int convType = findConvType(inception_4a_1x1_TypeSet, genericConvType);
            Wout = 14;
            Hout = 14;
            Win = 14;
            Hin = 14;
            numInLayer = 480;
            numOutLayer = 192;
            kernelSz = 1;
            int parallelOFMs = numOutLayer / convType;
            paraConvRelu(rs, convNet, /*Weight size F4*/kernelSz * kernelSz * numInLayer * (numOutLayer / 4), /*bias size*/numOutLayer, inception_4a_1x1_w,
                    inception_4a_1x1_b, Wout * Hout * parallelOFMs, kernelSz, Wout, Hout, Win, Hin,/*#Out Layers*/ numOutLayer,
                    /*#In Layers*/ numInLayer, /*S*/ 1, /*pad*/0,/*N_new*/ numInLayer / 4, /*offset*/0, pool5_Allocation,
                    output_4a_Allocation, convType, parallelOFMs, "GoogleNet/Logs",
                    String.valueOf(convType) + ".txt", "inception_4a_1x1 (ms): ", logEnable && uglyLoop);
            Log.i(tag, "inception_4a_1x1 finished: " + genericConvType + ", " + convType);


            /***************************************************************************/
            /* inception_4a_3x3 */
            convType = findConvType(inception_4a_3x3_TypeSet, genericConvType);
            Wout = 14;
            Hout = 14;
            Win = 14;
            Hin = 14;
            numInLayer = 96;
            numOutLayer = 208;
            kernelSz = 3;

            parallelOFMs = numOutLayer / convType;
            paraConvRelu(rs, convNet, /*Weight size F4*/kernelSz * kernelSz * numInLayer * (numOutLayer / 4), /*bias size*/numOutLayer, inception_4a_3x3_w,
                    inception_4a_3x3_b, Wout * Hout * parallelOFMs, kernelSz, Wout, Hout, Win, Hin,/*#Out Layers*/ numOutLayer,
                    /*#In Layers*/ numInLayer, /*S*/ 1, /*pad*/1,/*N_new*/ numInLayer / 4, /*offset*/14 * 14 * 192, inception_4a_3x3_reduce_Allocation,
                    output_4a_Allocation, convType, parallelOFMs, "GoogleNet/Logs",
                    String.valueOf(convType) + ".txt", "inception_4a_3x3 (ms): ", logEnable && uglyLoop);
            Log.i(tag, "inception_4a_3x3 finished: " + genericConvType + ", " + convType);

            /***************************************************************************/
            /* inception_4a_5x5 */
            convType = findConvType(inception_4a_5x5_TypeSet, genericConvType);
            Wout = 14;
            Hout = 14;
            Win = 14;
            Hin = 14;
            numInLayer = 16;
            numOutLayer = 48;
            kernelSz = 5;

            parallelOFMs = numOutLayer / convType;
            paraConvRelu(rs, convNet, /*Weight size F4*/kernelSz * kernelSz * numInLayer * (numOutLayer / 4), /*bias size*/numOutLayer, inception_4a_5x5_w,
                    inception_4a_5x5_b, Wout * Hout * parallelOFMs, kernelSz, Wout, Hout, Win, Hin,/*#Out Layers*/ numOutLayer,
                    /*#In Layers*/ numInLayer, /*S*/ 1, /*pad*/2,/*N_new*/ numInLayer / 4, /*offset*/14 * 14 * 192 + 14 * 14 * 208,
                    inception_4a_5x5_reduce_Allocation, output_4a_Allocation, convType, parallelOFMs, "GoogleNet/Logs",
                    String.valueOf(convType) + ".txt", "inception_4a_5x5 (ms): ", logEnable && uglyLoop);
            Log.i(tag, "inception_4a_5x5 finished: " + genericConvType + ", " + convType);

            /***************************************************************************/
            /* inception_4a_pool_proj */
            convType = findConvType(inception_4a_pool_proj_TypeSet, genericConvType);
            Wout = 14;
            Hout = 14;
            Win = 14;
            Hin = 14;
            numInLayer = 480;
            numOutLayer = 64;
            kernelSz = 1;

            parallelOFMs = numOutLayer / convType;
            paraConvRelu(rs, convNet, /*Weight size F4*/kernelSz * kernelSz * numInLayer * (numOutLayer / 4), /*bias size*/numOutLayer, inception_4a_pool_proj_w,
                    inception_4a_pool_proj_b, Wout * Hout * parallelOFMs, kernelSz, Wout, Hout, Win, Hin,/*#Out Layers*/ numOutLayer,
                    /*#In Layers*/ numInLayer, /*S*/ 1, /*pad*/0,/*N_new*/ numInLayer / 4,
                    /*offset*/14 * 14 * 192 + 14 * 14 * 208 + 14 * 14 * 48, pool6_Allocation,
                    output_4a_Allocation, convType, parallelOFMs, "GoogleNet/Logs",
                    String.valueOf(convType) + ".txt", "inception_4a_pool_proj (ms): ", logEnable && uglyLoop);
            Log.i(tag, "inception_4a_pool_proj finished: " + genericConvType + ", " + convType);

            if (debug) {
                float[] tmp = new float[14 * 14 * 512];
                output_4a_Allocation.copyTo(tmp);
                binaryDumper("GoogleNet/Intermediate_rslts/Parallel/" + String.valueOf(genericConvType),
                        "output_4a.bin", tmp);
            }
        }

        /* output_4b */
        Allocation output_4b_Allocation = Allocation.createSized(rs, Element.F32(rs), 14 * 14 * 512); //Output size
        /***************************************************************************/
        /* inception_4b_3x3_reduce */
        Wout = 14;
        Hout = 14;
        Win = 14;
        Hin = 14;
        numInLayer = 512;
        numOutLayer = 112;
        kernelSz = 1;
        Allocation inception_4b_3x3_reduce_Allocation = Allocation.createSized(rs, Element.F32(rs), Wout * Hout * numOutLayer); //Output size
        int[] inception_4b_3x3_reduce_TypeSet = {1, 2, 4, 7, 14, 28};
        for (int convType : inception_4b_3x3_reduce_TypeSet) {
            int parallelOFMs = numOutLayer / convType;
            paraConvRelu(rs, convNet, /*Weight size F4*/kernelSz * kernelSz * numInLayer * (numOutLayer / 4), /*bias size*/numOutLayer, inception_4b_3x3_reduce_w,
                    inception_4b_3x3_reduce_b, Wout * Hout * parallelOFMs, kernelSz, Wout, Hout, Win, Hin,/*#Out Layers*/ numOutLayer,
                    /*#In Layers*/ numInLayer, /*S*/ 1, /*pad*/0,/*N_new*/ numInLayer / 4, /*offset*/0, output_4a_Allocation,
                    inception_4b_3x3_reduce_Allocation, convType, parallelOFMs, "GoogleNet/Logs",
                    String.valueOf(convType) + ".txt", "inception_4b_3x3_reduce (ms): ", logEnable);
            if (debug) {
                float[] tmp = new float[Wout * Hout * numOutLayer];
                inception_4b_3x3_reduce_Allocation.copyTo(tmp);
                binaryDumper("GoogleNet/Intermediate_rslts/Parallel/" + String.valueOf(convType),
                        "inception_4b_3x3_reduce.bin", tmp);
            }
        }
        Log.i(tag, "inception_4b_3x3_reduce finished.");

        /***************************************************************************/
        /* inception_4b_5x5_reduce */
        Wout = 14;
        Hout = 14;
        Win = 14;
        Hin = 14;
        numInLayer = 512;
        numOutLayer = 24;
        kernelSz = 1;
        Allocation inception_4b_5x5_reduce_Allocation = Allocation.createSized(rs, Element.F32(rs), Wout * Hout * numOutLayer); //Output size
        int[] inception_4b_5x5_reduce_TypeSet = {1, 2, 6};
        for (int convType : inception_4b_5x5_reduce_TypeSet) {
            int parallelOFMs = numOutLayer / convType;
            paraConvRelu(rs, convNet, /*Weight size F4*/kernelSz * kernelSz * numInLayer * (numOutLayer / 4), /*bias size*/numOutLayer, inception_4b_5x5_reduce_w,
                    inception_4b_5x5_reduce_b, Wout * Hout * parallelOFMs, kernelSz, Wout, Hout, Win, Hin,/*#Out Layers*/ numOutLayer,
                    /*#In Layers*/ numInLayer, /*S*/ 1, /*pad*/0,/*N_new*/ numInLayer / 4, /*offset*/0, output_4a_Allocation,
                    inception_4b_5x5_reduce_Allocation, convType, parallelOFMs, "GoogleNet/Logs",
                    String.valueOf(convType) + ".txt", "inception_4b_5x5_reduce (ms): ", logEnable);
            if (debug) {
                float[] tmp = new float[Wout * Hout * numOutLayer];
                inception_4b_5x5_reduce_Allocation.copyTo(tmp);
                binaryDumper("GoogleNet/Intermediate_rslts/Parallel/" + String.valueOf(convType),
                        "inception_4b_5x5_reduce.bin", tmp);
            }
        }
        Log.i(tag, "inception_4b_5x5_reduce finished.");

        /***************************************************************************/
        /* Pool 7 */
        Wout = 14;
        Hout = 14;
        Win = 14;
        Hin = 14;
        numOutLayer = 512;
        kernelSz = 3;
        Allocation pool7_Allocation = Allocation.createSized(rs, Element.F32(rs), Wout * Hout * numOutLayer);
        paraMaxPoolNew(rs, convNet, Wout * Hout * (numOutLayer / 4), kernelSz, Win, Hin, Wout, Hout, 1,
                output_4a_Allocation, pool7_Allocation, "GoogleNet/Logs", "4.txt", "Pool7 (ms): ", logEnable);

        if (debug) {
            float[] tmp = new float[Wout * Hout * numOutLayer];
            pool7_Allocation.copyTo(tmp);
            binaryDumper("GoogleNet/Intermediate_rslts/Parallel/4", "Pool7.bin", tmp);
        }
        Log.i(tag, "Pool7 finished.");
        /***************************************************************************/
        int[] inception_4b_TypeSet ={4};// {1, 2, 4, 7, 8, 14, 16, 20, 28, 40, 56};

        int[] inception_4b_1x1_TypeSet ={4};//{1, 2, 4, 8, 20, 40};
        int[] inception_4b_3x3_TypeSet ={4};//{1, 2, 4, 7, 8, 14, 28, 56};
        int[] inception_4b_5x5_TypeSet ={4};//{1, 2, 4, 8, 16};
        int[] inception_4b_pool_proj_TypeSet ={4};//{1, 2, 4, 8, 16};


        for (int genericConvType : inception_4b_TypeSet) {
            /***************************************************************************/
            /* inception_4b_1x1 */
            int convType = findConvType(inception_4b_1x1_TypeSet, genericConvType);
            Wout = 14;
            Hout = 14;
            Win = 14;
            Hin = 14;
            numInLayer = 512;
            numOutLayer = 160;
            kernelSz = 1;
            int parallelOFMs = numOutLayer / convType;
            paraConvRelu(rs, convNet, /*Weight size F4*/kernelSz * kernelSz * numInLayer * (numOutLayer / 4), /*bias size*/numOutLayer, inception_4b_1x1_w,
                    inception_4b_1x1_b, Wout * Hout * parallelOFMs, kernelSz, Wout, Hout, Win, Hin,/*#Out Layers*/ numOutLayer,
                    /*#In Layers*/ numInLayer, /*S*/ 1, /*pad*/0,/*N_new*/ numInLayer / 4, /*offset*/0, output_4a_Allocation,
                    output_4b_Allocation, convType, parallelOFMs, "GoogleNet/Logs",
                    String.valueOf(convType) + ".txt", "inception_4b_1x1 (ms): ", logEnable && uglyLoop);
            Log.i(tag, "inception_4b_1x1 finished: " + genericConvType + ", " + convType);

            /***************************************************************************/
            /* inception_4b_3x3 */
            convType = findConvType(inception_4b_3x3_TypeSet, genericConvType);
            Wout = 14;
            Hout = 14;
            Win = 14;
            Hin = 14;
            numInLayer = 112;
            numOutLayer = 224;
            kernelSz = 3;
            parallelOFMs = numOutLayer / convType;
            paraConvRelu(rs, convNet, /*Weight size F4*/kernelSz * kernelSz * numInLayer * (numOutLayer / 4), /*bias size*/numOutLayer, inception_4b_3x3_w,
                    inception_4b_3x3_b, Wout * Hout * parallelOFMs, kernelSz, Wout, Hout, Win, Hin,/*#Out Layers*/ numOutLayer,
                    /*#In Layers*/ numInLayer, /*S*/ 1, /*pad*/1,/*N_new*/ numInLayer / 4, /*offset*/14 * 14 * 160, inception_4b_3x3_reduce_Allocation,
                    output_4b_Allocation, convType, parallelOFMs, "GoogleNet/Logs",
                    String.valueOf(convType) + ".txt", "inception_4b_3x3 (ms): ", logEnable && uglyLoop);
            Log.i(tag, "inception_4b_3x3 finished: " + genericConvType + ", " + convType);

            /***************************************************************************/
            /* inception_4b_5x5 */
            convType = findConvType(inception_4b_5x5_TypeSet, genericConvType);
            Wout = 14;
            Hout = 14;
            Win = 14;
            Hin = 14;
            numInLayer = 24;
            numOutLayer = 64;
            kernelSz = 5;
            parallelOFMs = numOutLayer / convType;
            paraConvRelu(rs, convNet, /*Weight size F4*/kernelSz * kernelSz * numInLayer * (numOutLayer / 4), /*bias size*/numOutLayer, inception_4b_5x5_w,
                    inception_4b_5x5_b, Wout * Hout * parallelOFMs, kernelSz, Wout, Hout, Win, Hin,/*#Out Layers*/ numOutLayer,
                    /*#In Layers*/ numInLayer, /*S*/ 1, /*pad*/2,/*N_new*/ numInLayer / 4, /*offset*/14 * 14 * 160 + 14 * 14 * 224,
                    inception_4b_5x5_reduce_Allocation, output_4b_Allocation, convType, parallelOFMs, "GoogleNet/Logs",
                    String.valueOf(convType) + ".txt", "inception_4b_5x5 (ms): ", logEnable && uglyLoop);
            Log.i(tag, "inception_4b_5x5 finished: " + genericConvType + ", " + convType);

            /***************************************************************************/
            /* inception_4b_pool_proj */
            convType = findConvType(inception_4b_pool_proj_TypeSet, genericConvType);
            Wout = 14;
            Hout = 14;
            Win = 14;
            Hin = 14;
            numInLayer = 512;
            numOutLayer = 64;
            kernelSz = 1;
            parallelOFMs = numOutLayer / convType;
            paraConvRelu(rs, convNet, /*Weight size F4*/kernelSz * kernelSz * numInLayer * (numOutLayer / 4), /*bias size*/numOutLayer, inception_4b_pool_proj_w,
                    inception_4b_pool_proj_b, Wout * Hout * parallelOFMs, kernelSz, Wout, Hout, Win, Hin,/*#Out Layers*/ numOutLayer,
                    /*#In Layers*/ numInLayer, /*S*/ 1, /*pad*/0,/*N_new*/ numInLayer / 4,
                    /*offset*/14 * 14 * 160 + 14 * 14 * 224 + 14 * 14 * 64, pool7_Allocation,
                    output_4b_Allocation, convType, parallelOFMs, "GoogleNet/Logs",
                    String.valueOf(convType) + ".txt", "inception_4b_pool_proj (ms): ", logEnable && uglyLoop);
            Log.i(tag, "inception_4b_pool_proj finished: " + genericConvType + ", " + convType);

            if (debug) {
                float[] tmp = new float[14 * 14 * 512];
                output_4b_Allocation.copyTo(tmp);
                binaryDumper("GoogleNet/Intermediate_rslts/Parallel/" + String.valueOf(genericConvType),
                        "output_4b.bin", tmp);
            }
        }

        /* output_4c */
        Allocation output_4c_Allocation = Allocation.createSized(rs, Element.F32(rs), 14 * 14 * 512); //Output size
        /***************************************************************************/
        /* inception_4c_3x3_reduce */
        Wout = 14;
        Hout = 14;
        Win = 14;
        Hin = 14;
        numInLayer = 512;
        numOutLayer = 128;
        kernelSz = 1;
        Allocation inception_4c_3x3_reduce_Allocation = Allocation.createSized(rs, Element.F32(rs), Wout * Hout * numOutLayer); //Output size
        int[] inception_4c_3x3_reduce_TypeSet = {4};//{1, 2, 4, 8, 16, 32};
        for (int convType : inception_4c_3x3_reduce_TypeSet) {
            int parallelOFMs = numOutLayer / convType;
            paraConvRelu(rs, convNet, /*Weight size F4*/kernelSz * kernelSz * numInLayer * (numOutLayer / 4), /*bias size*/numOutLayer, inception_4c_3x3_reduce_w,
                    inception_4c_3x3_reduce_b, Wout * Hout * parallelOFMs, kernelSz, Wout, Hout, Win, Hin,/*#Out Layers*/ numOutLayer,
                    /*#In Layers*/ numInLayer, /*S*/ 1, /*pad*/0,/*N_new*/ numInLayer / 4, /*offset*/0, output_4b_Allocation,
                    inception_4c_3x3_reduce_Allocation, convType, parallelOFMs, "GoogleNet/Logs",
                    String.valueOf(convType) + ".txt", "inception_4c_3x3_reduce (ms): ", logEnable);
            if (debug) {
                float[] tmp = new float[Wout * Hout * numOutLayer];
                inception_4c_3x3_reduce_Allocation.copyTo(tmp);
                binaryDumper("GoogleNet/Intermediate_rslts/Parallel/" + String.valueOf(convType),
                        "inception_4c_3x3_reduce.bin", tmp);
            }
        }
        Log.i(tag, "inception_4c_3x3_reduce finished.");

        /***************************************************************************/
        /* inception_4c_5x5_reduce */
        Wout = 14;
        Hout = 14;
        Win = 14;
        Hin = 14;
        numInLayer = 512;
        numOutLayer = 24;
        kernelSz = 1;
        Allocation inception_4c_5x5_reduce_Allocation = Allocation.createSized(rs, Element.F32(rs), Wout * Hout * numOutLayer); //Output size
        int[] inception_4c_5x5_reduce_TypeSet ={1, 2, 6};
        for (int convType : inception_4c_5x5_reduce_TypeSet) {
            int parallelOFMs = numOutLayer / convType;
            paraConvRelu(rs, convNet, /*Weight size F4*/kernelSz * kernelSz * numInLayer * (numOutLayer / 4), /*bias size*/numOutLayer, inception_4c_5x5_reduce_w,
                    inception_4c_5x5_reduce_b, Wout * Hout * parallelOFMs, kernelSz, Wout, Hout, Win, Hin,/*#Out Layers*/ numOutLayer,
                    /*#In Layers*/ numInLayer, /*S*/ 1, /*pad*/0,/*N_new*/ numInLayer / 4, /*offset*/0, output_4b_Allocation,
                    inception_4c_5x5_reduce_Allocation, convType, parallelOFMs, "GoogleNet/Logs",
                    String.valueOf(convType) + ".txt", "inception_4c_5x5_reduce (ms): ", logEnable);
            if (debug) {
                float[] tmp = new float[Wout * Hout * numOutLayer];
                inception_4c_5x5_reduce_Allocation.copyTo(tmp);
                binaryDumper("GoogleNet/Intermediate_rslts/Parallel/" + String.valueOf(convType),
                        "inception_4c_5x5_reduce.bin", tmp);
            }
        }
        Log.i(tag, "inception_4c_5x5_reduce finished.");

        /***************************************************************************/
        /* Pool 8 */
        Wout = 14;
        Hout = 14;
        Win = 14;
        Hin = 14;
        numOutLayer = 512;
        kernelSz = 3;
        Allocation pool8_Allocation = Allocation.createSized(rs, Element.F32(rs), Wout * Hout * numOutLayer);
        paraMaxPoolNew(rs, convNet, Wout * Hout * (numOutLayer / 4), kernelSz, Win, Hin, Wout, Hout, 1,
                output_4b_Allocation, pool8_Allocation, "GoogleNet/Logs", "4.txt", "pool8 (ms): ", logEnable);

        if (debug) {
            float[] tmp = new float[Wout * Hout * numOutLayer];
            pool8_Allocation.copyTo(tmp);
            binaryDumper("GoogleNet/Intermediate_rslts/Parallel/4", "pool8.bin", tmp);
        }
        Log.i(tag, "pool8 finished.");
        /***************************************************************************/
        int[] inception_4c_TypeSet = {4};//{1, 2, 4, 8, 16, 32, 64};

        int[] inception_4c_1x1_TypeSet = {4};//{1, 2, 4, 8, 16, 32};
        int[] inception_4c_3x3_TypeSet = {4};//{1, 2, 4, 8, 16, 32, 64};
        int[] inception_4c_5x5_TypeSet = {4};//{1, 2, 4, 8, 16};
        int[] inception_4c_pool_proj_TypeSet ={4};//{1, 2, 4, 8, 16};


        for (int genericConvType : inception_4c_TypeSet) {
            /***************************************************************************/
            /* inception_4c_1x1 */
            int convType = findConvType(inception_4c_1x1_TypeSet, genericConvType);
            Wout = 14;
            Hout = 14;
            Win = 14;
            Hin = 14;
            numInLayer = 512;
            numOutLayer = 128;
            kernelSz = 1;
            int parallelOFMs = numOutLayer / convType;
            paraConvRelu(rs, convNet, /*Weight size F4*/kernelSz * kernelSz * numInLayer * (numOutLayer / 4), /*bias size*/numOutLayer, inception_4c_1x1_w,
                    inception_4c_1x1_b, Wout * Hout * parallelOFMs, kernelSz, Wout, Hout, Win, Hin,/*#Out Layers*/ numOutLayer,
                    /*#In Layers*/ numInLayer, /*S*/ 1, /*pad*/0,/*N_new*/ numInLayer / 4, /*offset*/0, output_4b_Allocation,
                    output_4c_Allocation, convType, parallelOFMs, "GoogleNet/Logs",
                    String.valueOf(convType) + ".txt", "inception_4c_1x1 (ms): ", logEnable && uglyLoop);
            Log.i(tag, "inception_4c_1x1 finished: " + genericConvType + ", " + convType);


            /***************************************************************************/
            /* inception_4c_3x3 */
            convType = findConvType(inception_4c_3x3_TypeSet, genericConvType);
            Wout = 14;
            Hout = 14;
            Win = 14;
            Hin = 14;
            numInLayer = 128;
            numOutLayer = 256;
            kernelSz = 3;
            parallelOFMs = numOutLayer / convType;
            paraConvRelu(rs, convNet, /*Weight size F4*/kernelSz * kernelSz * numInLayer * (numOutLayer / 4), /*bias size*/numOutLayer, inception_4c_3x3_w,
                    inception_4c_3x3_b, Wout * Hout * parallelOFMs, kernelSz, Wout, Hout, Win, Hin,/*#Out Layers*/ numOutLayer,
                    /*#In Layers*/ numInLayer, /*S*/ 1, /*pad*/1,/*N_new*/ numInLayer / 4, /*offset*/14 * 14 * 128, inception_4c_3x3_reduce_Allocation,
                    output_4c_Allocation, convType, parallelOFMs, "GoogleNet/Logs",
                    String.valueOf(convType) + ".txt", "inception_4c_3x3 (ms): ", logEnable && uglyLoop);
            Log.i(tag, "inception_4c_3x3 finished: " + genericConvType + ", " + convType);

            /***************************************************************************/
            /* inception_4c_5x5 */
            convType = findConvType(inception_4c_5x5_TypeSet, genericConvType);
            Wout = 14;
            Hout = 14;
            Win = 14;
            Hin = 14;
            numInLayer = 24;
            numOutLayer = 64;
            kernelSz = 5;
            parallelOFMs = numOutLayer / convType;
            paraConvRelu(rs, convNet, /*Weight size F4*/kernelSz * kernelSz * numInLayer * (numOutLayer / 4), /*bias size*/numOutLayer, inception_4c_5x5_w,
                    inception_4c_5x5_b, Wout * Hout * parallelOFMs, kernelSz, Wout, Hout, Win, Hin,/*#Out Layers*/ numOutLayer,
                    /*#In Layers*/ numInLayer, /*S*/ 1, /*pad*/2,/*N_new*/ numInLayer / 4, /*offset*/14 * 14 * 128 + 14 * 14 * 256,
                    inception_4c_5x5_reduce_Allocation, output_4c_Allocation, convType, parallelOFMs, "GoogleNet/Logs",
                    String.valueOf(convType) + ".txt", "inception_4c_5x5 (ms): ", logEnable && uglyLoop);
            Log.i(tag, "inception_4c_5x5 finished: " + genericConvType + ", " + convType);

            /***************************************************************************/
            /* inception_4c_pool_proj */
            convType = findConvType(inception_4c_pool_proj_TypeSet, genericConvType);
            Wout = 14;
            Hout = 14;
            Win = 14;
            Hin = 14;
            numInLayer = 512;
            numOutLayer = 64;
            kernelSz = 1;
            parallelOFMs = numOutLayer / convType;
            paraConvRelu(rs, convNet, /*Weight size F4*/kernelSz * kernelSz * numInLayer * (numOutLayer / 4), /*bias size*/numOutLayer, inception_4c_pool_proj_w,
                    inception_4c_pool_proj_b, Wout * Hout * parallelOFMs, kernelSz, Wout, Hout, Win, Hin,/*#Out Layers*/ numOutLayer,
                    /*#In Layers*/ numInLayer, /*S*/ 1, /*pad*/0,/*N_new*/ numInLayer / 4,
                    /*offset*/14 * 14 * 128 + 14 * 14 * 256 + 14 * 14 * 64, pool8_Allocation,
                    output_4c_Allocation, convType, parallelOFMs, "GoogleNet/Logs",
                    String.valueOf(convType) + ".txt", "inception_4c_pool_proj (ms): ", logEnable && uglyLoop);
            Log.i(tag, "inception_4c_pool_proj finished: " + genericConvType + ", " + convType);

            if (debug) {
                float[] tmp = new float[14 * 14 * 512];
                output_4c_Allocation.copyTo(tmp);
                binaryDumper("GoogleNet/Intermediate_rslts/Parallel/" + String.valueOf(genericConvType),
                        "output_4c.bin", tmp);
            }
        }

        /* output_4d */
        Allocation output_4d_Allocation = Allocation.createSized(rs, Element.F32(rs), 14 * 14 * 528); //Output size
        /***************************************************************************/
        /* inception_4d_3x3_reduce */
        Wout = 14;
        Hout = 14;
        Win = 14;
        Hin = 14;
        numInLayer = 512;
        numOutLayer = 144;
        kernelSz = 1;
        Allocation inception_4d_3x3_reduce_Allocation = Allocation.createSized(rs, Element.F32(rs), Wout * Hout * numOutLayer); //Output size
        int[] inception_4d_3x3_reduce_TypeSet = {4};//{1, 2, 4, 9, 12, 18, 36};
        for (int convType : inception_4d_3x3_reduce_TypeSet) {
            int parallelOFMs = numOutLayer / convType;
            paraConvRelu(rs, convNet, /*Weight size F4*/kernelSz * kernelSz * numInLayer * (numOutLayer / 4), /*bias size*/numOutLayer, inception_4d_3x3_reduce_w,
                    inception_4d_3x3_reduce_b, Wout * Hout * parallelOFMs, kernelSz, Wout, Hout, Win, Hin,/*#Out Layers*/ numOutLayer,
                    /*#In Layers*/ numInLayer, /*S*/ 1, /*pad*/0,/*N_new*/ numInLayer / 4, /*offset*/0, output_4c_Allocation,
                    inception_4d_3x3_reduce_Allocation, convType, parallelOFMs, "GoogleNet/Logs",
                    String.valueOf(convType) + ".txt", "inception_4d_3x3_reduce (ms): ", logEnable);
            if (debug) {
                float[] tmp = new float[Wout * Hout * numOutLayer];
                inception_4d_3x3_reduce_Allocation.copyTo(tmp);
                binaryDumper("GoogleNet/Intermediate_rslts/Parallel/" + String.valueOf(convType),
                        "inception_4d_3x3_reduce.bin", tmp);
            }
        }
        Log.i(tag, "inception_4d_3x3_reduce finished.");

        /***************************************************************************/
        /* inception_4d_5x5_reduce */
        Wout = 14;
        Hout = 14;
        Win = 14;
        Hin = 14;
        numInLayer = 512;
        numOutLayer = 32;
        kernelSz = 1;
        Allocation inception_4d_5x5_reduce_Allocation = Allocation.createSized(rs, Element.F32(rs), Wout * Hout * numOutLayer); //Output size
        int[] inception_4d_5x5_reduce_TypeSet = {1, 2, 4, 8};
        for (int convType : inception_4d_5x5_reduce_TypeSet) {
            int parallelOFMs = numOutLayer / convType;
            paraConvRelu(rs, convNet, /*Weight size F4*/kernelSz * kernelSz * numInLayer * (numOutLayer / 4), /*bias size*/numOutLayer, inception_4d_5x5_reduce_w,
                    inception_4d_5x5_reduce_b, Wout * Hout * parallelOFMs, kernelSz, Wout, Hout, Win, Hin,/*#Out Layers*/ numOutLayer,
                    /*#In Layers*/ numInLayer, /*S*/ 1, /*pad*/0,/*N_new*/ numInLayer / 4, /*offset*/0, output_4c_Allocation,
                    inception_4d_5x5_reduce_Allocation, convType, parallelOFMs, "GoogleNet/Logs",
                    String.valueOf(convType) + ".txt", "inception_4d_5x5_reduce (ms): ", logEnable);
            if (debug) {
                float[] tmp = new float[Wout * Hout * numOutLayer];
                inception_4d_5x5_reduce_Allocation.copyTo(tmp);
                binaryDumper("GoogleNet/Intermediate_rslts/Parallel/" + String.valueOf(convType),
                        "inception_4d_5x5_reduce.bin", tmp);
            }
        }
        Log.i(tag, "inception_4d_5x5_reduce finished.");

        /***************************************************************************/
        /* Pool 9 */
        Wout = 14;
        Hout = 14;
        Win = 14;
        Hin = 14;
        numOutLayer = 512;
        kernelSz = 3;
        Allocation pool9_Allocation = Allocation.createSized(rs, Element.F32(rs), Wout * Hout * numOutLayer);
        paraMaxPoolNew(rs, convNet, Wout * Hout * (numOutLayer / 4), kernelSz, Win, Hin, Wout, Hout, 1,
                output_4c_Allocation, pool9_Allocation, "GoogleNet/Logs", "4.txt", "pool9 (ms): ", logEnable);

        if (debug) {
            float[] tmp = new float[Wout * Hout * numOutLayer];
            pool9_Allocation.copyTo(tmp);
            binaryDumper("GoogleNet/Intermediate_rslts/Parallel/4", "pool9.bin", tmp);
        }
        Log.i(tag, "pool9 finished.");
        /***************************************************************************/
        int[] inception_4d_TypeSet = {4};//{1, 2, 4, 6, 7, 8, 9, 12, 14, 16, 18, 24, 28, 36, 72};

        int[] inception_4d_1x1_TypeSet ={4};// {1, 2, 4, 7, 14, 28};
        int[] inception_4d_3x3_TypeSet ={4};// {1, 2, 4, 6, 8, 9, 12, 18, 24, 36, 72};
        int[] inception_4d_5x5_TypeSet = {4};//{1, 2, 4, 8, 16};
        int[] inception_4d_pool_proj_TypeSet = {4};//{1, 2, 4, 8, 16};


        for (int genericConvType : inception_4d_TypeSet) {
            /***************************************************************************/
            /* inception_4d_1x1 */
            int convType = findConvType(inception_4d_1x1_TypeSet, genericConvType);
            Wout = 14;
            Hout = 14;
            Win = 14;
            Hin = 14;
            numInLayer = 512;
            numOutLayer = 112;
            kernelSz = 1;
            int parallelOFMs = numOutLayer / convType;
            paraConvRelu(rs, convNet, /*Weight size F4*/kernelSz * kernelSz * numInLayer * (numOutLayer / 4), /*bias size*/numOutLayer, inception_4d_1x1_w,
                    inception_4d_1x1_b, Wout * Hout * parallelOFMs, kernelSz, Wout, Hout, Win, Hin,/*#Out Layers*/ numOutLayer,
                    /*#In Layers*/ numInLayer, /*S*/ 1, /*pad*/0,/*N_new*/ numInLayer / 4, /*offset*/0, output_4c_Allocation,
                    output_4d_Allocation, convType, parallelOFMs, "GoogleNet/Logs",
                    String.valueOf(convType) + ".txt", "inception_4d_1x1 (ms): ", logEnable && uglyLoop);
            Log.i(tag, "inception_4d_1x1 finished: " + genericConvType + ", " + convType);


            /***************************************************************************/
            /* inception_4d_3x3 */
            convType = findConvType(inception_4d_3x3_TypeSet, genericConvType);
            Wout = 14;
            Hout = 14;
            Win = 14;
            Hin = 14;
            numInLayer = 144;
            numOutLayer = 288;
            kernelSz = 3;
            parallelOFMs = numOutLayer / convType;
            paraConvRelu(rs, convNet, /*Weight size F4*/kernelSz * kernelSz * numInLayer * (numOutLayer / 4), /*bias size*/numOutLayer, inception_4d_3x3_w,
                    inception_4d_3x3_b, Wout * Hout * parallelOFMs, kernelSz, Wout, Hout, Win, Hin,/*#Out Layers*/ numOutLayer,
                    /*#In Layers*/ numInLayer, /*S*/ 1, /*pad*/1,/*N_new*/ numInLayer / 4, /*offset*/14 * 14 * 112, inception_4d_3x3_reduce_Allocation,
                    output_4d_Allocation, convType, parallelOFMs, "GoogleNet/Logs",
                    String.valueOf(convType) + ".txt", "inception_4d_3x3 (ms): ", logEnable && uglyLoop);
            Log.i(tag, "inception_4d_3x3 finished: " + genericConvType + ", " + convType);

            /***************************************************************************/
            /* inception_4d_5x5 */
            convType = findConvType(inception_4d_5x5_TypeSet, genericConvType);
            Wout = 14;
            Hout = 14;
            Win = 14;
            Hin = 14;
            numInLayer = 32;
            numOutLayer = 64;
            kernelSz = 5;
            parallelOFMs = numOutLayer / convType;
            paraConvRelu(rs, convNet, /*Weight size F4*/kernelSz * kernelSz * numInLayer * (numOutLayer / 4), /*bias size*/numOutLayer, inception_4d_5x5_w,
                    inception_4d_5x5_b, Wout * Hout * parallelOFMs, kernelSz, Wout, Hout, Win, Hin,/*#Out Layers*/ numOutLayer,
                    /*#In Layers*/ numInLayer, /*S*/ 1, /*pad*/2,/*N_new*/ numInLayer / 4, /*offset*/14 * 14 * 112 + 14 * 14 * 288,
                    inception_4d_5x5_reduce_Allocation, output_4d_Allocation, convType, parallelOFMs, "GoogleNet/Logs",
                    String.valueOf(convType) + ".txt", "inception_4d_5x5 (ms): ", logEnable && uglyLoop);
            Log.i(tag, "inception_4d_5x5 finished: " + genericConvType + ", " + convType);

            /***************************************************************************/
            /* inception_4d_pool_proj */
            convType = findConvType(inception_4d_pool_proj_TypeSet, genericConvType);
            Wout = 14;
            Hout = 14;
            Win = 14;
            Hin = 14;
            numInLayer = 512;
            numOutLayer = 64;
            kernelSz = 1;
            parallelOFMs = numOutLayer / convType;
            paraConvRelu(rs, convNet, /*Weight size F4*/kernelSz * kernelSz * numInLayer * (numOutLayer / 4), /*bias size*/numOutLayer, inception_4d_pool_proj_w,
                    inception_4d_pool_proj_b, Wout * Hout * parallelOFMs, kernelSz, Wout, Hout, Win, Hin,/*#Out Layers*/ numOutLayer,
                    /*#In Layers*/ numInLayer, /*S*/ 1, /*pad*/0,/*N_new*/ numInLayer / 4,
                    /*offset*/14 * 14 * 112 + 14 * 14 * 288 + 14 * 14 * 64, pool9_Allocation,
                    output_4d_Allocation, convType, parallelOFMs, "GoogleNet/Logs",
                    String.valueOf(convType) + ".txt", "inception_4d_pool_proj (ms): ", logEnable && uglyLoop);
            Log.i(tag, "inception_4d_pool_proj finished: " + genericConvType + ", " + convType);

            if (debug) {
                float[] tmp = new float[14 * 14 * 528];
                output_4d_Allocation.copyTo(tmp);
                binaryDumper("GoogleNet/Intermediate_rslts/Parallel/" + String.valueOf(genericConvType),
                        "output_4d.bin", tmp);
            }
        }

        /* output_4e */
        Allocation output_4e_Allocation = Allocation.createSized(rs, Element.F32(rs), 14 * 14 * 832); //Output size
        /***************************************************************************/
        /* inception_4e_3x3_reduce */
        Wout = 14;
        Hout = 14;
        Win = 14;
        Hin = 14;
        numInLayer = 528;
        numOutLayer = 160;
        kernelSz = 1;
        Allocation inception_4e_3x3_reduce_Allocation = Allocation.createSized(rs, Element.F32(rs), Wout * Hout * numOutLayer); //Output size
        int[] inception_4e_3x3_reduce_TypeSet = {4};//{1, 2, 4, 8, 20, 40};
        for (int convType : inception_4e_3x3_reduce_TypeSet) {
            int parallelOFMs = numOutLayer / convType;
            paraConvRelu(rs, convNet, /*Weight size F4*/kernelSz * kernelSz * numInLayer * (numOutLayer / 4), /*bias size*/numOutLayer, inception_4e_3x3_reduce_w,
                    inception_4e_3x3_reduce_b, Wout * Hout * parallelOFMs, kernelSz, Wout, Hout, Win, Hin,/*#Out Layers*/ numOutLayer,
                    /*#In Layers*/ numInLayer, /*S*/ 1, /*pad*/0,/*N_new*/ numInLayer / 4, /*offset*/0, output_4d_Allocation,
                    inception_4e_3x3_reduce_Allocation, convType, parallelOFMs, "GoogleNet/Logs",
                    String.valueOf(convType) + ".txt", "inception_4e_3x3_reduce (ms): ", logEnable);
            if (debug) {
                float[] tmp = new float[Wout * Hout * numOutLayer];
                inception_4e_3x3_reduce_Allocation.copyTo(tmp);
                binaryDumper("GoogleNet/Intermediate_rslts/Parallel/" + String.valueOf(convType),
                        "inception_4e_3x3_reduce.bin", tmp);
            }
        }
        Log.i(tag, "inception_4e_3x3_reduce finished.");

        /***************************************************************************/
        /* inception_4e_5x5_reduce */
        Wout = 14;
        Hout = 14;
        Win = 14;
        Hin = 14;
        numInLayer = 528;
        numOutLayer = 32;
        kernelSz = 1;
        Allocation inception_4e_5x5_reduce_Allocation = Allocation.createSized(rs, Element.F32(rs), Wout * Hout * numOutLayer); //Output size
        int[] inception_4e_5x5_reduce_TypeSet = {4};//{1, 2, 4, 8};
        for (int convType : inception_4e_5x5_reduce_TypeSet) {
            int parallelOFMs = numOutLayer / convType;
            paraConvRelu(rs, convNet, /*Weight size F4*/kernelSz * kernelSz * numInLayer * (numOutLayer / 4), /*bias size*/numOutLayer, inception_4e_5x5_reduce_w,
                    inception_4e_5x5_reduce_b, Wout * Hout * parallelOFMs, kernelSz, Wout, Hout, Win, Hin,/*#Out Layers*/ numOutLayer,
                    /*#In Layers*/ numInLayer, /*S*/ 1, /*pad*/0,/*N_new*/ numInLayer / 4, /*offset*/0, output_4d_Allocation,
                    inception_4e_5x5_reduce_Allocation, convType, parallelOFMs, "GoogleNet/Logs",
                    String.valueOf(convType) + ".txt", "inception_4e_5x5_reduce (ms): ", logEnable);
            if (debug) {
                float[] tmp = new float[Wout * Hout * numOutLayer];
                inception_4e_5x5_reduce_Allocation.copyTo(tmp);
                binaryDumper("GoogleNet/Intermediate_rslts/Parallel/" + String.valueOf(convType),
                        "inception_4e_5x5_reduce.bin", tmp);
            }
        }

        Log.i(tag, "inception_4e_5x5_reduce finished.");

        /***************************************************************************/
        /* Pool 10 */
        Wout = 14;
        Hout = 14;
        Win = 14;
        Hin = 14;
        numOutLayer = 528;
        kernelSz = 3;
        Allocation pool10_Allocation = Allocation.createSized(rs, Element.F32(rs), Wout * Hout * numOutLayer);

        paraMaxPoolNew(rs, convNet, Wout * Hout * (numOutLayer / 4), kernelSz, Win, Hin, Wout, Hout, 1,
                output_4d_Allocation, pool10_Allocation, "GoogleNet/Logs", "4.txt", "pool10 (ms): ", logEnable);

        if (debug) {
            float[] tmp = new float[Wout * Hout * numOutLayer];
            pool10_Allocation.copyTo(tmp);
            binaryDumper("GoogleNet/Intermediate_rslts/Parallel/4", "pool10.bin", tmp);
        }

        Log.i(tag, "pool10 finished.");
        /***************************************************************************/
        int[] inception_4e_TypeSet = {4};//{1, 2, 4, 5, 8, 10, 16, 20, 32, 40, 64, 80};

        int[] inception_4e_1x1_TypeSet ={4};//{1, 2, 4, 8, 16, 32, 64};
        int[] inception_4e_3x3_TypeSet = {4};//{1, 2, 4, 5, 8, 10, 16, 20, 40, 80};
        int[] inception_4e_5x5_TypeSet = {4};//{1, 2, 4, 8, 16, 32};
        int[] inception_4e_pool_proj_TypeSet = {4};//{1, 2, 4, 8, 16, 32};


        for (int genericConvType : inception_4e_TypeSet) {
            /***************************************************************************/
            /* inception_4e_1x1 */
            int convType = findConvType(inception_4e_1x1_TypeSet, genericConvType);
            Wout = 14;
            Hout = 14;
            Win = 14;
            Hin = 14;
            numInLayer = 528;
            numOutLayer = 256;
            kernelSz = 1;
            int parallelOFMs = numOutLayer / convType;
            paraConvRelu(rs, convNet, /*Weight size F4*/kernelSz * kernelSz * numInLayer * (numOutLayer / 4), /*bias size*/numOutLayer, inception_4e_1x1_w,
                    inception_4e_1x1_b, Wout * Hout * parallelOFMs, kernelSz, Wout, Hout, Win, Hin,/*#Out Layers*/ numOutLayer,
                    /*#In Layers*/ numInLayer, /*S*/ 1, /*pad*/0,/*N_new*/ numInLayer / 4, /*offset*/0, output_4d_Allocation,
                    output_4e_Allocation, convType, parallelOFMs, "GoogleNet/Logs",
                    String.valueOf(convType) + ".txt", "inception_4e_1x1 (ms): ", logEnable && uglyLoop);
            Log.i(tag, "inception_4e_1x1 finished: " + genericConvType + ", " + convType);


            /***************************************************************************/
            /* inception_4e_3x3 */
            convType = findConvType(inception_4e_3x3_TypeSet, genericConvType);
            Wout = 14;
            Hout = 14;
            Win = 14;
            Hin = 14;
            numInLayer = 160;
            numOutLayer = 320;
            kernelSz = 3;
            parallelOFMs = numOutLayer / convType;
            paraConvRelu(rs, convNet, /*Weight size F4*/kernelSz * kernelSz * numInLayer * (numOutLayer / 4), /*bias size*/numOutLayer, inception_4e_3x3_w,
                    inception_4e_3x3_b, Wout * Hout * parallelOFMs, kernelSz, Wout, Hout, Win, Hin,/*#Out Layers*/ numOutLayer,
                    /*#In Layers*/ numInLayer, /*S*/ 1, /*pad*/1,/*N_new*/ numInLayer / 4, /*offset*/14 * 14 * 256, inception_4e_3x3_reduce_Allocation,
                    output_4e_Allocation, convType, parallelOFMs, "GoogleNet/Logs",
                    String.valueOf(convType) + ".txt", "inception_4e_3x3 (ms): ", logEnable && uglyLoop);
            Log.i(tag, "inception_4e_3x3 finished: " + genericConvType + ", " + convType);

            /***************************************************************************/
            /* inception_4e_5x5 */
            convType = findConvType(inception_4e_5x5_TypeSet, genericConvType);
            Wout = 14;
            Hout = 14;
            Win = 14;
            Hin = 14;
            numInLayer = 32;
            numOutLayer = 128;
            kernelSz = 5;
            parallelOFMs = numOutLayer / convType;
            paraConvRelu(rs, convNet, /*Weight size F4*/kernelSz * kernelSz * numInLayer * (numOutLayer / 4), /*bias size*/numOutLayer, inception_4e_5x5_w,
                    inception_4e_5x5_b, Wout * Hout * parallelOFMs, kernelSz, Wout, Hout, Win, Hin,/*#Out Layers*/ numOutLayer,
                    /*#In Layers*/ numInLayer, /*S*/ 1, /*pad*/2,/*N_new*/ numInLayer / 4, /*offset*/14 * 14 * 256 + 14 * 14 * 320,
                    inception_4e_5x5_reduce_Allocation, output_4e_Allocation, convType, parallelOFMs, "GoogleNet/Logs",
                    String.valueOf(convType) + ".txt", "inception_4e_5x5 (ms): ", logEnable && uglyLoop);
            Log.i(tag, "inception_4e_5x5 finished: " + genericConvType + ", " + convType);

            /***************************************************************************/
            /* inception_4e_pool_proj */
            convType = findConvType(inception_4e_pool_proj_TypeSet, genericConvType);
            Wout = 14;
            Hout = 14;
            Win = 14;
            Hin = 14;
            numInLayer = 528;
            numOutLayer = 128;
            kernelSz = 1;
            parallelOFMs = numOutLayer / convType;
            paraConvRelu(rs, convNet, /*Weight size F4*/kernelSz * kernelSz * numInLayer * (numOutLayer / 4), /*bias size*/numOutLayer, inception_4e_pool_proj_w,
                    inception_4e_pool_proj_b, Wout * Hout * parallelOFMs, kernelSz, Wout, Hout, Win, Hin,/*#Out Layers*/ numOutLayer,
                    /*#In Layers*/ numInLayer, /*S*/ 1, /*pad*/0,/*N_new*/ numInLayer / 4,
                    /*offset*/14 * 14 * 256 + 14 * 14 * 320 + 14 * 14 * 128, pool10_Allocation,
                    output_4e_Allocation, convType, parallelOFMs, "GoogleNet/Logs",
                    String.valueOf(convType) + ".txt", "inception_4e_pool_proj (ms): ", logEnable && uglyLoop);
            Log.i(tag, "inception_4e_pool_proj finished: " + genericConvType + ", " + convType);

            if (debug) {
                float[] tmp = new float[14 * 14 * 832];
                output_4e_Allocation.copyTo(tmp);
                binaryDumper("GoogleNet/Intermediate_rslts/Parallel/" + String.valueOf(genericConvType),
                        "output_4e.bin", tmp);
            }
        }

        /***************************************************************************/
        /* Pool 11 */
        Wout = 7;
        Hout = 7;
        Win = 14;
        Hin = 14;
        numOutLayer = 832;
        kernelSz = 3;
        Allocation pool11_Allocation = Allocation.createSized(rs, Element.F32(rs), Wout * Hout * numOutLayer);

        paraMaxPool(rs, convNet, Wout * Hout * (numOutLayer / 4), kernelSz, Win, Hin, Wout, Hout, 2,
                output_4e_Allocation, pool11_Allocation, "GoogleNet/Logs", "4.txt", "Pool11 (ms): ", logEnable);

        if (debug) {
            float[] tmp = new float[Wout * Hout * numOutLayer];
            pool11_Allocation.copyTo(tmp);
            binaryDumper("GoogleNet/Intermediate_rslts/Parallel/4", "pool11.bin", tmp);
        }

        Log.i(tag, "Pool11 finished.");

        /* output_5a */
        Allocation output_5a_Allocation = Allocation.createSized(rs, Element.F32(rs), 7 * 7 * 832); //Output size
        /***************************************************************************/
        /* inception_5a_3x3_reduce */
        Wout = 7;
        Hout = 7;
        Win = 7;
        Hin = 7;
        numInLayer = 832;
        numOutLayer = 160;
        kernelSz = 1;
        Allocation inception_5a_3x3_reduce_Allocation = Allocation.createSized(rs, Element.F32(rs), Wout * Hout * numOutLayer); //Output size
        int[] inception_5a_3x3_reduce_TypeSet = {4};//{1, 2, 4, 8, 20, 40};
        for (int convType : inception_5a_3x3_reduce_TypeSet) {
            int parallelOFMs = numOutLayer / convType;
            paraConvRelu(rs, convNet, /*Weight size F4*/kernelSz * kernelSz * numInLayer * (numOutLayer / 4), /*bias size*/numOutLayer, inception_5a_3x3_reduce_w,
                    inception_5a_3x3_reduce_b, Wout * Hout * parallelOFMs, kernelSz, Wout, Hout, Win, Hin,/*#Out Layers*/ numOutLayer,
                    /*#In Layers*/ numInLayer, /*S*/ 1, /*pad*/0,/*N_new*/ numInLayer / 4, /*offset*/0, pool11_Allocation,
                    inception_5a_3x3_reduce_Allocation, convType, parallelOFMs, "GoogleNet/Logs",
                    String.valueOf(convType) + ".txt", "inception_5a_3x3_reduce (ms): ", logEnable);
            if (debug) {
                float[] tmp = new float[Wout * Hout * numOutLayer];
                inception_5a_3x3_reduce_Allocation.copyTo(tmp);
                binaryDumper("GoogleNet/Intermediate_rslts/Parallel/" + String.valueOf(convType),
                        "inception_5a_3x3_reduce.bin", tmp);
            }
        }

        Log.i(tag, "inception_5a_3x3_reduce finished.");

        /***************************************************************************/
        /* inception_5a_5x5_reduce */
        Wout = 7;
        Hout = 7;
        Win = 7;
        Hin = 7;
        numInLayer = 832;
        numOutLayer = 32;
        kernelSz = 1;
        Allocation inception_5a_5x5_reduce_Allocation = Allocation.createSized(rs, Element.F32(rs), Wout * Hout * numOutLayer); //Output size
        int[] inception_5a_5x5_reduce_TypeSet ={4};// {1, 2, 4, 8};
        for (int convType : inception_5a_5x5_reduce_TypeSet) {
            int parallelOFMs = numOutLayer / convType;
            paraConvRelu(rs, convNet, /*Weight size F4*/kernelSz * kernelSz * numInLayer * (numOutLayer / 4), /*bias size*/numOutLayer, inception_5a_5x5_reduce_w,
                    inception_5a_5x5_reduce_b, Wout * Hout * parallelOFMs, kernelSz, Wout, Hout, Win, Hin,/*#Out Layers*/ numOutLayer,
                    /*#In Layers*/ numInLayer, /*S*/ 1, /*pad*/0,/*N_new*/ numInLayer / 4, /*offset*/0, pool11_Allocation,
                    inception_5a_5x5_reduce_Allocation, convType, parallelOFMs, "GoogleNet/Logs",
                    String.valueOf(convType) + ".txt", "inception_5a_5x5_reduce (ms): ", logEnable);
            if (debug) {
                float[] tmp = new float[Wout * Hout * numOutLayer];
                inception_5a_5x5_reduce_Allocation.copyTo(tmp);
                binaryDumper("GoogleNet/Intermediate_rslts/Parallel/" + String.valueOf(convType),
                        "inception_5a_5x5_reduce.bin", tmp);
            }
        }

        Log.i(tag, "inception_5a_5x5_reduce finished.");

        /***************************************************************************/
        /* Pool 12 */
        Wout = 7;
        Hout = 7;
        Win = 7;
        Hin = 7;
        numOutLayer = 832;
        kernelSz = 3;
        Allocation pool12_Allocation = Allocation.createSized(rs, Element.F32(rs), Wout * Hout * numOutLayer);

        paraMaxPoolNew(rs, convNet, Wout * Hout * (numOutLayer / 4), kernelSz, Win, Hin, Wout, Hout, 1,
                pool11_Allocation, pool12_Allocation, "GoogleNet/Logs", "4.txt", "pool12 (ms): ", logEnable);

        if (debug) {
            float[] tmp = new float[Wout * Hout * numOutLayer];
            pool12_Allocation.copyTo(tmp);
            binaryDumper("GoogleNet/Intermediate_rslts/Parallel/4", "pool12.bin", tmp);
        }

        Log.i(tag, "pool12 finished.");
        /***************************************************************************/
        int[] inception_5a_TypeSet ={4};//{1, 2, 4, 5, 8, 10, 16, 20, 32, 64, 40, 80};

        int[] inception_5a_1x1_TypeSet = {4};//{1, 2, 4, 8, 16, 32, 64};
        int[] inception_5a_3x3_TypeSet = {4};//{1, 2, 4, 5, 8, 10, 16, 20, 40, 80};
        int[] inception_5a_5x5_TypeSet = {4};//{1, 2, 4, 8, 16, 32};
        int[] inception_5a_pool_proj_TypeSet ={4};//{1, 2, 4, 8, 16, 32};


        for (int genericConvType : inception_5a_TypeSet) {
            /***************************************************************************/
            /* inception_5a_1x1 */
            int convType = findConvType(inception_5a_1x1_TypeSet, genericConvType);
            Wout = 7;
            Hout = 7;
            Win = 7;
            Hin = 7;
            numInLayer = 832;
            numOutLayer = 256;
            kernelSz = 1;
            int parallelOFMs = numOutLayer / convType;
            paraConvRelu(rs, convNet, /*Weight size F4*/kernelSz * kernelSz * numInLayer * (numOutLayer / 4), /*bias size*/numOutLayer, inception_5a_1x1_w,
                    inception_5a_1x1_b, Wout * Hout * parallelOFMs, kernelSz, Wout, Hout, Win, Hin,/*#Out Layers*/ numOutLayer,
                    /*#In Layers*/ numInLayer, /*S*/ 1, /*pad*/0,/*N_new*/ numInLayer / 4, /*offset*/0, pool11_Allocation,
                    output_5a_Allocation, convType, parallelOFMs, "GoogleNet/Logs",
                    String.valueOf(convType) + ".txt", "inception_5a_1x1 (ms): ", logEnable && uglyLoop);
            Log.i(tag, "inception_5a_1x1 finished: " + genericConvType + ", " + convType);


            /***************************************************************************/
            /* inception_5a_3x3 */
            convType = findConvType(inception_5a_3x3_TypeSet, genericConvType);
            Wout = 7;
            Hout = 7;
            Win = 7;
            Hin = 7;
            numInLayer = 160;
            numOutLayer = 320;
            kernelSz = 3;
            parallelOFMs = numOutLayer / convType;
            paraConvRelu(rs, convNet, /*Weight size F4*/kernelSz * kernelSz * numInLayer * (numOutLayer / 4), /*bias size*/numOutLayer, inception_5a_3x3_w,
                    inception_5a_3x3_b, Wout * Hout * parallelOFMs, kernelSz, Wout, Hout, Win, Hin,/*#Out Layers*/ numOutLayer,
                    /*#In Layers*/ numInLayer, /*S*/ 1, /*pad*/1,/*N_new*/ numInLayer / 4, /*offset*/7 * 7 * 256, inception_5a_3x3_reduce_Allocation,
                    output_5a_Allocation, convType, parallelOFMs, "GoogleNet/Logs",
                    String.valueOf(convType) + ".txt", "inception_5a_3x3 (ms): ", logEnable && uglyLoop);
            Log.i(tag, "inception_5a_3x3 finished: " + genericConvType + ", " + convType);

            /***************************************************************************/
            /* inception_5a_5x5 */
            convType = findConvType(inception_5a_5x5_TypeSet, genericConvType);
            Wout = 7;
            Hout = 7;
            Win = 7;
            Hin = 7;
            numInLayer = 32;
            numOutLayer = 128;
            kernelSz = 5;
            parallelOFMs = numOutLayer / convType;
            paraConvRelu(rs, convNet, /*Weight size F4*/kernelSz * kernelSz * numInLayer * (numOutLayer / 4), /*bias size*/numOutLayer, inception_5a_5x5_w,
                    inception_5a_5x5_b, Wout * Hout * parallelOFMs, kernelSz, Wout, Hout, Win, Hin,/*#Out Layers*/ numOutLayer,
                    /*#In Layers*/ numInLayer, /*S*/ 1, /*pad*/2,/*N_new*/ numInLayer / 4, /*offset*/7 * 7 * 256 + 7 * 7 * 320,
                    inception_5a_5x5_reduce_Allocation, output_5a_Allocation, convType, parallelOFMs, "GoogleNet/Logs",
                    String.valueOf(convType) + ".txt", "inception_5a_5x5 (ms): ", logEnable && uglyLoop);
            Log.i(tag, "inception_5a_5x5 finished: " + genericConvType + ", " + convType);

            /***************************************************************************/
            /* inception_5a_pool_proj */
            convType = findConvType(inception_5a_pool_proj_TypeSet, genericConvType);
            Wout = 7;
            Hout = 7;
            Win = 7;
            Hin = 7;
            numInLayer = 832;
            numOutLayer = 128;
            kernelSz = 1;
            parallelOFMs = numOutLayer / convType;
            paraConvRelu(rs, convNet, /*Weight size F4*/kernelSz * kernelSz * numInLayer * (numOutLayer / 4), /*bias size*/numOutLayer, inception_5a_pool_proj_w,
                    inception_5a_pool_proj_b, Wout * Hout * parallelOFMs, kernelSz, Wout, Hout, Win, Hin,/*#Out Layers*/ numOutLayer,
                    /*#In Layers*/ numInLayer, /*S*/ 1, /*pad*/0,/*N_new*/ numInLayer / 4,
                    /*offset*/7 * 7 * 256 + 7 * 7 * 320 + 7 * 7 * 128, pool12_Allocation,
                    output_5a_Allocation, convType, parallelOFMs, "GoogleNet/Logs",
                    String.valueOf(convType) + ".txt", "inception_5a_pool_proj (ms): ", logEnable && uglyLoop);
            Log.i(tag, "inception_5a_pool_proj finished: " + genericConvType + ", " + convType);

            if (debug){
                float[] tmp = new float[7 * 7 * 832];
                output_5a_Allocation.copyTo(tmp);
                binaryDumper("GoogleNet/Intermediate_rslts/Parallel/" + String.valueOf(genericConvType),
                        "output_5a.bin", tmp);
            }
        }




        /* output_5b */
        Allocation output_5b_Allocation = Allocation.createSized(rs, Element.F32(rs), 7 * 7 * 1024); //Output size
        /***************************************************************************/
        /* inception_5b_3x3_reduce */
        Wout = 7;
        Hout = 7;
        Win = 7;
        Hin = 7;
        numInLayer = 832;
        numOutLayer = 192;
        kernelSz = 1;
        Allocation inception_5b_3x3_reduce_Allocation = Allocation.createSized(rs, Element.F32(rs), Wout * Hout * numOutLayer); //Output size
        int[] inception_5b_3x3_reduce_TypeSet ={4};//{1, 2, 4, 6, 8, 12, 16, 24, 48};
        for (int convType : inception_5b_3x3_reduce_TypeSet) {
            int parallelOFMs = numOutLayer / convType;
            paraConvRelu(rs, convNet, /*Weight size F4*/kernelSz * kernelSz * numInLayer * (numOutLayer / 4), /*bias size*/numOutLayer, inception_5b_3x3_reduce_w,
                    inception_5b_3x3_reduce_b, Wout * Hout * parallelOFMs, kernelSz, Wout, Hout, Win, Hin,/*#Out Layers*/ numOutLayer,
                    /*#In Layers*/ numInLayer, /*S*/ 1, /*pad*/0,/*N_new*/ numInLayer / 4, /*offset*/0, output_5a_Allocation,
                    inception_5b_3x3_reduce_Allocation, convType, parallelOFMs, "GoogleNet/Logs",
                    String.valueOf(convType) + ".txt", "inception_5b_3x3_reduce (ms): ", logEnable);
            if (debug) {
                float[] tmp = new float[Wout * Hout * numOutLayer];
                inception_5b_3x3_reduce_Allocation.copyTo(tmp);
                binaryDumper("GoogleNet/Intermediate_rslts/Parallel/" + String.valueOf(convType),
                        "inception_5b_3x3_reduce.bin", tmp);
            }
        }

        Log.i(tag, "inception_5b_3x3_reduce finished.");

        /***************************************************************************/
        /* inception_5b_5x5_reduce */
        Wout = 7;
        Hout = 7;
        Win = 7;
        Hin = 7;
        numInLayer = 832;
        numOutLayer = 48;
        kernelSz = 1;
        Allocation inception_5b_5x5_reduce_Allocation = Allocation.createSized(rs, Element.F32(rs), Wout * Hout * numOutLayer); //Output size
        int[] inception_5b_5x5_reduce_TypeSet ={4};//{1, 2, 4, 6, 12};
        for (int convType : inception_5b_5x5_reduce_TypeSet){
            int parallelOFMs = numOutLayer / convType;
            paraConvRelu(rs, convNet, /*Weight size F4*/kernelSz * kernelSz * numInLayer * (numOutLayer / 4), /*bias size*/numOutLayer, inception_5b_5x5_reduce_w,
                    inception_5b_5x5_reduce_b, Wout * Hout * parallelOFMs, kernelSz, Wout, Hout, Win, Hin,/*#Out Layers*/ numOutLayer,
                    /*#In Layers*/ numInLayer, /*S*/ 1, /*pad*/0,/*N_new*/ numInLayer / 4, /*offset*/0, output_5a_Allocation,
                    inception_5b_5x5_reduce_Allocation, convType, parallelOFMs, "GoogleNet/Logs",
                    String.valueOf(convType) + ".txt", "inception_5b_5x5_reduce (ms): ", logEnable);
            if (debug) {
                float[] tmp = new float[Wout * Hout * numOutLayer];
                inception_5b_5x5_reduce_Allocation.copyTo(tmp);
                binaryDumper("GoogleNet/Intermediate_rslts/Parallel/" + String.valueOf(convType),
                        "inception_5b_5x5_reduce.bin", tmp);
            }
        }

        Log.i(tag, "inception_5b_5x5_reduce finished.");

        /***************************************************************************/
        /* Pool 13 */
        Wout = 7;
        Hout = 7;
        Win = 7;
        Hin = 7;
        numOutLayer = 832;
        kernelSz = 3;
        Allocation pool13_Allocation = Allocation.createSized(rs, Element.F32(rs), Wout * Hout * numOutLayer);

        paraMaxPoolNew(rs, convNet, Wout * Hout * (numOutLayer / 4), kernelSz, Win, Hin, Wout, Hout, 1,
                output_5a_Allocation, pool13_Allocation, "GoogleNet/Logs", "4.txt", "pool13 (ms): ", logEnable);

        if (debug) {
            float[] tmp = new float[Wout * Hout * numOutLayer];
            pool13_Allocation.copyTo(tmp);
            binaryDumper("GoogleNet/Intermediate_rslts/Parallel/4", "pool13.bin", tmp);
        }

        Log.i(tag, "pool13 finished.");
        /***************************************************************************/
        int[] inception_5b_TypeSet = {4};//{1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 96};

        int[] inception_5b_1x1_TypeSet = {4};//{1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 96};
        int[] inception_5b_3x3_TypeSet = {4};//{1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 96};
        int[] inception_5b_5x5_TypeSet = {4};//{1, 2, 4, 8, 16, 32};
        int[] inception_5b_pool_proj_TypeSet = {4};//{1, 2, 4, 8, 16, 32};



        for (int genericConvType : inception_5b_TypeSet) {
            /***************************************************************************/
            /* inception_5b_1x1 */
            int convType = findConvType(inception_5b_1x1_TypeSet, genericConvType);
            Wout = 7;
            Hout = 7;
            Win = 7;
            Hin = 7;
            numInLayer = 832;
            numOutLayer = 384;
            kernelSz = 1;
            int parallelOFMs = numOutLayer / convType;
            paraConvRelu(rs, convNet, /*Weight size F4*/kernelSz * kernelSz * numInLayer * (numOutLayer / 4), /*bias size*/numOutLayer, inception_5b_1x1_w,
                    inception_5b_1x1_b, Wout * Hout * parallelOFMs, kernelSz, Wout, Hout, Win, Hin,/*#Out Layers*/ numOutLayer,
                    /*#In Layers*/ numInLayer, /*S*/ 1, /*pad*/0,/*N_new*/ numInLayer / 4, /*offset*/0, output_5a_Allocation,
                    output_5b_Allocation, convType, parallelOFMs, "GoogleNet/Logs",
                    String.valueOf(convType) + ".txt", "inception_5b_1x1 (ms): ", logEnable && uglyLoop);
            Log.i(tag, "inception_5b_1x1 finished: " + genericConvType + ", " + convType);
            /***************************************************************************/
            /* inception_5b_3x3 */
            convType = findConvType(inception_5b_3x3_TypeSet, genericConvType);
            Wout = 7;
            Hout = 7;
            Win = 7;
            Hin = 7;
            numInLayer = 192;
            numOutLayer = 384;
            kernelSz = 3;
            parallelOFMs = numOutLayer / convType;
            paraConvRelu(rs, convNet, /*Weight size F4*/kernelSz * kernelSz * numInLayer * (numOutLayer / 4), /*bias size*/numOutLayer, inception_5b_3x3_w,
                    inception_5b_3x3_b, Wout * Hout * parallelOFMs, kernelSz, Wout, Hout, Win, Hin,/*#Out Layers*/ numOutLayer,
                    /*#In Layers*/ numInLayer, /*S*/ 1, /*pad*/1,/*N_new*/ numInLayer / 4, /*offset*/7 * 7 * 384, inception_5b_3x3_reduce_Allocation,
                    output_5b_Allocation, convType, parallelOFMs, "GoogleNet/Logs",
                    String.valueOf(convType) + ".txt", "inception_5b_3x3 (ms): ", logEnable && uglyLoop);
            Log.i(tag, "inception_5b_3x3 finished: " + genericConvType + ", " + convType);

            /***************************************************************************/
            /* inception_5b_5x5 */
            convType = findConvType(inception_5b_5x5_TypeSet, genericConvType);
            Wout = 7;
            Hout = 7;
            Win = 7;
            Hin = 7;
            numInLayer = 48;
            numOutLayer = 128;
            kernelSz = 5;
            parallelOFMs = numOutLayer / convType;
            paraConvRelu(rs, convNet, /*Weight size F4*/kernelSz * kernelSz * numInLayer * (numOutLayer / 4), /*bias size*/numOutLayer, inception_5b_5x5_w,
                    inception_5b_5x5_b, Wout * Hout * parallelOFMs, kernelSz, Wout, Hout, Win, Hin,/*#Out Layers*/ numOutLayer,
                    /*#In Layers*/ numInLayer, /*S*/ 1, /*pad*/2,/*N_new*/ numInLayer / 4, /*offset*/7 * 7 * 384 + 7 * 7 * 384,
                    inception_5b_5x5_reduce_Allocation, output_5b_Allocation, convType, parallelOFMs, "GoogleNet/Logs",
                    String.valueOf(convType) + ".txt", "inception_5b_5x5 (ms): ", logEnable && uglyLoop);
            Log.i(tag, "inception_5b_5x5 finished: " + genericConvType + ", " + convType);

            /***************************************************************************/
            /* inception_5b_pool_proj */
            convType = findConvType(inception_5b_pool_proj_TypeSet, genericConvType);
            Wout = 7;
            Hout = 7;
            Win = 7;
            Hin = 7;
            numInLayer = 832;
            numOutLayer = 128;
            kernelSz = 1;
            parallelOFMs = numOutLayer / convType;
            paraConvRelu(rs, convNet, /*Weight size F4*/kernelSz * kernelSz * numInLayer * (numOutLayer / 4), /*bias size*/numOutLayer, inception_5b_pool_proj_w,
                    inception_5b_pool_proj_b, Wout * Hout * parallelOFMs, kernelSz, Wout, Hout, Win, Hin,/*#Out Layers*/ numOutLayer,
                    /*#In Layers*/ numInLayer, /*S*/ 1, /*pad*/0,/*N_new*/ numInLayer / 4,
                    /*offset*/7 * 7 * 384 + 7 * 7 * 384 + 7 * 7 * 128, pool13_Allocation,
                    output_5b_Allocation, convType, parallelOFMs, "GoogleNet/Logs",
                    String.valueOf(convType) + ".txt", "inception_5b_pool_proj (ms): ", logEnable && uglyLoop);
            Log.i(tag, "inception_5b_pool_proj finished: " + genericConvType + ", " + convType);


            if (debug) {
                float[] tmp = new float[7 * 7 * 1024];
                output_5b_Allocation.copyTo(tmp);
                binaryDumper("GoogleNet/Intermediate_rslts/Parallel/" + String.valueOf(genericConvType),
                        "output_5b.bin", tmp);
            }
        }


        /***************************************************************************/
        /* AVG Pool */
        Wout = 1;
        Hout = 1;
        Win = 7;
        Hin = 7;
        numOutLayer = 1024;
        kernelSz = 7;
        Allocation pool14_Allocation = Allocation.createSized(rs, Element.F32(rs), Wout * Hout * numOutLayer);

        paraAvgPool(rs, convNet, Wout * Hout * (numOutLayer / 4), kernelSz, Win, Hin, Wout, Hout,

                1,
                output_5b_Allocation, pool14_Allocation, "GoogleNet/Logs", "4.txt", "pool14 (ms): ", logEnable);

        if (debug)

        {
            float[] tmp = new float[Wout * Hout * numOutLayer];
            pool14_Allocation.copyTo(tmp);
            binaryDumper("GoogleNet/Intermediate_rslts/Parallel/4", "pool14.bin", tmp);
        }

        Log.i(tag, "pool14 finished.");

        float[] pool14 = new float[1024];
        pool14_Allocation.copyTo(pool14);

        /***************************************************************************/
        /* FC */
        //TODO change 1000 to 101 to fit my net
        float[] fc1 = new float[101];

        dense(pool14, fc1, loss3_classifier_w, loss3_classifier_b, 1024, 101, "GoogleNet/Logs", "paraSeq.txt", "FC1 (ms)", logEnable);

        if (debug)

            binaryDumper("GoogleNet/Intermediate_rslts/Parallel/Sequential", "fc1.bin", fc1);

        Log.i(tag, "FC1 finished.");

        float[] prob = new float[101];

        softmax(fc1, prob, 101, "GoogleNet/Logs", "paraSeq.txt", "Prob (ms)", false);

        double[] probD = new double[101];
        for (int i = 0;i<101;i++){
            probD[i]=prob[i];
        }

        //put results into the predictions array (first 5)
        predictions = Utility.getTopKPredictions(probD,5);

        if (result)
            binaryDumper("GoogleNet/Intermediate_rslts/Parallel/Sequential", "prob.bin", prob);

        Log.i(tag, "Prob finished.");

        Log.i("CLASSIFICATION ", "End");
        Intent stopProfiling = new Intent("com.quicinc.trepn.stop_profiling");
        context.sendBroadcast(stopProfiling);

    }

    public RecognizedClass[] getPredictions(){
        return predictions;
    }

    /*
    * Sequential implementation of the GoogLeNet: mainly used for benchmarking
    * */
    public void seqGoogleNet(Context context) throws InterruptedException {
        //loadParameters("GoogleNet/Logs", "sequential.txt", "/GoogleNet/Parameters/Normal", logEnable);

        //Thread.sleep(sleepTime);

        Intent createDatabase = new Intent("com.quicinc.trepn.start_profiling");
        createDatabase.putExtra("com.quicinc.trepn.database_file", "Sequential");
        context.sendBroadcast(createDatabase);

        Intent startProfiling = new Intent("com.quicinc.trepn.start_profiling");
        context.sendBroadcast(startProfiling);

        /***************************************************************************/
        /* conv1 */

        float[] conv1 = new float[114 * 114 * 64];
        convRelu(img, conv1, conv1_w, conv1_b, 227, 227, 3, 64, 7, 2, 3, 1,
                "GoogleNet/Logs", "sequential.txt", "Conv1 (ms): ", logEnable);
        if (debug)
            binaryDumper("GoogleNet/Intermediate_rslts/Sequential", "conv1.bin", conv1);
        Log.i(tag, "Conv1 finished.");

        /***************************************************************************/
        /* pool 1 */

        float[] pool1 = new float[57 * 57 * 64];
        maxpoolOUTPUT(conv1, pool1, 114, 114, 64, 3, 2, 57, 57, "GoogleNet/Logs", "sequential.txt",
                "Pool1 (ms)", logEnable);
        if (debug)
            binaryDumper("GoogleNet/Intermediate_rslts/Sequential", "pool1.bin", pool1);
        Log.i(tag, "Pool1 finished.");

        /***************************************************************************/
        /* lrn1 */

        float[] lrn1 = new float[57 * 57 * 64];
        lrn(pool1, lrn1, 57, 57, 64, 5, 0.0001f, 0.75f, 1, "GoogleNet/Logs", "sequential.txt",
                "LRN1 (ms)", logEnable);
        if (debug)
            binaryDumper("GoogleNet/Intermediate_rslts/Sequential", "lrn1.bin", lrn1);
        Log.i(tag, "LRN1 finished.");

        /***************************************************************************/
        /* conv2_3x3_reduce */

        float[] conv2_3x3_reduce = new float[57 * 57 * 64];
        convRelu(lrn1, conv2_3x3_reduce, conv2_3x3_reduce_w, conv2_3x3_reduce_b, 57, 57, 64, 64, 1,
                1, 0, 1, "GoogleNet/Logs", "sequential.txt", "conv2_3x3_reduce (ms): ", logEnable);
        if (debug)
            binaryDumper("GoogleNet/Intermediate_rslts/Sequential", "conv2_3x3_reduce.bin", conv2_3x3_reduce);
        Log.i(tag, "conv2_3x3_reduce finished.");

        /***************************************************************************/
        /* conv2_3x3 */

        float[] conv2_3x3 = new float[57 * 57 * 192];
        convRelu(conv2_3x3_reduce, conv2_3x3, conv2_3x3_w, conv2_3x3_b, 57, 57, 64, 192, 3, 1, 1, 1,
                "GoogleNet/Logs", "sequential.txt", "conv2_3x3 (ms): ", logEnable);
        if (debug)
            binaryDumper("GoogleNet/Intermediate_rslts/Sequential", "conv2_3x3.bin", conv2_3x3);
        Log.i(tag, "conv2_3x3 finished.");

        /***************************************************************************/
        /* LRN 2 */
        float[] lrn2 = new float[57 * 57 * 192];
        lrn(conv2_3x3, lrn2, 57, 57, 192, 5, 0.0001f, 0.75f, 1, "GoogleNet/Logs", "sequential.txt",
                "LRN2 (ms)", logEnable);
        if (debug)
            binaryDumper("GoogleNet/Intermediate_rslts/Sequential", "lrn2.bin", lrn2);
        Log.i(tag, "LRN2 finished.");

        /***************************************************************************/
        /* Pool 2 */
        float[] pool2 = new float[28 * 28 * 192];
        maxpool(lrn2, pool2, 57, 57, 192, 3, 2, "GoogleNet/Logs", "sequential.txt",
                "Pool2 (ms)", logEnable);
        if (debug)
            binaryDumper("GoogleNet/Intermediate_rslts/Sequential", "pool2.bin", pool2);
        Log.i(tag, "Pool2 finished.");

        /***************************************************************************/
        /* inception_3a_1x1 */

        float[] inception_3a_1x1 = new float[28 * 28 * 64];
        convRelu(pool2, inception_3a_1x1, inception_3a_1x1_w, inception_3a_1x1_b, 28, 28, 192, 64,
                1, 1, 0, 1, "GoogleNet/Logs", "sequential.txt", "inception_3a_1x1 (ms): ", logEnable);
        if (debug)
            binaryDumper("GoogleNet/Intermediate_rslts/Sequential", "inception_3a_1x1.bin", inception_3a_1x1);
        Log.i(tag, "inception_3a_1x1 finished.");

        /***************************************************************************/
        /* inception_3a_3x3_reduce */

        float[] inception_3a_3x3_reduce = new float[28 * 28 * 96];
        convRelu(pool2, inception_3a_3x3_reduce, inception_3a_3x3_reduce_w, inception_3a_3x3_reduce_b, 28, 28, 192, 96,
                1, 1, 0, 1, "GoogleNet/Logs", "sequential.txt", "inception_3a_3x3_reduce (ms): ", logEnable);
        if (debug)
            binaryDumper("GoogleNet/Intermediate_rslts/Sequential", "inception_3a_3x3_reduce.bin", inception_3a_3x3_reduce);
        Log.i(tag, "inception_3a_3x3_reduce finished.");

        /***************************************************************************/
        /* inception_3a_5x5_reduce */

        float[] inception_3a_5x5_reduce = new float[28 * 28 * 16];
        convRelu(pool2, inception_3a_5x5_reduce, inception_3a_5x5_reduce_w, inception_3a_5x5_reduce_b, 28, 28, 192, 16,
                1, 1, 0, 1, "GoogleNet/Logs", "sequential.txt", "inception_3a_5x5_reduce (ms): ", logEnable);
        if (debug)
            binaryDumper("GoogleNet/Intermediate_rslts/Sequential", "inception_3a_5x5_reduce.bin", inception_3a_5x5_reduce);
        Log.i(tag, "inception_3a_5x5_reduce finished.");

        /***************************************************************************/
        /* Pool 3 */
        float[] pool3 = new float[28 * 28 * 192];
        maxpoolNew(pool2, pool3, 28, 28, 192, 3, 1, 28, 28, "GoogleNet/Logs", "sequential.txt",
                "Pool3 (ms)", logEnable);
        if (debug)
            binaryDumper("GoogleNet/Intermediate_rslts/Sequential", "pool3.bin", pool3);
        Log.i(tag, "Pool3 finished.");

        /***************************************************************************/
        /* inception_3a_3x3 */

        float[] inception_3a_3x3 = new float[28 * 28 * 128];
        convRelu(inception_3a_3x3_reduce, inception_3a_3x3, inception_3a_3x3_w, inception_3a_3x3_b, 28, 28, 96, 128,
                3, 1, 1, 1, "GoogleNet/Logs", "sequential.txt", "inception_3a_3x3 (ms): ", logEnable);
        if (debug)
            binaryDumper("GoogleNet/Intermediate_rslts/Sequential", "inception_3a_3x3.bin", inception_3a_3x3);
        Log.i(tag, "inception_3a_3x3 finished.");
        //Thread.sleep(sleepTime);

        /***************************************************************************/
        /* inception_3a_5x5 */

        float[] inception_3a_5x5 = new float[28 * 28 * 32];
        convRelu(inception_3a_5x5_reduce, inception_3a_5x5, inception_3a_5x5_w, inception_3a_5x5_b, 28, 28, 16, 32,
                5, 1, 2, 1, "GoogleNet/Logs", "sequential.txt", "inception_3a_5x5 (ms): ", logEnable);
        if (debug)
            binaryDumper("GoogleNet/Intermediate_rslts/Sequential", "inception_3a_5x5.bin", inception_3a_5x5);
        Log.i(tag, "inception_3a_5x5 finished.");

        /***************************************************************************/
        /* inception_3a_pool_proj */

        float[] inception_3a_pool_proj = new float[28 * 28 * 32];
        convRelu(pool3, inception_3a_pool_proj, inception_3a_pool_proj_w, inception_3a_pool_proj_b, 28, 28, 192, 32,
                1, 1, 0, 1, "GoogleNet/Logs", "sequential.txt", "inception_3a_pool_proj (ms): ", logEnable);
        if (debug)
            binaryDumper("GoogleNet/Intermediate_rslts/Sequential", "inception_3a_pool_proj.bin", inception_3a_pool_proj);
        Log.i(tag, "inception_3a_pool_proj finished.");

        /***************************************************************************/
        /* output_3a */

        float[] output_3a = new float[28 * 28 * 256];
        System.arraycopy(inception_3a_1x1, 0, output_3a, 0, 28 * 28 * 64);
        System.arraycopy(inception_3a_3x3, 0, output_3a, 28 * 28 * 64, 28 * 28 * 128);
        System.arraycopy(inception_3a_5x5, 0, output_3a, 28 * 28 * 192, 28 * 28 * 32);
        System.arraycopy(inception_3a_pool_proj, 0, output_3a, 28 * 28 * 224, 28 * 28 * 32);
        if (debug)
            binaryDumper("GoogleNet/Intermediate_rslts/Sequential", "output_3a.bin", output_3a);
        Log.i(tag, "3A finished.");

        /***************************************************************************/
        /* inception_3b_1x1 */

        float[] inception_3b_1x1 = new float[28 * 28 * 128];
        convRelu(output_3a, inception_3b_1x1, inception_3b_1x1_w, inception_3b_1x1_b, 28, 28, 256, 128,
                1, 1, 0, 1, "GoogleNet/Logs", "sequential.txt", "inception_3b_1x1 (ms): ", logEnable);
        if (debug)
            binaryDumper("GoogleNet/Intermediate_rslts/Sequential", "inception_3b_1x1.bin", inception_3b_1x1);
        Log.i(tag, "inception_3b_1x1 finished.");

        /***************************************************************************/
        /* inception_3b_3x3_reduce */

        float[] inception_3b_3x3_reduce = new float[28 * 28 * 128];
        convRelu(output_3a, inception_3b_3x3_reduce, inception_3b_3x3_reduce_w, inception_3b_3x3_reduce_b, 28, 28, 256, 128,
                1, 1, 0, 1, "GoogleNet/Logs", "sequential.txt", "inception_3b_3x3_reduce (ms): ", logEnable);
        if (debug)
            binaryDumper("GoogleNet/Intermediate_rslts/Sequential", "inception_3b_3x3_reduce.bin", inception_3b_3x3_reduce);
        Log.i(tag, "inception_3b_3x3_reduce finished.");

        /***************************************************************************/
        /* inception_3b_5x5_reduce */

        float[] inception_3b_5x5_reduce = new float[28 * 28 * 32];
        convRelu(output_3a, inception_3b_5x5_reduce, inception_3b_5x5_reduce_w, inception_3b_5x5_reduce_b, 28, 28, 256, 32,
                1, 1, 0, 1, "GoogleNet/Logs", "sequential.txt", "inception_3b_5x5_reduce (ms): ", logEnable);
        if (debug)
            binaryDumper("GoogleNet/Intermediate_rslts/Sequential", "inception_3b_5x5_reduce.bin", inception_3b_5x5_reduce);
        Log.i(tag, "inception_3b_5x5_reduce finished.");

        /***************************************************************************/
        /* Pool 4 */
        float[] pool4 = new float[28 * 28 * 256];
        maxpoolNew(output_3a, pool4, 28, 28, 256, 3, 1, 28, 28, "GoogleNet/Logs", "sequential.txt",
                "Pool4 (ms)", logEnable);
        if (debug)
            binaryDumper("GoogleNet/Intermediate_rslts/Sequential", "pool4.bin", pool4);
        Log.i(tag, "Pool4 finished.");

        /***************************************************************************/
        /* inception_3b_3x3 */

        float[] inception_3b_3x3 = new float[28 * 28 * 192];
        convRelu(inception_3b_3x3_reduce, inception_3b_3x3, inception_3b_3x3_w, inception_3b_3x3_b, 28, 28, 128, 192,
                3, 1, 1, 1, "GoogleNet/Logs", "sequential.txt", "inception_3b_3x3 (ms): ", logEnable);
        if (debug)
            binaryDumper("GoogleNet/Intermediate_rslts/Sequential", "inception_3b_3x3.bin", inception_3b_3x3);
        Log.i(tag, "inception_3b_3x3 finished.");
        //Thread.sleep(sleepTime);

        /***************************************************************************/
        /* inception_3b_5x5 */

        float[] inception_3b_5x5 = new float[28 * 28 * 96];
        convRelu(inception_3b_5x5_reduce, inception_3b_5x5, inception_3b_5x5_w, inception_3b_5x5_b, 28, 28, 32, 96,
                5, 1, 2, 1, "GoogleNet/Logs", "sequential.txt", "inception_3b_5x5 (ms): ", logEnable);
        if (debug)
            binaryDumper("GoogleNet/Intermediate_rslts/Sequential", "inception_3b_5x5.bin", inception_3b_5x5);
        Log.i(tag, "inception_3b_5x5 finished.");

        /***************************************************************************/
        /* inception_3b_pool_proj */

        float[] inception_3b_pool_proj = new float[28 * 28 * 64];
        convRelu(pool4, inception_3b_pool_proj, inception_3b_pool_proj_w, inception_3b_pool_proj_b, 28, 28, 256, 64,
                1, 1, 0, 1, "GoogleNet/Logs", "sequential.txt", "inception_3b_pool_proj (ms): ", logEnable);
        if (debug)
            binaryDumper("GoogleNet/Intermediate_rslts/Sequential", "inception_3b_pool_proj.bin", inception_3b_pool_proj);
        Log.i(tag, "inception_3b_pool_proj finished.");

        /***************************************************************************/
        /* output_3b */

        float[] output_3b = new float[28 * 28 * 480];
        System.arraycopy(inception_3b_1x1, 0, output_3b, 0, 28 * 28 * 128);
        System.arraycopy(inception_3b_3x3, 0, output_3b, 28 * 28 * 128, 28 * 28 * 192);
        System.arraycopy(inception_3b_5x5, 0, output_3b, 28 * 28 * 320, 28 * 28 * 96);
        System.arraycopy(inception_3b_pool_proj, 0, output_3b, 28 * 28 * 416, 28 * 28 * 64);
        if (debug)
            binaryDumper("GoogleNet/Intermediate_rslts/Sequential", "output_3b.bin", output_3b);
        Log.i(tag, "3b finished.");

        /***************************************************************************/
        /* Pool 5 */
        float[] pool5 = new float[14 * 14 * 480];
        maxpoolOUTPUT(output_3b, pool5, 28, 28, 480, 3, 2, 14, 14, "GoogleNet/Logs", "sequential.txt",
                "Pool5 (ms)", logEnable);
        if (debug)
            binaryDumper("GoogleNet/Intermediate_rslts/Sequential", "pool5.bin", pool5);
        Log.i(tag, "Pool5 finished.");

        /***************************************************************************/
        /* inception_4a_1x1 */

        float[] inception_4a_1x1 = new float[14 * 14 * 192];
        convRelu(pool5, inception_4a_1x1, inception_4a_1x1_w, inception_4a_1x1_b, 14, 14, 480, 192,
                1, 1, 0, 1, "GoogleNet/Logs", "sequential.txt", "inception_4a_1x1 (ms): ", logEnable);
        if (debug)
            binaryDumper("GoogleNet/Intermediate_rslts/Sequential", "inception_4a_1x1.bin", inception_4a_1x1);
        Log.i(tag, "inception_4a_1x1 finished.");

        /***************************************************************************/
        /* inception_4a_3x3_reduce */

        float[] inception_4a_3x3_reduce = new float[14 * 14 * 96];
        convRelu(pool5, inception_4a_3x3_reduce, inception_4a_3x3_reduce_w, inception_4a_3x3_reduce_b, 14, 14, 480, 96,
                1, 1, 0, 1, "GoogleNet/Logs", "sequential.txt", "inception_4a_3x3_reduce (ms): ", logEnable);
        if (debug)
            binaryDumper("GoogleNet/Intermediate_rslts/Sequential", "inception_4a_3x3_reduce.bin", inception_4a_3x3_reduce);
        Log.i(tag, "inception_4a_3x3_reduce finished.");

        /***************************************************************************/
        /* inception_4a_5x5_reduce */

        float[] inception_4a_5x5_reduce = new float[14 * 14 * 16];
        convRelu(pool5, inception_4a_5x5_reduce, inception_4a_5x5_reduce_w, inception_4a_5x5_reduce_b, 14, 14, 480, 16,
                1, 1, 0, 1, "GoogleNet/Logs", "sequential.txt", "inception_4a_5x5_reduce (ms): ", logEnable);
        if (debug)
            binaryDumper("GoogleNet/Intermediate_rslts/Sequential", "inception_4a_5x5_reduce.bin", inception_4a_5x5_reduce);
        Log.i(tag, "inception_4a_5x5_reduce finished.");

        /***************************************************************************/
        /* Pool 6 */
        float[] pool6 = new float[14 * 14 * 480];
        maxpoolNew(pool5, pool6, 14, 14, 480, 3, 1, 14, 14, "GoogleNet/Logs", "sequential.txt",
                "Pool6 (ms)", logEnable);
        if (debug)
            binaryDumper("GoogleNet/Intermediate_rslts/Sequential", "pool6.bin", pool6);
        Log.i(tag, "Pool6 finished.");

        /***************************************************************************/
        /* inception_4a_3x3 */

        float[] inception_4a_3x3 = new float[14 * 14 * 208];
        convRelu(inception_4a_3x3_reduce, inception_4a_3x3, inception_4a_3x3_w, inception_4a_3x3_b, 14, 14, 96, 208,
                3, 1, 1, 1, "GoogleNet/Logs", "sequential.txt", "inception_4a_3x3 (ms): ", logEnable);
        if (debug)
            binaryDumper("GoogleNet/Intermediate_rslts/Sequential", "inception_4a_3x3.bin", inception_4a_3x3);
        Log.i(tag, "inception_4a_3x3 finished.");
        //Thread.sleep(sleepTime);

        /***************************************************************************/
        /* inception_4a_5x5 */

        float[] inception_4a_5x5 = new float[14 * 14 * 48];
        convRelu(inception_4a_5x5_reduce, inception_4a_5x5, inception_4a_5x5_w, inception_4a_5x5_b, 14, 14, 16, 48,
                5, 1, 2, 1, "GoogleNet/Logs", "sequential.txt", "inception_4a_5x5 (ms): ", logEnable);
        if (debug)
            binaryDumper("GoogleNet/Intermediate_rslts/Sequential", "inception_4a_5x5.bin", inception_4a_5x5);
        Log.i(tag, "inception_4a_5x5 finished.");

        /***************************************************************************/
        /* inception_4a_pool_proj */

        float[] inception_4a_pool_proj = new float[14 * 14 * 64];
        convRelu(pool6, inception_4a_pool_proj, inception_4a_pool_proj_w, inception_4a_pool_proj_b, 14, 14, 480, 64,
                1, 1, 0, 1, "GoogleNet/Logs", "sequential.txt", "inception_4a_pool_proj (ms): ", logEnable);
        if (debug)
            binaryDumper("GoogleNet/Intermediate_rslts/Sequential", "inception_4a_pool_proj.bin", inception_4a_pool_proj);
        Log.i(tag, "inception_4a_pool_proj finished.");

        /***************************************************************************/
        /* output_4a */

        float[] output_4a = new float[14 * 14 * 512];
        System.arraycopy(inception_4a_1x1, 0, output_4a, 0, 14 * 14 * 192);
        System.arraycopy(inception_4a_3x3, 0, output_4a, 14 * 14 * 192, 14 * 14 * 208);
        System.arraycopy(inception_4a_5x5, 0, output_4a, 14 * 14 * 400, 14 * 14 * 48);
        System.arraycopy(inception_4a_pool_proj, 0, output_4a, 14 * 14 * 448, 14 * 14 * 64);
        if (debug)
            binaryDumper("GoogleNet/Intermediate_rslts/Sequential", "output_4a.bin", output_4a);
        Log.i(tag, "4a finished.");

        /***************************************************************************/
        /* inception_4b_1x1 */

        float[] inception_4b_1x1 = new float[14 * 14 * 160];
        convRelu(output_4a, inception_4b_1x1, inception_4b_1x1_w, inception_4b_1x1_b, 14, 14, 512, 160,
                1, 1, 0, 1, "GoogleNet/Logs", "sequential.txt", "inception_4b_1x1 (ms): ", logEnable);
        if (debug)
            binaryDumper("GoogleNet/Intermediate_rslts/Sequential", "inception_4b_1x1.bin", inception_4b_1x1);
        Log.i(tag, "inception_4b_1x1 finished.");

        /***************************************************************************/
        /* inception_4b_3x3_reduce */

        float[] inception_4b_3x3_reduce = new float[14 * 14 * 112];
        convRelu(output_4a, inception_4b_3x3_reduce, inception_4b_3x3_reduce_w, inception_4b_3x3_reduce_b, 14, 14, 512, 112,
                1, 1, 0, 1, "GoogleNet/Logs", "sequential.txt", "inception_4b_3x3_reduce (ms): ", logEnable);
        if (debug)
            binaryDumper("GoogleNet/Intermediate_rslts/Sequential", "inception_4b_3x3_reduce.bin", inception_4b_3x3_reduce);
        Log.i(tag, "inception_4b_3x3_reduce finished.");

        /***************************************************************************/
        /* inception_4b_5x5_reduce */

        float[] inception_4b_5x5_reduce = new float[14 * 14 * 24];
        convRelu(output_4a, inception_4b_5x5_reduce, inception_4b_5x5_reduce_w, inception_4b_5x5_reduce_b, 14, 14, 512, 24,
                1, 1, 0, 1, "GoogleNet/Logs", "sequential.txt", "inception_4b_5x5_reduce (ms): ", logEnable);
        if (debug)
            binaryDumper("GoogleNet/Intermediate_rslts/Sequential", "inception_4b_5x5_reduce.bin", inception_4b_5x5_reduce);
        Log.i(tag, "inception_4b_5x5_reduce finished.");

        /***************************************************************************/
        /* Pool 7 */
        float[] pool7 = new float[14 * 14 * 512];
        maxpoolNew(output_4a, pool7, 14, 14, 512, 3, 1, 14, 14, "GoogleNet/Logs", "sequential.txt",
                "pool7 (ms)", logEnable);
        if (debug)
            binaryDumper("GoogleNet/Intermediate_rslts/Sequential", "pool7.bin", pool7);
        Log.i(tag, "pool7 finished.");

        /***************************************************************************/
        /* inception_4b_3x3 */

        float[] inception_4b_3x3 = new float[14 * 14 * 224];
        convRelu(inception_4b_3x3_reduce, inception_4b_3x3, inception_4b_3x3_w, inception_4b_3x3_b, 14, 14, 112, 224,
                3, 1, 1, 1, "GoogleNet/Logs", "sequential.txt", "inception_4b_3x3 (ms): ", logEnable);
        if (debug)
            binaryDumper("GoogleNet/Intermediate_rslts/Sequential", "inception_4b_3x3.bin", inception_4b_3x3);
        Log.i(tag, "inception_4b_3x3 finished.");
        //Thread.sleep(sleepTime);

        /***************************************************************************/
        /* inception_4b_5x5 */

        float[] inception_4b_5x5 = new float[14 * 14 * 64];
        convRelu(inception_4b_5x5_reduce, inception_4b_5x5, inception_4b_5x5_w, inception_4b_5x5_b, 14, 14, 24, 64,
                5, 1, 2, 1, "GoogleNet/Logs", "sequential.txt", "inception_4b_5x5 (ms): ", logEnable);
        if (debug)
            binaryDumper("GoogleNet/Intermediate_rslts/Sequential", "inception_4b_5x5.bin", inception_4b_5x5);
        Log.i(tag, "inception_4b_5x5 finished.");

        /***************************************************************************/
        /* inception_4b_pool_proj */

        float[] inception_4b_pool_proj = new float[14 * 14 * 64];
        convRelu(pool7, inception_4b_pool_proj, inception_4b_pool_proj_w, inception_4b_pool_proj_b, 14, 14, 512, 64,
                1, 1, 0, 1, "GoogleNet/Logs", "sequential.txt", "inception_4b_pool_proj (ms): ", logEnable);
        if (debug)
            binaryDumper("GoogleNet/Intermediate_rslts/Sequential", "inception_4b_pool_proj.bin", inception_4b_pool_proj);
        Log.i(tag, "inception_4b_pool_proj finished.");

        /***************************************************************************/
        /* output_4b */

        float[] output_4b = new float[14 * 14 * 512];
        System.arraycopy(inception_4b_1x1, 0, output_4b, 0, 14 * 14 * 160);
        System.arraycopy(inception_4b_3x3, 0, output_4b, 14 * 14 * 160, 14 * 14 * 224);
        System.arraycopy(inception_4b_5x5, 0, output_4b, 14 * 14 * 384, 14 * 14 * 64);
        System.arraycopy(inception_4b_pool_proj, 0, output_4b, 14 * 14 * 448, 14 * 14 * 64);
        if (debug)
            binaryDumper("GoogleNet/Intermediate_rslts/Sequential", "output_4b.bin", output_4b);
        Log.i(tag, "4b finished.");

        /***************************************************************************/
        /* inception_4c_1x1 */

        float[] inception_4c_1x1 = new float[14 * 14 * 128];
        convRelu(output_4b, inception_4c_1x1, inception_4c_1x1_w, inception_4c_1x1_b, 14, 14, 512, 128,
                1, 1, 0, 1, "GoogleNet/Logs", "sequential.txt", "inception_4c_1x1 (ms): ", logEnable);
        if (debug)
            binaryDumper("GoogleNet/Intermediate_rslts/Sequential", "inception_4c_1x1.bin", inception_4c_1x1);
        Log.i(tag, "inception_4c_1x1 finished.");

        /***************************************************************************/
        /* inception_4c_3x3_reduce */

        float[] inception_4c_3x3_reduce = new float[14 * 14 * 128];
        convRelu(output_4b, inception_4c_3x3_reduce, inception_4c_3x3_reduce_w, inception_4c_3x3_reduce_b, 14, 14, 512, 128,
                1, 1, 0, 1, "GoogleNet/Logs", "sequential.txt", "inception_4c_3x3_reduce (ms): ", logEnable);
        if (debug)
            binaryDumper("GoogleNet/Intermediate_rslts/Sequential", "inception_4c_3x3_reduce.bin", inception_4c_3x3_reduce);
        Log.i(tag, "inception_4c_3x3_reduce finished.");

        /***************************************************************************/
        /* inception_4c_5x5_reduce */

        float[] inception_4c_5x5_reduce = new float[14 * 14 * 24];
        convRelu(output_4b, inception_4c_5x5_reduce, inception_4c_5x5_reduce_w, inception_4c_5x5_reduce_b, 14, 14, 512, 24,
                1, 1, 0, 1, "GoogleNet/Logs", "sequential.txt", "inception_4c_5x5_reduce (ms): ", logEnable);
        if (debug)
            binaryDumper("GoogleNet/Intermediate_rslts/Sequential", "inception_4c_5x5_reduce.bin", inception_4c_5x5_reduce);
        Log.i(tag, "inception_4c_5x5_reduce finished.");

        /***************************************************************************/
        /* Pool 8 */
        float[] pool8 = new float[14 * 14 * 512];
        maxpoolNew(output_4b, pool8, 14, 14, 512, 3, 1, 14, 14, "GoogleNet/Logs", "sequential.txt",
                "pool8 (ms)", logEnable);
        if (debug)
            binaryDumper("GoogleNet/Intermediate_rslts/Sequential", "pool8.bin", pool8);
        Log.i(tag, "pool8 finished.");

        /***************************************************************************/
        /* inception_4c_3x3 */

        float[] inception_4c_3x3 = new float[14 * 14 * 256];
        convRelu(inception_4c_3x3_reduce, inception_4c_3x3, inception_4c_3x3_w, inception_4c_3x3_b, 14, 14, 128, 256,
                3, 1, 1, 1, "GoogleNet/Logs", "sequential.txt", "inception_4c_3x3 (ms): ", logEnable);
        if (debug)
            binaryDumper("GoogleNet/Intermediate_rslts/Sequential", "inception_4c_3x3.bin", inception_4c_3x3);
        Log.i(tag, "inception_4c_3x3 finished.");
        //Thread.sleep(sleepTime);

        /***************************************************************************/
        /* inception_4c_5x5 */

        float[] inception_4c_5x5 = new float[14 * 14 * 64];
        convRelu(inception_4c_5x5_reduce, inception_4c_5x5, inception_4c_5x5_w, inception_4c_5x5_b, 14, 14, 24, 64,
                5, 1, 2, 1, "GoogleNet/Logs", "sequential.txt", "inception_4c_5x5 (ms): ", logEnable);
        if (debug)
            binaryDumper("GoogleNet/Intermediate_rslts/Sequential", "inception_4c_5x5.bin", inception_4c_5x5);
        Log.i(tag, "inception_4c_5x5 finished.");

        /***************************************************************************/
        /* inception_4c_pool_proj */

        float[] inception_4c_pool_proj = new float[14 * 14 * 64];
        convRelu(pool8, inception_4c_pool_proj, inception_4c_pool_proj_w, inception_4c_pool_proj_b, 14, 14, 512, 64,
                1, 1, 0, 1, "GoogleNet/Logs", "sequential.txt", "inception_4c_pool_proj (ms): ", logEnable);
        if (debug)
            binaryDumper("GoogleNet/Intermediate_rslts/Sequential", "inception_4c_pool_proj.bin", inception_4c_pool_proj);
        Log.i(tag, "inception_4c_pool_proj finished.");

        /***************************************************************************/
        /* output_4c */

        float[] output_4c = new float[14 * 14 * 512];
        System.arraycopy(inception_4c_1x1, 0, output_4c, 0, 14 * 14 * 128);
        System.arraycopy(inception_4c_3x3, 0, output_4c, 14 * 14 * 128, 14 * 14 * 256);
        System.arraycopy(inception_4c_5x5, 0, output_4c, 14 * 14 * 384, 14 * 14 * 64);
        System.arraycopy(inception_4c_pool_proj, 0, output_4c, 14 * 14 * 448, 14 * 14 * 64);
        if (debug)
            binaryDumper("GoogleNet/Intermediate_rslts/Sequential", "output_4c.bin", output_4c);
        Log.i(tag, "4c finished.");

        /***************************************************************************/
        /* inception_4d_1x1 */

        float[] inception_4d_1x1 = new float[14 * 14 * 112];
        convRelu(output_4c, inception_4d_1x1, inception_4d_1x1_w, inception_4d_1x1_b, 14, 14, 512, 112,
                1, 1, 0, 1, "GoogleNet/Logs", "sequential.txt", "inception_4d_1x1 (ms): ", logEnable);
        if (debug)
            binaryDumper("GoogleNet/Intermediate_rslts/Sequential", "inception_4d_1x1.bin", inception_4d_1x1);
        Log.i(tag, "inception_4d_1x1 finished.");

        /***************************************************************************/
        /* inception_4d_3x3_reduce */

        float[] inception_4d_3x3_reduce = new float[14 * 14 * 144];
        convRelu(output_4c, inception_4d_3x3_reduce, inception_4d_3x3_reduce_w, inception_4d_3x3_reduce_b, 14, 14, 512, 144,
                1, 1, 0, 1, "GoogleNet/Logs", "sequential.txt", "inception_4d_3x3_reduce (ms): ", logEnable);
        if (debug)
            binaryDumper("GoogleNet/Intermediate_rslts/Sequential", "inception_4d_3x3_reduce.bin", inception_4d_3x3_reduce);
        Log.i(tag, "inception_4d_3x3_reduce finished.");

        /***************************************************************************/
        /* inception_4d_5x5_reduce */

        float[] inception_4d_5x5_reduce = new float[14 * 14 * 32];
        convRelu(output_4c, inception_4d_5x5_reduce, inception_4d_5x5_reduce_w, inception_4d_5x5_reduce_b, 14, 14, 512, 32,
                1, 1, 0, 1, "GoogleNet/Logs", "sequential.txt", "inception_4d_5x5_reduce (ms): ", logEnable);
        if (debug)
            binaryDumper("GoogleNet/Intermediate_rslts/Sequential", "inception_4d_5x5_reduce.bin", inception_4d_5x5_reduce);
        Log.i(tag, "inception_4d_5x5_reduce finished.");

        /***************************************************************************/
        /* Pool 9 */
        float[] pool9 = new float[14 * 14 * 512];
        maxpoolNew(output_4c, pool9, 14, 14, 512, 3, 1, 14, 14, "GoogleNet/Logs", "sequential.txt",
                "pool9 (ms)", logEnable);
        if (debug)
            binaryDumper("GoogleNet/Intermediate_rslts/Sequential", "pool9.bin", pool9);
        Log.i(tag, "pool9 finished.");

        /***************************************************************************/
        /* inception_4d_3x3 */

        float[] inception_4d_3x3 = new float[14 * 14 * 288];
        convRelu(inception_4d_3x3_reduce, inception_4d_3x3, inception_4d_3x3_w, inception_4d_3x3_b, 14, 14, 144, 288,
                3, 1, 1, 1, "GoogleNet/Logs", "sequential.txt", "inception_4d_3x3 (ms): ", logEnable);
        if (debug)
            binaryDumper("GoogleNet/Intermediate_rslts/Sequential", "inception_4d_3x3.bin", inception_4d_3x3);
        Log.i(tag, "inception_4d_3x3 finished.");
        //Thread.sleep(sleepTime);

        /***************************************************************************/
        /* inception_4d_5x5 */

        float[] inception_4d_5x5 = new float[14 * 14 * 64];
        convRelu(inception_4d_5x5_reduce, inception_4d_5x5, inception_4d_5x5_w, inception_4d_5x5_b, 14, 14, 32, 64,
                5, 1, 2, 1, "GoogleNet/Logs", "sequential.txt", "inception_4d_5x5 (ms): ", logEnable);
        if (debug)
            binaryDumper("GoogleNet/Intermediate_rslts/Sequential", "inception_4d_5x5.bin", inception_4d_5x5);
        Log.i(tag, "inception_4d_5x5 finished.");

        /***************************************************************************/
        /* inception_4d_pool_proj */

        float[] inception_4d_pool_proj = new float[14 * 14 * 64];
        convRelu(pool9, inception_4d_pool_proj, inception_4d_pool_proj_w, inception_4d_pool_proj_b, 14, 14, 512, 64,
                1, 1, 0, 1, "GoogleNet/Logs", "sequential.txt", "inception_4d_pool_proj (ms): ", logEnable);
        if (debug)
            binaryDumper("GoogleNet/Intermediate_rslts/Sequential", "inception_4d_pool_proj.bin", inception_4d_pool_proj);
        Log.i(tag, "inception_4d_pool_proj finished.");

        /***************************************************************************/
        /* output_4d */

        float[] output_4d = new float[14 * 14 * 528];
        System.arraycopy(inception_4d_1x1, 0, output_4d, 0, 14 * 14 * 112);
        System.arraycopy(inception_4d_3x3, 0, output_4d, 14 * 14 * 112, 14 * 14 * 288);
        System.arraycopy(inception_4d_5x5, 0, output_4d, 14 * 14 * 400, 14 * 14 * 64);
        System.arraycopy(inception_4d_pool_proj, 0, output_4d, 14 * 14 * 464, 14 * 14 * 64);
        if (debug)
            binaryDumper("GoogleNet/Intermediate_rslts/Sequential", "output_4d.bin", output_4d);
        Log.i(tag, "4d finished.");

        /***************************************************************************/
        /* inception_4e_1x1 */

        float[] inception_4e_1x1 = new float[14 * 14 * 256];
        convRelu(output_4d, inception_4e_1x1, inception_4e_1x1_w, inception_4e_1x1_b, 14, 14, 528, 256,
                1, 1, 0, 1, "GoogleNet/Logs", "sequential.txt", "inception_4e_1x1 (ms): ", logEnable);
        if (debug)
            binaryDumper("GoogleNet/Intermediate_rslts/Sequential", "inception_4e_1x1.bin", inception_4e_1x1);
        Log.i(tag, "inception_4e_1x1 finished.");

        /***************************************************************************/
        /* inception_4e_3x3_reduce */

        float[] inception_4e_3x3_reduce = new float[14 * 14 * 160];
        convRelu(output_4d, inception_4e_3x3_reduce, inception_4e_3x3_reduce_w, inception_4e_3x3_reduce_b, 14, 14, 528, 160,
                1, 1, 0, 1, "GoogleNet/Logs", "sequential.txt", "inception_4e_3x3_reduce (ms): ", logEnable);
        if (debug)
            binaryDumper("GoogleNet/Intermediate_rslts/Sequential", "inception_4e_3x3_reduce.bin", inception_4e_3x3_reduce);
        Log.i(tag, "inception_4e_3x3_reduce finished.");

        /***************************************************************************/
        /* inception_4e_5x5_reduce */

        float[] inception_4e_5x5_reduce = new float[14 * 14 * 32];
        convRelu(output_4d, inception_4e_5x5_reduce, inception_4e_5x5_reduce_w, inception_4e_5x5_reduce_b, 14, 14, 528, 32,
                1, 1, 0, 1, "GoogleNet/Logs", "sequential.txt", "inception_4e_5x5_reduce (ms): ", logEnable);
        if (debug)
            binaryDumper("GoogleNet/Intermediate_rslts/Sequential", "inception_4e_5x5_reduce.bin", inception_4e_5x5_reduce);
        Log.i(tag, "inception_4e_5x5_reduce finished.");

        /***************************************************************************/
        /* Pool 10 */
        float[] pool10 = new float[14 * 14 * 528];
        maxpoolNew(output_4d, pool10, 14, 14, 528, 3, 1, 14, 14, "GoogleNet/Logs", "sequential.txt",
                "pool10 (ms)", logEnable);
        if (debug)
            binaryDumper("GoogleNet/Intermediate_rslts/Sequential", "pool10.bin", pool10);
        Log.i(tag, "pool10 finished.");

        /***************************************************************************/
        /* inception_4e_3x3 */

        float[] inception_4e_3x3 = new float[14 * 14 * 320];
        convRelu(inception_4e_3x3_reduce, inception_4e_3x3, inception_4e_3x3_w, inception_4e_3x3_b, 14, 14, 160, 320,
                3, 1, 1, 1, "GoogleNet/Logs", "sequential.txt", "inception_4e_3x3 (ms): ", logEnable);
        if (debug)
            binaryDumper("GoogleNet/Intermediate_rslts/Sequential", "inception_4e_3x3.bin", inception_4e_3x3);
        Log.i(tag, "inception_4e_3x3 finished.");
        //Thread.sleep(sleepTime);

        /***************************************************************************/
        /* inception_4e_5x5 */

        float[] inception_4e_5x5 = new float[14 * 14 * 128];
        convRelu(inception_4e_5x5_reduce, inception_4e_5x5, inception_4e_5x5_w, inception_4e_5x5_b, 14, 14, 32, 128,
                5, 1, 2, 1, "GoogleNet/Logs", "sequential.txt", "inception_4e_5x5 (ms): ", logEnable);
        if (debug)
            binaryDumper("GoogleNet/Intermediate_rslts/Sequential", "inception_4e_5x5.bin", inception_4e_5x5);
        Log.i(tag, "inception_4e_5x5 finished.");

        /***************************************************************************/
        /* inception_4e_pool_proj */

        float[] inception_4e_pool_proj = new float[14 * 14 * 128];
        convRelu(pool10, inception_4e_pool_proj, inception_4e_pool_proj_w, inception_4e_pool_proj_b, 14, 14, 528, 128,
                1, 1, 0, 1, "GoogleNet/Logs", "sequential.txt", "inception_4e_pool_proj (ms): ", logEnable);
        if (debug)
            binaryDumper("GoogleNet/Intermediate_rslts/Sequential", "inception_4e_pool_proj.bin", inception_4e_pool_proj);
        Log.i(tag, "inception_4e_pool_proj finished.");

        /***************************************************************************/
        /* output_4e */

        float[] output_4e = new float[14 * 14 * 832];
        System.arraycopy(inception_4e_1x1, 0, output_4e, 0, 14 * 14 * 256);
        System.arraycopy(inception_4e_3x3, 0, output_4e, 14 * 14 * 256, 14 * 14 * 320);
        System.arraycopy(inception_4e_5x5, 0, output_4e, 14 * 14 * 576, 14 * 14 * 128);
        System.arraycopy(inception_4e_pool_proj, 0, output_4e, 14 * 14 * 704, 14 * 14 * 128);
        if (debug)
            binaryDumper("GoogleNet/Intermediate_rslts/Sequential", "output_4e.bin", output_4e);
        Log.i(tag, "4e finished.");

        /***************************************************************************/
        /* Pool 11 */
        float[] pool11 = new float[7 * 7 * 832];
        maxpoolOUTPUT(output_4e, pool11, 14, 14, 832, 3, 2, 7, 7, "GoogleNet/Logs", "sequential.txt",
                "pool11 (ms)", logEnable);
        if (debug)
            binaryDumper("GoogleNet/Intermediate_rslts/Sequential", "pool11.bin", pool11);
        Log.i(tag, "pool11 finished.");

        /***************************************************************************/
        /* inception_5a_1x1 */

        float[] inception_5a_1x1 = new float[7 * 7 * 256];
        convRelu(pool11, inception_5a_1x1, inception_5a_1x1_w, inception_5a_1x1_b, 7, 7, 832, 256,
                1, 1, 0, 1, "GoogleNet/Logs", "sequential.txt", "inception_5a_1x1 (ms): ", logEnable);
        if (debug)
            binaryDumper("GoogleNet/Intermediate_rslts/Sequential", "inception_5a_1x1.bin", inception_5a_1x1);
        Log.i(tag, "inception_5a_1x1 finished.");

        /***************************************************************************/
        /* inception_5a_3x3_reduce */

        float[] inception_5a_3x3_reduce = new float[7 * 7 * 160];
        convRelu(pool11, inception_5a_3x3_reduce, inception_5a_3x3_reduce_w, inception_5a_3x3_reduce_b, 7, 7, 832, 160,
                1, 1, 0, 1, "GoogleNet/Logs", "sequential.txt", "inception_5a_3x3_reduce (ms): ", logEnable);
        if (debug)
            binaryDumper("GoogleNet/Intermediate_rslts/Sequential", "inception_5a_3x3_reduce.bin", inception_5a_3x3_reduce);
        Log.i(tag, "inception_5a_3x3_reduce finished.");

        /***************************************************************************/
        /* inception_5a_5x5_reduce */

        float[] inception_5a_5x5_reduce = new float[7 * 7 * 32];
        convRelu(pool11, inception_5a_5x5_reduce, inception_5a_5x5_reduce_w, inception_5a_5x5_reduce_b, 7, 7, 832, 32,
                1, 1, 0, 1, "GoogleNet/Logs", "sequential.txt", "inception_5a_5x5_reduce (ms): ", logEnable);
        if (debug)
            binaryDumper("GoogleNet/Intermediate_rslts/Sequential", "inception_5a_5x5_reduce.bin", inception_5a_5x5_reduce);
        Log.i(tag, "inception_5a_5x5_reduce finished.");

        /***************************************************************************/
        /* Pool 12 */
        float[] pool12 = new float[7 * 7 * 832];
        maxpoolNew(pool11, pool12, 7, 7, 832, 3, 1, 7, 7, "GoogleNet/Logs", "sequential.txt",
                "pool12 (ms)", logEnable);
        if (debug)
            binaryDumper("GoogleNet/Intermediate_rslts/Sequential", "pool12.bin", pool12);
        Log.i(tag, "pool12 finished.");

        /***************************************************************************/
        /* inception_5a_3x3 */

        float[] inception_5a_3x3 = new float[7 * 7 * 320];
        convRelu(inception_5a_3x3_reduce, inception_5a_3x3, inception_5a_3x3_w, inception_5a_3x3_b, 7, 7, 160, 320,
                3, 1, 1, 1, "GoogleNet/Logs", "sequential.txt", "inception_5a_3x3 (ms): ", logEnable);
        if (debug)
            binaryDumper("GoogleNet/Intermediate_rslts/Sequential", "inception_5a_3x3.bin", inception_5a_3x3);
        Log.i(tag, "inception_5a_3x3 finished.");
        //Thread.sleep(sleepTime);

        /***************************************************************************/
        /* inception_5a_5x5 */

        float[] inception_5a_5x5 = new float[7 * 7 * 128];
        convRelu(inception_5a_5x5_reduce, inception_5a_5x5, inception_5a_5x5_w, inception_5a_5x5_b, 7, 7, 32, 128,
                5, 1, 2, 1, "GoogleNet/Logs", "sequential.txt", "inception_5a_5x5 (ms): ", logEnable);
        if (debug)
            binaryDumper("GoogleNet/Intermediate_rslts/Sequential", "inception_5a_5x5.bin", inception_5a_5x5);
        Log.i(tag, "inception_5a_5x5 finished.");

        /***************************************************************************/
        /* inception_5a_pool_proj */

        float[] inception_5a_pool_proj = new float[7 * 7 * 128];
        convRelu(pool12, inception_5a_pool_proj, inception_5a_pool_proj_w, inception_5a_pool_proj_b, 7, 7, 832, 128,
                1, 1, 0, 1, "GoogleNet/Logs", "sequential.txt", "inception_5a_pool_proj (ms): ", logEnable);
        if (debug)
            binaryDumper("GoogleNet/Intermediate_rslts/Sequential", "inception_5a_pool_proj.bin", inception_5a_pool_proj);
        Log.i(tag, "inception_5a_pool_proj finished.");

        /***************************************************************************/
        /* output_5a */

        float[] output_5a = new float[7 * 7 * 832];
        System.arraycopy(inception_5a_1x1, 0, output_5a, 0, 7 * 7 * 256);
        System.arraycopy(inception_5a_3x3, 0, output_5a, 7 * 7 * 256, 7 * 7 * 320);
        System.arraycopy(inception_5a_5x5, 0, output_5a, 7 * 7 * 576, 7 * 7 * 128);
        System.arraycopy(inception_5a_pool_proj, 0, output_5a, 7 * 7 * 704, 7 * 7 * 128);
        if (debug)
            binaryDumper("GoogleNet/Intermediate_rslts/Sequential", "output_5a.bin", output_5a);
        Log.i(tag, "5a finished.");

        /***************************************************************************/
        /* inception_5b_1x1 */

        float[] inception_5b_1x1 = new float[7 * 7 * 384];
        convRelu(output_5a, inception_5b_1x1, inception_5b_1x1_w, inception_5b_1x1_b, 7, 7, 832, 384,
                1, 1, 0, 1, "GoogleNet/Logs", "sequential.txt", "inception_5b_1x1 (ms): ", logEnable);
        if (debug)
            binaryDumper("GoogleNet/Intermediate_rslts/Sequential", "inception_5b_1x1.bin", inception_5b_1x1);
        Log.i(tag, "inception_5b_1x1 finished.");

        /***************************************************************************/
        /* inception_5b_3x3_reduce */

        float[] inception_5b_3x3_reduce = new float[7 * 7 * 192];
        convRelu(output_5a, inception_5b_3x3_reduce, inception_5b_3x3_reduce_w, inception_5b_3x3_reduce_b, 7, 7, 832, 192,
                1, 1, 0, 1, "GoogleNet/Logs", "sequential.txt", "inception_5b_3x3_reduce (ms): ", logEnable);
        if (debug)
            binaryDumper("GoogleNet/Intermediate_rslts/Sequential", "inception_5b_3x3_reduce.bin", inception_5b_3x3_reduce);
        Log.i(tag, "inception_5b_3x3_reduce finished.");

        /***************************************************************************/
        /* inception_5b_5x5_reduce */

        float[] inception_5b_5x5_reduce = new float[7 * 7 * 48];
        convRelu(output_5a, inception_5b_5x5_reduce, inception_5b_5x5_reduce_w, inception_5b_5x5_reduce_b, 7, 7, 832, 48,
                1, 1, 0, 1, "GoogleNet/Logs", "sequential.txt", "inception_5b_5x5_reduce (ms): ", logEnable);
        if (debug)
            binaryDumper("GoogleNet/Intermediate_rslts/Sequential", "inception_5b_5x5_reduce.bin", inception_5b_5x5_reduce);
        Log.i(tag, "inception_5b_5x5_reduce finished.");

        /***************************************************************************/
        /* Pool 13 */
        float[] pool13 = new float[7 * 7 * 832];
        maxpoolNew(output_5a, pool13, 7, 7, 832, 3, 1, 7, 7, "GoogleNet/Logs", "sequential.txt",
                "pool13 (ms)", logEnable);
        if (debug)
            binaryDumper("GoogleNet/Intermediate_rslts/Sequential", "pool13.bin", pool13);
        Log.i(tag, "pool13 finished.");

        /***************************************************************************/
        /* inception_5b_3x3 */

        float[] inception_5b_3x3 = new float[7 * 7 * 384];
        convRelu(inception_5b_3x3_reduce, inception_5b_3x3, inception_5b_3x3_w, inception_5b_3x3_b, 7, 7, 192, 384,
                3, 1, 1, 1, "GoogleNet/Logs", "sequential.txt", "inception_5b_3x3 (ms): ", logEnable);
        if (debug)
            binaryDumper("GoogleNet/Intermediate_rslts/Sequential", "inception_5b_3x3.bin", inception_5b_3x3);
        Log.i(tag, "inception_5b_3x3 finished.");
        //Thread.sleep(sleepTime);

        /***************************************************************************/
        /* inception_5b_5x5 */

        float[] inception_5b_5x5 = new float[7 * 7 * 128];
        convRelu(inception_5b_5x5_reduce, inception_5b_5x5, inception_5b_5x5_w, inception_5b_5x5_b, 7, 7, 48, 128,
                5, 1, 2, 1, "GoogleNet/Logs", "sequential.txt", "inception_5b_5x5 (ms): ", logEnable);
        if (debug)
            binaryDumper("GoogleNet/Intermediate_rslts/Sequential", "inception_5b_5x5.bin", inception_5b_5x5);
        Log.i(tag, "inception_5b_5x5 finished.");

        /***************************************************************************/
        /* inception_5b_pool_proj */

        float[] inception_5b_pool_proj = new float[7 * 7 * 128];
        convRelu(pool13, inception_5b_pool_proj, inception_5b_pool_proj_w, inception_5b_pool_proj_b, 7, 7, 832, 128,
                1, 1, 0, 1, "GoogleNet/Logs", "sequential.txt", "inception_5b_pool_proj (ms): ", logEnable);
        if (debug)
            binaryDumper("GoogleNet/Intermediate_rslts/Sequential", "inception_5b_pool_proj.bin", inception_5b_pool_proj);
        Log.i(tag, "inception_5b_pool_proj finished.");

        /***************************************************************************/
        /* output_5b */

        float[] output_5b = new float[7 * 7 * 1024];
        System.arraycopy(inception_5b_1x1, 0, output_5b, 0, 7 * 7 * 384);
        System.arraycopy(inception_5b_3x3, 0, output_5b, 7 * 7 * 384, 7 * 7 * 384);
        System.arraycopy(inception_5b_5x5, 0, output_5b, 7 * 7 * 768, 7 * 7 * 128);
        System.arraycopy(inception_5b_pool_proj, 0, output_5b, 7 * 7 * 896, 7 * 7 * 128);
        if (debug)
            binaryDumper("GoogleNet/Intermediate_rslts/Sequential", "output_5b.bin", output_5b);
        Log.i(tag, "5b finished.");

        /***************************************************************************/
        /* AVG Pool */

        float[] pool14 = new float[1024];
        avgpool(output_5b, pool14, 7, 7, 1024, 7, 1, "GoogleNet/Logs", "sequential.txt", "Pool14 (ms)", logEnable);
        if (debug)
            binaryDumper("GoogleNet/Intermediate_rslts/Sequential", "pool14.bin", pool14);
        Log.i(tag, "Pool14 finished.");

        /***************************************************************************/
        /* FC */
        float[] fc1 = new float[1000];
        dense(pool14, fc1, loss3_classifier_w, loss3_classifier_b, 1024, 1000, "GoogleNet/Logs", "sequential.txt", "FC1 (ms)", logEnable);
        if (debug)
            binaryDumper("GoogleNet/Intermediate_rslts/Sequential", "fc1.bin", fc1);
        Log.i(tag, "FC1 finished.");

        float[] prob = new float[1000];
        softmax(fc1, prob, 1000, "GoogleNet/Logs", "sequential.txt", "Prob (ms)", false);

        if (result)
            binaryDumper("GoogleNet/Intermediate_rslts/Sequential", "prob.bin", prob);
        Log.i(tag, "Prob finished.");

        Intent stopProfiling = new Intent("com.quicinc.trepn.stop_profiling");
        context.sendBroadcast(stopProfiling);
    }

    //LAYERS
    private void paraAvgPool(RenderScript rs, ScriptC_convNet convNet, int exeSpaceSz, int K,
                             int Win, int Hin, int Wout, int Hout, int S, Allocation inputAllocation,
                             Allocation outputAllocation, String logFolder,
                             String logFile, String logMsg, boolean logEnable) {
        long start = System.currentTimeMillis();

        convNet.set_K(K);
        convNet.set_Wout(Wout);
        convNet.set_Hout(Hout);
        convNet.set_Hin(Hin);
        convNet.set_Win(Win);
        convNet.set_S(S);

        Allocation exeSpace = Allocation.createSized(rs, Element.F32(rs), exeSpaceSz);
        convNet.set_in(inputAllocation);
        convNet.set_output(outputAllocation);

        convNet.forEach_avgPool(exeSpace);

        long end = System.currentTimeMillis();
        if (logEnable)
            logWriter(logFolder, logFile, logMsg + "\t" + String.valueOf(end - start) + "\n");
    }

    private void paraMaxPoolNew(RenderScript rs, ScriptC_convNet convNet, int exeSpaceSz, int K,
                                int Win, int Hin, int Wout, int Hout, int S, Allocation inputAllocation,
                                Allocation outputAllocation, String logFolder,
                                String logFile, String logMsg, boolean logEnable) {
        long start = System.currentTimeMillis();

        convNet.set_K(K);
        convNet.set_Wout(Wout);
        convNet.set_Hout(Hout);
        convNet.set_Hin(Hin);
        convNet.set_Win(Win);
        convNet.set_S(S);

        Allocation exeSpace = Allocation.createSized(rs, Element.F32(rs), exeSpaceSz);
        convNet.set_in(inputAllocation);
        convNet.set_output(outputAllocation);

        convNet.forEach_maxPoolNew(exeSpace);

        long end = System.currentTimeMillis();
        if (logEnable)
            logWriter(logFolder, logFile, logMsg + "\t" + String.valueOf(end - start) + "\n");

    }

    private void paraMaxPool(RenderScript rs, ScriptC_convNet convNet, int exeSpaceSz, int K,
                             int Win, int Hin, int Wout, int Hout, int S, Allocation inputAllocation,
                             Allocation outputAllocation, String logFolder,
                             String logFile, String logMsg, boolean logEnable) {
        long start = System.currentTimeMillis();

        convNet.set_K(K);
        convNet.set_Wout(Wout);
        convNet.set_Hout(Hout);
        convNet.set_Hin(Hin);
        convNet.set_Win(Win);
        convNet.set_S(S);

        Allocation exeSpace = Allocation.createSized(rs, Element.F32(rs), exeSpaceSz);
        convNet.set_in(inputAllocation);
        convNet.set_output(outputAllocation);

        convNet.forEach_maxPool(exeSpace);

        long end = System.currentTimeMillis();
        if (logEnable)
            logWriter(logFolder, logFile, logMsg + "\t" + String.valueOf(end - start) + "\n");

    }

    private void paraConvRelu(RenderScript rs, ScriptC_convNet convNet, int wghtSzF4, int biasSz,
                              float[] wght, float[] bias, int exeSpaceSz, int K,
                              int Wout, int Hout, int Win, int Hin, int M, int N, int S, int pad,
                              int N_new, int offset, Allocation inputAllocation,
                              Allocation outputAllocation, int convType, int parallelOFMs,
                              String logFolder, String logFile,
                              String logMsg, boolean logEnable) {
        //Intent createDatabase = new Intent("com.quicinc.trepn.start_profiling");
        //createDatabase.putExtra("com.quicinc.trepn.database_file", dataBaseName);
        //context.sendBroadcast(createDatabase);

        //Intent startProfiling = new Intent("com.quicinc.trepn.start_profiling");
        //context.sendBroadcast(startProfiling);

        long start = System.currentTimeMillis();

        Type.Builder weightType = new Type.Builder(rs, Element.F32_4(rs)).setX(wghtSzF4);
        Allocation weightAllocation = Allocation.createTyped(rs, weightType.create());
        weightAllocation.copyFromUnchecked(wght);

        Allocation biasAllocation = Allocation.createSized(rs, Element.F32(rs), biasSz);
        biasAllocation.copy1DRangeFrom(0, biasSz, bias);

        Allocation exeSpace = Allocation.createSized(rs, Element.F32(rs), exeSpaceSz);

        convNet.set_K(K);
        convNet.set_Wout(Wout);
        convNet.set_Hout(Hout);
        convNet.set_Hin(Hin);
        convNet.set_Win(Win);
        convNet.set_M(M);
        convNet.set_N(N);
        convNet.set_S(S);
        convNet.set_pad(pad);
        convNet.set_N_new(N_new);
        convNet.set_offset(offset);
        convNet.set_parallelOFMs(parallelOFMs);

        convNet.set_weight(weightAllocation);
        convNet.set_bias(biasAllocation);
        convNet.set_in(inputAllocation);

        convNet.set_output(outputAllocation);

        switch (convType) {
            case 1:
                convNet.forEach_conv_1(exeSpace);
                break;
            case 2:
                convNet.forEach_conv_2(exeSpace);
                break;
            case 4:
                convNet.forEach_conv_4(exeSpace);
                break;
            case 5:
                convNet.forEach_conv_5(exeSpace);
                break;
            case 6:
                convNet.forEach_conv_6(exeSpace);
                break;
            case 7:
                convNet.forEach_conv_7(exeSpace);
                break;
            case 8:
                convNet.forEach_conv_8(exeSpace);
                break;
            case 9:
                convNet.forEach_conv_9(exeSpace);
                break;
            case 10:
                convNet.forEach_conv_10(exeSpace);
                break;
            case 12:
                convNet.forEach_conv_12(exeSpace);
                break;
            case 13:
                convNet.forEach_conv_13(exeSpace);
                break;
            case 14:
                convNet.forEach_conv_14(exeSpace);
                break;
            case 16:
                convNet.forEach_conv_16(exeSpace);
                break;
            case 18:
                convNet.forEach_conv_18(exeSpace);
                break;
            case 20:
                convNet.forEach_conv_20(exeSpace);
                break;
            case 24:
                convNet.forEach_conv_24(exeSpace);
                break;
            case 26:
                convNet.forEach_conv_26(exeSpace);
                break;
            case 28:
                convNet.forEach_conv_28(exeSpace);
                break;
            case 32:
                convNet.forEach_conv_32(exeSpace);
                break;
            case 36:
                convNet.forEach_conv_36(exeSpace);
                break;
            case 40:
                convNet.forEach_conv_40(exeSpace);
                break;
            case 44:
                convNet.forEach_conv_44(exeSpace);
                break;
            case 48:
                convNet.forEach_conv_48(exeSpace);
                break;
            case 52:
                convNet.forEach_conv_52(exeSpace);
                break;
            case 56:
                convNet.forEach_conv_56(exeSpace);
                break;
            case 64:
                convNet.forEach_conv_64(exeSpace);
                break;
            case 72:
                convNet.forEach_conv_72(exeSpace);
                break;
            case 80:
                convNet.forEach_conv_80(exeSpace);
                break;
            case 96:
                convNet.forEach_conv_96(exeSpace);
                break;
            default:
                Log.e(tag, convType + " is an invalid size for convolution kernel.");
                break;
        }

        long end = System.currentTimeMillis();
        //Intent stopProfiling = new Intent("com.quicinc.trepn.stop_profiling");
        //context.sendBroadcast(stopProfiling);
        if (logEnable)
            logWriter(logFolder, logFile, logMsg + "\t" + String.valueOf(end - start) + "\n");
    }

    private void lrn(float[] in, float[] out, int Win, int Hin, int N, int localSize,
                     float alpha, float beta, int K, String logFolder,
                     String logFile, String logMsg, boolean logEnable) {
        long start = System.currentTimeMillis();
        double tmp;
        int w, h, n, Mstart, Mend, m;

        for (w = 0; w < Win; w++) {
            for (h = 0; h < Hin; h++) {
                for (n = 0; n < N; n++) {
                    tmp = Math.pow(in[w + Win * h + Win * Hin * n], 2);
                    Mstart = Math.max(0, n - localSize / 2);
                    Mend = Math.min(n + localSize / 2 + 1, N);
                    for (m = Mstart; m < Mend; m++) {
                        out[w + Win * h + Win * Hin * m] += tmp;
                    }
                }
            }
        }
        //Calculate the output.
        for (w = 0; w < Win; w++) {
            for (h = 0; h < Hin; h++) {
                for (n = 0; n < N; n++) {
                    out[w + Win * h + Win * Hin * n] =
                            in[w + Win * h + Win * Hin * n] / (float) Math.pow(K +
                                    alpha / (float) localSize * out[w + Win * h + Win * Hin * n], beta);
                }
            }
        }
        long end = System.currentTimeMillis();
        if (logEnable)
            logWriter(logFolder, logFile, logMsg + "\t" + String.valueOf(end - start) + "\n");

    }

    private void convRelu(float[] in, float[] out, float[] weight,
                          float[] bias, int Win, int Hin, int N,
                          int M, int K, int S, int pad, int group, String logFolder,
                          String logFile, String logMsg, boolean logEnable) {

        long start = System.currentTimeMillis();

        int Wout, Hout, w, h, m, n, i, j;
        Wout = (Win + 2 * pad - K) / S + 1;
        Hout = (Hin + 2 * pad - K) / S + 1;
        //Convolve the input feature maps with the kernels.
        //The access to the input is shifted by the number of padding pixels.
        //Before every MAC, check if this is in the zero-padded area.
        //ToDo: Get rid of this initialization
        for (i = 0; i < Wout; i++) {
            for (j = 0; j < Hout; j++) {
                for (int k = 0; k < M; k++) {
                    out[i * Hout * M + j * M + k] = 0;
                }
            }
        }
        switch (group) {
            //The output depends on all input feature maps.
            case 1:
                for (w = 0; w < Wout; w++) {
                    for (h = 0; h < Hout; h++) {
                        for (m = 0; m < M; m++) {
                            for (n = 0; n < N; n++) {
                                for (i = 0; i < K; i++) {
                                    for (j = 0; j < K; j++) {
                                        if (w * S + i - pad < 0 || w * S + i - pad >= Win ||
                                                h * S + j - pad < 0 || h * S + j - pad >= Hin)
                                            continue;
                                        out[(w) + (Wout * h) + (Wout * Hout * m)] +=
                                                in[(w * S + i - pad) + Win * (h * S + j - pad) + (Win * Hin * n)] *
                                                        weight[(i) + (K * j) + (K * K * n) + (K * K * N * m)];
                                    }
                                }
                            }
                        }
                    }
                }
                break;
            //The input and output are split into 2 groups.
            case 2:
                //The first half of input feature maps depends on the first half of output feature maps.
                for (w = 0; w < Wout; w++) {
                    for (h = 0; h < Hout; h++) {
                        for (m = 0; m < M / 2; m++) {
                            for (n = 0; n < N / 2; n++) {
                                for (i = 0; i < K; i++) {
                                    for (j = 0; j < K; j++) {
                                        if (w * S + i - pad < 0 || w * S + i - pad >= Win ||
                                                h * S + j - pad < 0 || h * S + j - pad >= Hin)
                                            continue;
                                        out[(w) + (Wout * h) + (Wout * Hout * m)] +=
                                                in[(w * S + i - pad) + Win * (h * S + j - pad) + (Win * Hin * n)] *
                                                        weight[(i) + (K * j) + (K * K * n) + (K * K * N / 2 * m)];
                                    }
                                }
                            }
                        }
                    }
                }
                //The second half of input feature maps depends on the second half of output feature maps.
                for (w = 0; w < Wout; w++) {
                    for (h = 0; h < Hout; h++) {
                        for (m = M / 2; m < M; m++) {
                            for (n = N / 2; n < N; n++) {
                                for (i = 0; i < K; i++) {
                                    for (j = 0; j < K; j++) {
                                        if (w * S + i - pad < 0 || w * S + i - pad >= Win ||
                                                h * S + j - pad < 0 || h * S + j - pad >= Hin)
                                            continue;
                                        out[(w) + (Wout * h) + (Wout * Hout * m)] +=
                                                in[(w * S + i - pad) + Win * (h * S + j - pad) + (Win * Hin * n)] *
                                                        weight[(i) + (K * j) + (K * K * (n - N / 2)) + (K * K * N / 2 * m)];
                                    }
                                }
                            }
                        }
                    }
                }
                break;
            default:
                Log.e(tag, "ERROR: Convolution group must be 1 or 2.\n");
                break;
        }
        //Add the bias per output feature map.
        for (w = 0; w < Wout; w++) {
            for (h = 0; h < Hout; h++) {
                for (m = 0; m < M; m++) {
                    out[(w) + (Wout * h) + (Wout * Hout * m)] += bias[m];
                }
            }
        }

        for (w = 0; w < Wout; w++) {
            for (h = 0; h < Hout; h++) {
                for (n = 0; n < M; n++) {
                    out[w + Wout * h + Wout * Hout * n] = Math.max(0, out[w + Wout * h + Wout * Hout * n]);
                }
            }
        }
        long end = System.currentTimeMillis();
        if (logEnable)
            logWriter(logFolder, logFile, logMsg + "\t" + String.valueOf(end - start) + "\n");
    }

    private void maxpool(float[] in, float[] out, int Win, int Hin,
                         int N, int kernelSize, int S, String logFolder,
                         String logFile, String logMsg, boolean logEnable) {
        long start = System.currentTimeMillis();

        int w, ww, h, hh, n, Wout, Hout, Wstart, Wend, Hstart, Hend;
        float max;
        Wout = (Win - kernelSize) / S + 1;
        Hout = (Hin - kernelSize) / S + 1;
        for (w = 0; w < Wout; w++) {
            for (h = 0; h < Hout; h++) {
                for (n = 0; n < N; n++) {
                    Wstart = w * S;
                    Hstart = h * S;
                    Wend = Wstart + kernelSize;
                    Hend = Hstart + kernelSize;
                    max = -Float.MAX_VALUE;
                    for (ww = Wstart; ww < Wend; ww++) {
                        for (hh = Hstart; hh < Hend; hh++) {
                            max = Math.max(max, in[ww + Win * hh + Win * Hin * n]);
                        }
                    }
                    out[w + Wout * h + Wout * Hout * n] = max;
                }
            }
        }

        long end = System.currentTimeMillis();
        if (logEnable)
            logWriter(logFolder, logFile, logMsg + "\t" + String.valueOf(end - start) + "\n");
    }

    private void maxpoolOUTPUT(float[] in, float[] out, int Win, int Hin,
                               int N, int kernelSize, int S, int Wout, int Hout, String logFolder,
                               String logFile, String logMsg, boolean logEnable) {
        long start = System.currentTimeMillis();

        int w, ww, h, hh, n, Wstart, Wend, Hstart, Hend;
        float max;
        //Wout = (Win - kernelSize) / S + 1;
        //Hout = (Hin - kernelSize) / S + 1;
        for (w = 0; w < Wout; w++) {
            for (h = 0; h < Hout; h++) {
                for (n = 0; n < N; n++) {
                    Wstart = w * S;
                    Hstart = h * S;
                    Wend = Math.min(Wstart + kernelSize, Win);
                    Hend = Math.min(Hstart + kernelSize, Hin);
                    max = -Float.MAX_VALUE;
                    for (ww = Wstart; ww < Wend; ww++) {
                        for (hh = Hstart; hh < Hend; hh++) {
                            max = Math.max(max, in[ww + Win * hh + Win * Hin * n]);
                        }
                    }
                    out[w + Wout * h + Wout * Hout * n] = max;
                }
            }
        }

        long end = System.currentTimeMillis();
        if (logEnable)
            logWriter(logFolder, logFile, logMsg + "\t" + String.valueOf(end - start) + "\n");
    }

    private void maxpoolNew(float[] in, float[] out, int Win, int Hin,
                            int N, int kernelSize, int S, int Wout, int Hout, String logFolder,
                            String logFile, String logMsg, boolean logEnable) {
        long start = System.currentTimeMillis();
        int w0 = 0;
        int h0 = 0;
        int wEnd = Wout;
        int hEnd = Hout;
        // Implicit Padding
        if (S == 1) {
            w0 = -1;
            h0 = -1;
            wEnd = Wout + 1;
            hEnd = Hout + 1;
        }
        int w, ww, h, hh, n, Wstart, Wend, Hstart, Hend;
        float max;
        //for (w = 0; w<Wout; w++){
        for (w = -1; w < Wout - 1; w++) {
            //for (h = 0; h<Hout; h++){
            for (h = -1; h < Hout - 1; h++) {
                for (n = 0; n < N; n++) {
                    Wstart = w * S;
                    Hstart = h * S;
                    Wend = Math.min(Wstart + kernelSize, Win + 1);
                    Hend = Math.min(Hstart + kernelSize, Hin + 1);
                    max = -Float.MAX_VALUE;
                    for (ww = Wstart; ww < Wend; ww++) {
                        for (hh = Hstart; hh < Hend; hh++) {
                            if (ww < 0 || ww >= Win || hh < 0 || hh >= Hin) continue;
                            max = Math.max(max, in[ww + Win * hh + Win * Hin * n]);
                        }
                    }
                    out[w + 1 + Wout * (h + 1) + Wout * Hout * n] = max;
                }
            }
        }

        long end = System.currentTimeMillis();
        if (logEnable)
            logWriter(logFolder, logFile, logMsg + "\t" + String.valueOf(end - start) + "\n");
    }

    private void avgpool(float[] in, float[] out, int Win, int Hin,
                         int N, int kernelSize, int S, String logFolder,
                         String logFile, String logMsg, boolean logEnable) {
        long start = System.currentTimeMillis();

        int w, ww, h, hh, n, Wout, Hout, Wstart, Wend, Hstart, Hend;
        float sum = 0;
        Wout = (Win - kernelSize) / S + 1;
        Hout = (Hin - kernelSize) / S + 1;
        for (w = 0; w < Wout; w++) {
            for (h = 0; h < Hout; h++) {
                for (n = 0; n < N; n++) {
                    Wstart = w * S;
                    Hstart = h * S;
                    Wend = Wstart + kernelSize;
                    Hend = Hstart + kernelSize;
                    sum = 0;
                    for (ww = Wstart; ww < Wend; ww++) {
                        for (hh = Hstart; hh < Hend; hh++) {
                            sum += in[ww + Win * hh + Win * Hin * n];
                        }
                    }
                    out[w + Wout * h + Wout * Hout * n] = sum / (kernelSize * kernelSize);
                }
            }
        }

        long end = System.currentTimeMillis();
        if (logEnable)
            logWriter(logFolder, logFile, logMsg + "\t" + String.valueOf(end - start) + "\n");
    }

    private void softmax(float[] in, float[] out, int N, String logFolder, String logFile,
                         String logMsg, boolean logEnable) {
        long start = System.currentTimeMillis();

        float maxIn, sum;
        int n;
        maxIn = -Float.MAX_VALUE;
        sum = 0;
        //Calculate max input.
        for (n = 0; n < N; n++) {
            maxIn = Math.max(maxIn, in[n]);
        }
        //Calculate the exponential of each input.
        for (n = 0; n < N; n++) {
            out[n] = (float) Math.exp(in[n] - maxIn);
        }
        //Calculate the sum of all exponentials.
        for (n = 0; n < N; n++) {
            sum += out[n];
        }
        //Calculate the output.
        for (n = 0; n < N; n++) {
            out[n] = out[n] / sum;
        }

        long end = System.currentTimeMillis();
        if (logEnable)
            logWriter(logFolder, logFile, logMsg + "\t" + String.valueOf(end - start) + "\n");
    }

    private void dense(float[] in, float[] out, float[] weight,
                       float[] bias, int N, int M, String logFolder,
                       String logFile, String logMsg, boolean logEnable) {
        long start = System.currentTimeMillis();
        int m, n;
        //Fully connected (matrix-vector multiplication).
        for (m = 0; m < M; m++) {
            for (n = 0; n < N; n++) {
                out[m] += weight[n + N * m] * in[n];
            }
        }
        //Add bias.
        for (m = 0; m < M; m++) {
            out[m] += bias[m];
        }
        long end = System.currentTimeMillis();
        if (logEnable)
            logWriter(logFolder, logFile, logMsg + "\t" + String.valueOf(end - start) + "\n");
    }

    //UTILITIES
    private void loader(int size, String address, float[] array, Resources resources, String packageName) {
        try {
            /* To load weights from external storage use this address*/
            address = Environment.getExternalStorageDirectory().getPath() + address;

            String[] string = address.split("/");
            String weightFileName = string[7].substring(0,string[7].length()-4);

            InputStream inputStream = resources.openRawResource(
                    resources.getIdentifier(weightFileName, "raw", packageName));
            byte[] bytes = new byte[4*size];
            inputStream.read(bytes);

            //convert byte[] to float[]
            for (int i=0;i<size;i++){
                int asInt = (bytes[i*4+0] & 0xFF) | ((bytes[i*4+1] & 0xFF) << 8) | ((bytes[i*4+2] & 0xFF) << 16) | ((bytes[i*4+3] & 0xFF) << 24);
                array[i] = Float.intBitsToFloat(asInt);
            }

            /*
            RandomAccessFile file = new RandomAccessFile(address, "rw");
            FileChannel channel = file.getChannel();
            ByteBuffer buf = ByteBuffer.allocate(4 * size);
            buf.clear();
            channel.read(buf);
            buf.rewind();
            buf.order(ByteOrder.LITTLE_ENDIAN);
            buf.asFloatBuffer().get(array);
            channel.close();
            file.close(); */

            Log.i("WEIGHTS for ", weightFileName + "Layer loaded");
        } catch (IOException e) {
            Log.e(tag, e.getMessage());
        }
    }

    private void IntegerLoader(int size, String address, int[] array) {
        try {
            address = Environment.getExternalStorageDirectory().getPath() + address;
            RandomAccessFile file = new RandomAccessFile(address, "rw");
            FileChannel channel = file.getChannel();
            ByteBuffer buf = ByteBuffer.allocate(4 * size);
            buf.clear();
            channel.read(buf);
            buf.rewind();
            buf.order(ByteOrder.LITTLE_ENDIAN);
            buf.asIntBuffer().get(array);
            channel.close();
            file.close();
        } catch (IOException e) {
            Log.e(tag, e.getMessage());
        }
    }

    private void logWriter(String dirs, String fileName, String msg) {
        File logDirectory = new File(Environment.getExternalStorageDirectory().getPath() + "/" + dirs);
        logDirectory.mkdirs();
        try {
            FileWriter log = new FileWriter(logDirectory + "/" + fileName, true);
            log.write(msg);
            log.close();
        } catch (IOException e) {
            Log.e(tag, "Error: Cannot write tag file: " + "/" + fileName + "\n" + e.getMessage());
        }
    }

    private int findConvType(int[] typeSet, int genericConvType) {
        int convType = typeSet[0];
        uglyLoop = false;
        for (int cnt : typeSet) {
            if (cnt == genericConvType) {
                convType = genericConvType;
                uglyLoop = true;
                return convType;
            }
        }

        return convType;
    }

    private void binaryDumper(String dirs, String fileName, float[] array) {
        try {
            File dumpDirectory = new File(Environment.getExternalStorageDirectory().getPath() + "/" + dirs);
            dumpDirectory.mkdirs();

            RandomAccessFile file = new RandomAccessFile(dumpDirectory + "/" + fileName, "rw");
            FileChannel channel = file.getChannel();
            //ByteBuffer buf = ByteBuffer.allocate(4 * size);
            ByteBuffer buf = ByteBuffer.allocate(4 * array.length);
            buf.clear();

            //buf.order(ByteOrder.LITTLE_ENDIAN);
            buf.asFloatBuffer().put(array);
            channel.write(buf);
            buf.rewind();
            channel.close();
            file.close();
        } catch (IOException e) {
            Log.e(tag, e.getMessage());
        }
    }

    private void paraReshape(RenderScript rs, ScriptC_convNet convNet, Allocation outputAllocation) {
        Allocation imgAllocation = Allocation.createSized(rs, Element.F32(rs), 227 * 227 * 3);
        imgAllocation.copyFrom(img);

        convNet.set_Hout(227);
        convNet.set_Wout(227);

        Allocation exeSpace = Allocation.createSized(rs, Element.F32(rs), 227 * 227);

        convNet.set_in(imgAllocation);
        convNet.set_output(outputAllocation);
        convNet.forEach_reshape(exeSpace);
    }

    private void bitmapToFloatArray(Bitmap image, float[] array){
        float[][][] inputSingle = new float[3][227][227]; //[channels][image height][image width]
        int z = 0; //index for the meanImg array
        for (int j = 0; j < 227; ++j)
            for (int k = 0; k < 227; ++k) {
                int color = image.getPixel(j, k);
                inputSingle[0][k][j] = (float) (blue(color)) - (float) 104.0079317889;
                inputSingle[1][k][j] = (float) (green(color)) - (float) 116.66876761696767;
                inputSingle[2][k][j] = (float) (red(color)) - (float) 122.6789143406786;

                //BGR into the meanImg float array
                array[z]=inputSingle[0][k][j];
                array[z+227*227]=inputSingle[1][k][j];
                array[z+2*227*227]=inputSingle[2][k][j];
                z++;
            }
    }

    public void loadParameters(String logFolder, String logFile, String paramsDir, boolean logEnable, Resources resources, String packageName) {
        long start = System.currentTimeMillis();

        int sz; //size for the allocation of the float arrays

        sz = 227 * 227 * 3;
        meanImg = new float[sz];
        //Bitmap meanBitmap = BitmapFactory.decodeFile(Environment.getExternalStorageDirectory().getPath()+paramsDir + "/mean_image.jpg");
        Bitmap meanBitmap = BitmapFactory.decodeResource(resources,R.raw.mean_image);
        meanBitmap = Bitmap.createScaledBitmap(meanBitmap, 227, 227, true);    //make the foto squared
        bitmapToFloatArray(meanBitmap,meanImg);

        sz = 7 * 7 * 4 * 64;
        conv1_w = new float[sz];
        loader(sz, paramsDir + "/conv1_7x7_s2_w.bin", conv1_w, resources, packageName);

        sz = 64;
        conv1_b = new float[sz];
        loader(sz, paramsDir + "/conv1_7x7_s2_b.bin", conv1_b, resources, packageName);

        sz = 64 * 64; // 1 * 1 * 64 * 64
        conv2_3x3_reduce_w = new float[sz];
        loader(sz, paramsDir + "/conv2_3x3_reduce_w.bin", conv2_3x3_reduce_w, resources, packageName);

        sz = 64;
        conv2_3x3_reduce_b = new float[sz];
        loader(sz, paramsDir + "/conv2_3x3_reduce_b.bin", conv2_3x3_reduce_b, resources, packageName);

        sz = 3 * 3 * 64 * 192;
        conv2_3x3_w = new float[sz];
        loader(sz, paramsDir + "/conv2_3x3_w.bin", conv2_3x3_w, resources, packageName);

        sz = 192;
        conv2_3x3_b = new float[sz];
        loader(sz, paramsDir + "/conv2_3x3_b.bin", conv2_3x3_b, resources, packageName);

        sz = 3 * 3 * 192 * 64;
        inception_3a_1x1_w = new float[sz];
        loader(sz, paramsDir + "/inception_3a_1x1_w.bin", inception_3a_1x1_w, resources, packageName);

        sz = 64;
        inception_3a_1x1_b = new float[sz];
        loader(sz, paramsDir + "/inception_3a_1x1_b.bin", inception_3a_1x1_b, resources, packageName);

        sz = 192 * 96; //1 * 1 * 192 * 96
        inception_3a_3x3_reduce_w = new float[sz];
        loader(sz, paramsDir + "/inception_3a_3x3_reduce_w.bin", inception_3a_3x3_reduce_w, resources, packageName);

        sz = 96;
        inception_3a_3x3_reduce_b = new float[sz];
        loader(sz, paramsDir + "/inception_3a_3x3_reduce_b.bin", inception_3a_3x3_reduce_b, resources, packageName);

        sz = 192 * 16;//1 * 1 * 192 * 16
        inception_3a_5x5_reduce_w = new float[sz];
        loader(sz, paramsDir + "/inception_3a_5x5_reduce_w.bin", inception_3a_5x5_reduce_w, resources, packageName);

        sz = 16;
        inception_3a_5x5_reduce_b = new float[sz];
        loader(sz, paramsDir + "/inception_3a_5x5_reduce_b.bin", inception_3a_5x5_reduce_b, resources, packageName);

        sz = 3 * 3 * 96 * 128;
        inception_3a_3x3_w = new float[sz];
        loader(sz, paramsDir + "/inception_3a_3x3_w.bin", inception_3a_3x3_w, resources, packageName);

        sz = 128;
        inception_3a_3x3_b = new float[sz];
        loader(sz, paramsDir + "/inception_3a_3x3_b.bin", inception_3a_3x3_b, resources, packageName);

        sz = 5 * 5 * 16 * 32;
        inception_3a_5x5_w = new float[sz];
        loader(sz, paramsDir + "/inception_3a_5x5_w.bin", inception_3a_5x5_w, resources, packageName);

        sz = 32;
        inception_3a_5x5_b = new float[sz];
        loader(sz, paramsDir + "/inception_3a_5x5_b.bin", inception_3a_5x5_b, resources, packageName);

        sz = 192 * 32;//1 * 1 * 192 * 32;
        inception_3a_pool_proj_w = new float[sz];
        loader(sz, paramsDir + "/inception_3a_pool_proj_w.bin", inception_3a_pool_proj_w, resources, packageName);

        sz = 32;
        inception_3a_pool_proj_b = new float[sz];
        loader(sz, paramsDir + "/inception_3a_pool_proj_b.bin", inception_3a_pool_proj_b, resources, packageName);

        sz = 256 * 128; //1 * 1 * 256 * 128
        inception_3b_1x1_w = new float[sz];
        loader(sz, paramsDir + "/inception_3b_1x1_w.bin", inception_3b_1x1_w, resources, packageName);

        sz = 128;
        inception_3b_1x1_b = new float[sz];
        loader(sz, paramsDir + "/inception_3b_1x1_b.bin", inception_3b_1x1_b, resources, packageName);

        sz = 256 * 128;//1 * 1 * 256 * 128
        inception_3b_3x3_reduce_w = new float[sz];
        loader(sz, paramsDir + "/inception_3b_3x3_reduce_w.bin", inception_3b_3x3_reduce_w, resources, packageName);

        sz = 128;
        inception_3b_3x3_reduce_b = new float[sz];
        loader(sz, paramsDir + "/inception_3b_3x3_reduce_b.bin", inception_3b_3x3_reduce_b, resources, packageName);

        sz = 256 * 32; //1 * 1 * 256 * 32
        inception_3b_5x5_reduce_w = new float[sz];
        loader(sz, paramsDir + "/inception_3b_5x5_reduce_w.bin", inception_3b_5x5_reduce_w, resources, packageName);

        sz = 32;
        inception_3b_5x5_reduce_b = new float[sz];
        loader(sz, paramsDir + "/inception_3b_5x5_reduce_b.bin", inception_3b_5x5_reduce_b, resources, packageName);

        sz = 3 * 3 * 128 * 192;
        inception_3b_3x3_w = new float[sz];
        loader(sz, paramsDir + "/inception_3b_3x3_w.bin", inception_3b_3x3_w, resources, packageName);

        sz = 192;
        inception_3b_3x3_b = new float[sz];
        loader(sz, paramsDir + "/inception_3b_3x3_b.bin", inception_3b_3x3_b, resources, packageName);

        sz = 5 * 5 * 32 * 96;
        inception_3b_5x5_w = new float[sz];
        loader(sz, paramsDir + "/inception_3b_5x5_w.bin", inception_3b_5x5_w, resources, packageName);

        sz = 96;
        inception_3b_5x5_b = new float[sz];
        loader(sz, paramsDir + "/inception_3b_5x5_b.bin", inception_3b_5x5_b, resources, packageName);

        sz = 256 * 64; //1 * 1 * 256 * 64
        inception_3b_pool_proj_w = new float[sz];
        loader(sz, paramsDir + "/inception_3b_pool_proj_w.bin", inception_3b_pool_proj_w, resources, packageName);

        sz = 64;
        inception_3b_pool_proj_b = new float[sz];
        loader(sz, paramsDir + "/inception_3b_pool_proj_b.bin", inception_3b_pool_proj_b, resources, packageName);

        sz = 480 * 192; //1 * 1 * 480 * 192
        inception_4a_1x1_w = new float[sz];
        loader(sz, paramsDir + "/inception_4a_1x1_w.bin", inception_4a_1x1_w, resources, packageName);

        sz = 192;
        inception_4a_1x1_b = new float[sz];
        loader(sz, paramsDir + "/inception_4a_1x1_b.bin", inception_4a_1x1_b, resources, packageName);

        sz = 480 * 96; //1 * 1 * 480 * 96
        inception_4a_3x3_reduce_w = new float[sz];
        loader(sz, paramsDir + "/inception_4a_3x3_reduce_w.bin", inception_4a_3x3_reduce_w, resources, packageName);

        sz = 96;
        inception_4a_3x3_reduce_b = new float[sz];
        loader(sz, paramsDir + "/inception_4a_3x3_reduce_b.bin", inception_4a_3x3_reduce_b, resources, packageName);

        sz = 480 * 16; //1 * 1 * 48 * 192
        inception_4a_5x5_reduce_w = new float[sz];
        loader(sz, paramsDir + "/inception_4a_5x5_reduce_w.bin", inception_4a_5x5_reduce_w, resources, packageName);

        sz = 16;
        inception_4a_5x5_reduce_b = new float[sz];
        loader(sz, paramsDir + "/inception_4a_5x5_reduce_b.bin", inception_4a_5x5_reduce_b, resources, packageName);

        sz = 3 * 3 * 96 * 208;
        inception_4a_3x3_w = new float[sz];
        loader(sz, paramsDir + "/inception_4a_3x3_w.bin", inception_4a_3x3_w, resources, packageName);

        sz = 208;
        inception_4a_3x3_b = new float[sz];
        loader(sz, paramsDir + "/inception_4a_3x3_b.bin", inception_4a_3x3_b, resources, packageName);

        sz = 5 * 5 * 16 * 48;//5 * 5 * 16 * 48
        inception_4a_5x5_w = new float[sz];
        loader(sz, paramsDir + "/inception_4a_5x5_w.bin", inception_4a_5x5_w, resources, packageName);

        sz = 48;
        inception_4a_5x5_b = new float[sz];
        loader(sz, paramsDir + "/inception_4a_5x5_b.bin", inception_4a_5x5_b, resources, packageName);

        sz = 480 * 64; //1 * 1 * 480 * 64
        inception_4a_pool_proj_w = new float[sz];
        loader(sz, paramsDir + "/inception_4a_pool_proj_w.bin", inception_4a_pool_proj_w, resources, packageName);

        sz = 64;
        inception_4a_pool_proj_b = new float[sz];
        loader(sz, paramsDir + "/inception_4a_pool_proj_b.bin", inception_4a_pool_proj_b, resources, packageName);

        sz = 512 * 160; //1 * 1 * 512 * 160
        inception_4b_1x1_w = new float[sz];
        loader(sz, paramsDir + "/inception_4b_1x1_w.bin", inception_4b_1x1_w, resources, packageName);

        sz = 160;
        inception_4b_1x1_b = new float[sz];
        loader(sz, paramsDir + "/inception_4b_1x1_b.bin", inception_4b_1x1_b, resources, packageName);

        sz = 512 * 112; //1 * 1 * 512 * 112
        inception_4b_3x3_reduce_w = new float[sz];
        loader(sz, paramsDir + "/inception_4b_3x3_reduce_w.bin", inception_4b_3x3_reduce_w, resources, packageName);

        sz = 112;
        inception_4b_3x3_reduce_b = new float[sz];
        loader(sz, paramsDir + "/inception_4b_3x3_reduce_b.bin", inception_4b_3x3_reduce_b, resources, packageName);

        sz = 512 * 24;//1 * 1 * 512 * 24
        inception_4b_5x5_reduce_w = new float[sz];
        loader(sz, paramsDir + "/inception_4b_5x5_reduce_w.bin", inception_4b_5x5_reduce_w, resources, packageName);

        sz = 24;
        inception_4b_5x5_reduce_b = new float[sz];
        loader(sz, paramsDir + "/inception_4b_5x5_reduce_b.bin", inception_4b_5x5_reduce_b, resources, packageName);

        sz = 3 * 3 * 112 * 224;
        inception_4b_3x3_w = new float[sz];
        loader(sz, paramsDir + "/inception_4b_3x3_w.bin", inception_4b_3x3_w, resources, packageName);

        sz = 224;
        inception_4b_3x3_b = new float[sz];
        loader(sz, paramsDir + "/inception_4b_3x3_b.bin", inception_4b_3x3_b, resources, packageName);

        sz = 5 * 5 * 24 * 64;
        inception_4b_5x5_w = new float[sz];
        loader(sz, paramsDir + "/inception_4b_5x5_w.bin", inception_4b_5x5_w, resources, packageName);

        sz = 64;
        inception_4b_5x5_b = new float[sz];
        loader(sz, paramsDir + "/inception_4b_5x5_b.bin", inception_4b_5x5_b, resources, packageName);

        sz = 512 * 64; // 1 * 1 * 512 * 64
        inception_4b_pool_proj_w = new float[sz];
        loader(sz, paramsDir + "/inception_4b_pool_proj_w.bin", inception_4b_pool_proj_w, resources, packageName);

        sz = 64;
        inception_4b_pool_proj_b = new float[sz];
        loader(sz, paramsDir + "/inception_4b_pool_proj_b.bin", inception_4b_pool_proj_b, resources, packageName);

        sz = 512 * 128; //1 * 1 * 512 * 128
        inception_4c_1x1_w = new float[sz];
        loader(sz, paramsDir + "/inception_4c_1x1_w.bin", inception_4c_1x1_w, resources, packageName);

        sz = 128;
        inception_4c_1x1_b = new float[sz];
        loader(sz, paramsDir + "/inception_4c_1x1_b.bin", inception_4c_1x1_b, resources, packageName);

        sz = 512 * 128; // 1 * 1 * 512 * 128
        inception_4c_3x3_reduce_w = new float[sz];
        loader(sz, paramsDir + "/inception_4c_3x3_reduce_w.bin", inception_4c_3x3_reduce_w, resources, packageName);

        sz = 128;
        inception_4c_3x3_reduce_b = new float[sz];
        loader(sz, paramsDir + "/inception_4c_3x3_reduce_b.bin", inception_4c_3x3_reduce_b, resources, packageName);

        sz = 512 * 24;//1 * 1 * 512 * 24
        inception_4c_5x5_reduce_w = new float[sz];
        loader(sz, paramsDir + "/inception_4c_5x5_reduce_w.bin", inception_4c_5x5_reduce_w, resources, packageName);

        sz = 24;
        inception_4c_5x5_reduce_b = new float[sz];
        loader(sz, paramsDir + "/inception_4c_5x5_reduce_b.bin", inception_4c_5x5_reduce_b, resources, packageName);

        sz = 3 * 3 * 128 * 256;
        inception_4c_3x3_w = new float[sz];
        loader(sz, paramsDir + "/inception_4c_3x3_w.bin", inception_4c_3x3_w, resources, packageName);

        sz = 256;
        inception_4c_3x3_b = new float[sz];
        loader(sz, paramsDir + "/inception_4c_3x3_b.bin", inception_4c_3x3_b, resources, packageName);

        sz = 5 * 5 * 24 * 64;
        inception_4c_5x5_w = new float[sz];
        loader(sz, paramsDir + "/inception_4c_5x5_w.bin", inception_4c_5x5_w, resources, packageName);

        sz = 64;
        inception_4c_5x5_b = new float[sz];
        loader(sz, paramsDir + "/inception_4c_5x5_b.bin", inception_4c_5x5_b, resources, packageName);

        sz = 512 * 64; // 1 * 1 * 512 * 64
        inception_4c_pool_proj_w = new float[sz];
        loader(sz, paramsDir + "/inception_4c_pool_proj_w.bin", inception_4c_pool_proj_w, resources, packageName);

        sz = 64;
        inception_4c_pool_proj_b = new float[sz];
        loader(sz, paramsDir + "/inception_4c_pool_proj_b.bin", inception_4c_pool_proj_b, resources, packageName);

        sz = 512 * 112; //1 * 1 * 512 * 112
        inception_4d_1x1_w = new float[sz];
        loader(sz, paramsDir + "/inception_4d_1x1_w.bin", inception_4d_1x1_w, resources, packageName);

        sz = 112;
        inception_4d_1x1_b = new float[sz];
        loader(sz, paramsDir + "/inception_4d_1x1_b.bin", inception_4d_1x1_b, resources, packageName);

        sz = 512 * 144; // 1 * 1 * 512 * 144
        inception_4d_3x3_reduce_w = new float[sz];
        loader(sz, paramsDir + "/inception_4d_3x3_reduce_w.bin", inception_4d_3x3_reduce_w, resources, packageName);

        sz = 144;
        inception_4d_3x3_reduce_b = new float[sz];
        loader(sz, paramsDir + "/inception_4d_3x3_reduce_b.bin", inception_4d_3x3_reduce_b, resources, packageName);

        sz = 512 * 32;//1 * 1 * 512 * 32
        inception_4d_5x5_reduce_w = new float[sz];
        loader(sz, paramsDir + "/inception_4d_5x5_reduce_w.bin", inception_4d_5x5_reduce_w, resources, packageName);

        sz = 32;
        inception_4d_5x5_reduce_b = new float[sz];
        loader(sz, paramsDir + "/inception_4d_5x5_reduce_b.bin", inception_4d_5x5_reduce_b, resources, packageName);

        sz = 3 * 3 * 144 * 288;
        inception_4d_3x3_w = new float[sz];
        loader(sz, paramsDir + "/inception_4d_3x3_w.bin", inception_4d_3x3_w, resources, packageName);

        sz = 288;
        inception_4d_3x3_b = new float[sz];
        loader(sz, paramsDir + "/inception_4d_3x3_b.bin", inception_4d_3x3_b, resources, packageName);

        sz = 5 * 5 * 32 * 64;
        inception_4d_5x5_w = new float[sz];
        loader(sz, paramsDir + "/inception_4d_5x5_w.bin", inception_4d_5x5_w, resources, packageName);

        sz = 64;
        inception_4d_5x5_b = new float[sz];
        loader(sz, paramsDir + "/inception_4d_5x5_b.bin", inception_4d_5x5_b, resources, packageName);

        sz = 512 * 64; // 1 * 1 * 512 * 64
        inception_4d_pool_proj_w = new float[sz];
        loader(sz, paramsDir + "/inception_4d_pool_proj_w.bin", inception_4d_pool_proj_w, resources, packageName);

        sz = 64;
        inception_4d_pool_proj_b = new float[sz];
        loader(sz, paramsDir + "/inception_4d_pool_proj_b.bin", inception_4d_pool_proj_b, resources, packageName);

        sz = 528 * 256; //1 * 1 * 528 * 256
        inception_4e_1x1_w = new float[sz];
        loader(sz, paramsDir + "/inception_4e_1x1_w.bin", inception_4e_1x1_w, resources, packageName);

        sz = 256;
        inception_4e_1x1_b = new float[sz];
        loader(sz, paramsDir + "/inception_4e_1x1_b.bin", inception_4e_1x1_b, resources, packageName);

        sz = 528 * 160; // 1 * 1 * 528 * 160
        inception_4e_3x3_reduce_w = new float[sz];
        loader(sz, paramsDir + "/inception_4e_3x3_reduce_w.bin", inception_4e_3x3_reduce_w, resources, packageName);

        sz = 160;
        inception_4e_3x3_reduce_b = new float[sz];
        loader(sz, paramsDir + "/inception_4e_3x3_reduce_b.bin", inception_4e_3x3_reduce_b, resources, packageName);

        sz = 528 * 32;//1 * 1 * 528 * 32
        inception_4e_5x5_reduce_w = new float[sz];
        loader(sz, paramsDir + "/inception_4e_5x5_reduce_w.bin", inception_4e_5x5_reduce_w, resources, packageName);

        sz = 32;
        inception_4e_5x5_reduce_b = new float[sz];
        loader(sz, paramsDir + "/inception_4e_5x5_reduce_b.bin", inception_4e_5x5_reduce_b, resources, packageName);

        sz = 3 * 3 * 160 * 320;
        inception_4e_3x3_w = new float[sz];
        loader(sz, paramsDir + "/inception_4e_3x3_w.bin", inception_4e_3x3_w, resources, packageName);

        sz = 320;
        inception_4e_3x3_b = new float[sz];
        loader(sz, paramsDir + "/inception_4e_3x3_b.bin", inception_4e_3x3_b, resources, packageName);

        sz = 5 * 5 * 32 * 128;
        inception_4e_5x5_w = new float[sz];
        loader(sz, paramsDir + "/inception_4e_5x5_w.bin", inception_4e_5x5_w, resources, packageName);

        sz = 128;
        inception_4e_5x5_b = new float[sz];
        loader(sz, paramsDir + "/inception_4e_5x5_b.bin", inception_4e_5x5_b, resources, packageName);

        sz = 528 * 128; // 1 * 1 * 528 * 128
        inception_4e_pool_proj_w = new float[sz];
        loader(sz, paramsDir + "/inception_4e_pool_proj_w.bin", inception_4e_pool_proj_w, resources, packageName);

        sz = 128;
        inception_4e_pool_proj_b = new float[sz];
        loader(sz, paramsDir + "/inception_4e_pool_proj_b.bin", inception_4e_pool_proj_b, resources, packageName);

        sz = 832 * 256; //1 * 1 * 832 * 256
        inception_5a_1x1_w = new float[sz];
        loader(sz, paramsDir + "/inception_5a_1x1_w.bin", inception_5a_1x1_w, resources, packageName);

        sz = 256;
        inception_5a_1x1_b = new float[sz];
        loader(sz, paramsDir + "/inception_5a_1x1_b.bin", inception_5a_1x1_b, resources, packageName);

        sz = 832 * 160; // 1 * 1 * 832 * 160
        inception_5a_3x3_reduce_w = new float[sz];
        loader(sz, paramsDir + "/inception_5a_3x3_reduce_w.bin", inception_5a_3x3_reduce_w, resources, packageName);

        sz = 160;
        inception_5a_3x3_reduce_b = new float[sz];
        loader(sz, paramsDir + "/inception_5a_3x3_reduce_b.bin", inception_5a_3x3_reduce_b, resources, packageName);

        sz = 832 * 32;//1 * 1 * 832 * 32
        inception_5a_5x5_reduce_w = new float[sz];
        loader(sz, paramsDir + "/inception_5a_5x5_reduce_w.bin", inception_5a_5x5_reduce_w, resources, packageName);

        sz = 32;
        inception_5a_5x5_reduce_b = new float[sz];
        loader(sz, paramsDir + "/inception_5a_5x5_reduce_b.bin", inception_5a_5x5_reduce_b, resources, packageName);

        sz = 3 * 3 * 160 * 320;
        inception_5a_3x3_w = new float[sz];
        loader(sz, paramsDir + "/inception_5a_3x3_w.bin", inception_5a_3x3_w, resources, packageName);

        sz = 320;
        inception_5a_3x3_b = new float[sz];
        loader(sz, paramsDir + "/inception_5a_3x3_b.bin", inception_5a_3x3_b, resources, packageName);

        sz = 5 * 5 * 32 * 128;
        inception_5a_5x5_w = new float[sz];
        loader(sz, paramsDir + "/inception_5a_5x5_w.bin", inception_5a_5x5_w, resources, packageName);

        sz = 128;
        inception_5a_5x5_b = new float[sz];
        loader(sz, paramsDir + "/inception_5a_5x5_b.bin", inception_5a_5x5_b, resources, packageName);

        sz = 832 * 128; // 1 * 1 * 528 * 128
        inception_5a_pool_proj_w = new float[sz];
        loader(sz, paramsDir + "/inception_5a_pool_proj_w.bin", inception_5a_pool_proj_w, resources, packageName);

        sz = 128;
        inception_5a_pool_proj_b = new float[sz];
        loader(sz, paramsDir + "/inception_5a_pool_proj_b.bin", inception_5a_pool_proj_b, resources, packageName);

        sz = 832 * 384; //1 * 1 * 832 * 384
        inception_5b_1x1_w = new float[sz];
        loader(sz, paramsDir + "/inception_5b_1x1_w.bin", inception_5b_1x1_w, resources, packageName);

        sz = 384;
        inception_5b_1x1_b = new float[sz];
        loader(sz, paramsDir + "/inception_5b_1x1_b.bin", inception_5b_1x1_b, resources, packageName);

        sz = 832 * 192; // 1 * 1 * 832 * 192
        inception_5b_3x3_reduce_w = new float[sz];
        loader(sz, paramsDir + "/inception_5b_3x3_reduce_w.bin", inception_5b_3x3_reduce_w, resources, packageName);

        sz = 192;
        inception_5b_3x3_reduce_b = new float[sz];
        loader(sz, paramsDir + "/inception_5b_3x3_reduce_b.bin", inception_5b_3x3_reduce_b, resources, packageName);

        sz = 832 * 48;//1 * 1 * 832 * 48
        inception_5b_5x5_reduce_w = new float[sz];
        loader(sz, paramsDir + "/inception_5b_5x5_reduce_w.bin", inception_5b_5x5_reduce_w, resources, packageName);

        sz = 48;
        inception_5b_5x5_reduce_b = new float[sz];
        loader(sz, paramsDir + "/inception_5b_5x5_reduce_b.bin", inception_5b_5x5_reduce_b, resources, packageName);

        sz = 3 * 3 * 192 * 384;
        inception_5b_3x3_w = new float[sz];
        loader(sz, paramsDir + "/inception_5b_3x3_w.bin", inception_5b_3x3_w, resources, packageName);

        sz = 384;
        inception_5b_3x3_b = new float[sz];
        loader(sz, paramsDir + "/inception_5b_3x3_b.bin", inception_5b_3x3_b, resources, packageName);

        sz = 5 * 5 * 48 * 128;
        inception_5b_5x5_w = new float[sz];
        loader(sz, paramsDir + "/inception_5b_5x5_w.bin", inception_5b_5x5_w, resources, packageName);

        sz = 128;
        inception_5b_5x5_b = new float[sz];
        loader(sz, paramsDir + "/inception_5b_5x5_b.bin", inception_5b_5x5_b, resources, packageName);

        sz = 832 * 128; // 1 * 1 * 832 * 128
        inception_5b_pool_proj_w = new float[sz];
        loader(sz, paramsDir + "/inception_5b_pool_proj_w.bin", inception_5b_pool_proj_w, resources, packageName);

        sz = 128;
        inception_5b_pool_proj_b = new float[sz];
        loader(sz, paramsDir + "/inception_5b_pool_proj_b.bin", inception_5b_pool_proj_b, resources, packageName);

        //TODO change the 1000 to 101 to fit the dimension of my net
        //SIZES FOR 101 CLASSES
        sz = 1024*101;
        loss3_classifier_w = new float[sz];
        loader(sz, paramsDir + "/loss3_classifier_new_w.bin", loss3_classifier_w, resources, packageName);

        sz = 101;
        loss3_classifier_b = new float[sz];
        loader(sz, paramsDir + "/loss3_classifier_new_b.bin", loss3_classifier_b, resources, packageName);

        long end = System.currentTimeMillis();

        if (logEnable)
            logWriter(logFolder, logFile, "Loading Parameters from SD Card (ms):\t" + String.valueOf(end - start) + "\n");
    }

    private void normalToVectorized(RenderScript rs, ScriptC_convNet convNet, float[] in, Allocation out,
                                    int Width, int Height, int depth) {
        /* Converts a normal array to a vectorized format. The input is in CPU side and output in
        GPU side.
         */
        Allocation inAllocation = Allocation.createSized(rs, Element.F32(rs), Width * Height * depth);
        inAllocation.copyFrom(in);

        Allocation exeSpace = Allocation.createSized(rs, Element.F32(rs), Width * Height);
        convNet.set_Wout(Width);
        convNet.set_Hout(Height);
        convNet.set_N(depth);
        convNet.set_output(out);
        convNet.set_in(inAllocation);
        convNet.forEach_normalToVectorized(exeSpace);
    }

    private void vectorizedToNormal(Allocation in, float[] out, int Width, int Height, int depth) {
        /* Converts a vectorized data structure to normal. Can be used
        when it is desired to perform one layer on CPU. Input is an
         Allocation (RenderScript) and output is an array.
        */
        int size = Width * Height * depth;
        float[] aux = new float[size];
        in.copyTo(aux);
        int counter = 0;
        int i = 0;
        int j = 0;
        int base = 0;
        while (counter < size) {
            out[counter] = aux[base + 4 * i + j];
            counter++;
            if (i < (Width * Height - 1)) {
                i++;
            } else if (j < 4 - 1) {
                i = 0;
                j++;
            } else {
                i = 0;
                j = 0;
                base += 4 * Width * Height;
            }
        }
    }

}
