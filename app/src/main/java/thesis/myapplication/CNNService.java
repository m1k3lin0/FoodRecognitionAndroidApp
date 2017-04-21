package thesis.myapplication;

import android.app.IntentService;
import android.content.Intent;
import android.content.SharedPreferences;
import android.graphics.Bitmap;
import android.preference.PreferenceManager;
import android.support.v4.content.LocalBroadcastManager;
import android.util.Log;

import java.io.BufferedReader;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.regex.Pattern;

public class CNNService extends IntentService {

    //Utilities
    private static final String TAG = "CNNService"; //tag for debugging
    SharedPreferences sharedPref;

    static URL url;         //Server URL for CNN on server
    GoogleNet googleNet;    //local implementation of the CNN

    public CNNService(){
        super(CNNService.class.getName());

        //Initializations of the parameters
        googleNet = new GoogleNet();
        try {
            url = new URL(Utility.URL);
        } catch (MalformedURLException e) {
            e.printStackTrace();
        }
    }

    /*
    * Manage the request from the application by querying the requested CNN
    * */
    @Override
    protected void onHandleIntent(Intent intent) {
        Log.i(TAG, "Service Started!");
        Bitmap image = (Bitmap) intent.getParcelableExtra(Utility.BITMAP_EXTRA);
        sharedPref = PreferenceManager.getDefaultSharedPreferences(getApplicationContext());

        try {
            //1 = online mode, 0 = offline mode
            if (sharedPref.getInt(Utility.MODE,0)==0){
                localCNN(image);
            }
            else {
                serverCNN(image);
            }
        } catch (IOException e) {
            e.printStackTrace();
            //send "exception" to the main activity
            Intent localIntent =
                    new Intent(Utility.BROADCAST_ACTION);

            // broadcast the Intent to the receivers
            LocalBroadcastManager.getInstance(this).sendBroadcast(localIntent);
        }
    }

    /*
    * Query the local CNN
    * */
    public void localCNN(Bitmap image) throws IOException {
        googleNet.loadParameters("GoogleNet/Logs", "parallel.txt", "/GoogleNet/Parameters/Vectorized",false,getResources(),getApplicationContext().getPackageName());
        googleNet.loadImage(image);

        try {
            googleNet.parGoogleNet(getApplicationContext());
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        sendResponse(googleNet.getPredictions());
    }

    /*
    * Query the server CNN
    * */
    public void serverCNN(Bitmap image) throws IOException {

        //Save bitmap to file
        File dir = new File(Utility.TEMP_DIR_PATH);
        if (!dir.exists())
            dir.mkdirs();
        File file = new File(dir, Utility.TEMP_CROP_NAME);
        FileOutputStream fOut = null;
        try {
            fOut = new FileOutputStream(file);
            image.compress(Bitmap.CompressFormat.PNG, 0, fOut); //TODO compress lose quality, how to keep quality? maybe setting 0
            fOut.flush();
            fOut.close();
        } catch (Exception e) {
            e.printStackTrace();
        }

        HttpURLConnection urlConnection = (HttpURLConnection)url.openConnection();
        postImage(file, urlConnection);


        double[] result = readResponse(urlConnection);

        sendResponse(Utility.getTopKPredictions(result,5));
    }

    /*
    * Send the CNN response to the receiver
    * */
    public void sendResponse(RecognizedClass[] response){
        //put the result in the intent
        Intent localIntent =
                new Intent(Utility.BROADCAST_ACTION)
                    //put the status into the Intent
                    .putExtra(Utility.LABELS_EXTRA, Utility.getLabels(response))
                    .putExtra(Utility.SCORES_EXTRA, Utility.getScores(response));

        // broadcast the Intent to the receivers
        LocalBroadcastManager.getInstance(this).sendBroadcast(localIntent);
    }

    /*
    * Parse the json string and returns an array containing the scores for each class
    * */
    public double[] parseJson(String json) {
        String[] jsonSplit = json.split(Pattern.quote("}"));
        double[] result = new double[101];
        for (int i = 0; i<101; i++){
            String score = jsonSplit[i].substring(jsonSplit[i].lastIndexOf(":")+1);
            result[i] = Double.valueOf(score).doubleValue();
        }
        return result;
    }

    /*
    * Post image on server
    * */
    public void postImage(File file, HttpURLConnection urlConnection) throws IOException{
        //Prepare the post message and send it to the server
        String crlf = "\r\n";
        String twoHyphens = "--";
        String boundary = "1q2w3e4r";
        String attachmentName = "img";
        String attachmentFileName = "crop.png";

        //Connection parameters
        urlConnection.setUseCaches(false);
        urlConnection.setDoOutput(true);
        urlConnection.setDoInput(true);
        urlConnection.setRequestMethod("POST");
        urlConnection.setRequestProperty("Connection", "Keep-Alive");
        urlConnection.setRequestProperty("Cache-Control", "no-cache");
        urlConnection.setRequestProperty("Content-Type", "multipart/form-data;boundary=" + boundary);

        //Write the file on the stream
        DataOutputStream request = new DataOutputStream(urlConnection.getOutputStream());
        request.writeBytes(twoHyphens + boundary + crlf);
        request.writeBytes("Content-Disposition: form-data; name=\"" + attachmentName + "\";filename=\"" + attachmentFileName + "\"" + crlf);
        request.writeBytes("Content-Type: application/x-www-form-urlencoded" + crlf);
        request.writeBytes(crlf);

        // read file and write it into form
        int maxBufferSize = 1024000000; //TODO correct number?
        FileInputStream fileInputStream = new FileInputStream(file);
        int bytesAvailable = fileInputStream.available();
        int bufferSize = Math.min(bytesAvailable, maxBufferSize);
        byte[] buffer = new byte[bufferSize];
        int bytesRead = fileInputStream.read(buffer, 0, bufferSize);
        while (bytesRead > 0) {
            request.write(buffer, 0, bufferSize);
            bytesAvailable = fileInputStream.available();
            bufferSize = Math.min(bytesAvailable, maxBufferSize);
            bytesRead = fileInputStream.read(buffer, 0, bufferSize);
        }
        request.writeBytes(crlf);
        request.writeBytes(twoHyphens + boundary + twoHyphens + crlf);
        request.flush();
        request.close();
    }

    /*
    * Get response from the server
    * */
    public double[] readResponse(HttpURLConnection urlConnection) throws IOException{
        StringBuffer sb = new StringBuffer();
        BufferedReader br = new BufferedReader(new InputStreamReader( urlConnection.getInputStream(),"utf-8"));
        String line = null;
        while ((line = br.readLine()) != null) {
            sb.append(line + "\n");
        }
        br.close();
        String json = sb.toString();

        //Process the json
        double[] result = parseJson(json);
        return result;
    }
}
