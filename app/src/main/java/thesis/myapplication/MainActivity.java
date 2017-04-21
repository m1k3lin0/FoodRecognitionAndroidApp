package thesis.myapplication;

import android.Manifest;
import android.app.AlertDialog;
import android.app.ProgressDialog;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.IntentFilter;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.net.ConnectivityManager;
import android.net.NetworkInfo;
import android.net.Uri;
import android.os.Build;
import android.preference.PreferenceManager;
import android.provider.MediaStore;
import android.support.design.widget.FloatingActionButton;
import android.support.design.widget.Snackbar;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.LocalBroadcastManager;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;

import android.support.v4.app.Fragment;
import android.support.v4.app.FragmentManager;
import android.support.v4.app.FragmentPagerAdapter;
import android.support.v4.view.ViewPager;
import android.os.Bundle;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.view.ViewGroup;

import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.ListView;
import android.widget.TextView;
import android.widget.Toast;

import org.w3c.dom.Text;

import java.io.File;
import java.io.FileOutputStream;
import java.text.DecimalFormat;
import java.text.NumberFormat;

public class MainActivity extends AppCompatActivity {

    //ADAPTERS FOR THE SECTIONS
    private SectionsPagerAdapter mSectionsPagerAdapter; //the resultsAdapter that will provide fragments for each of the section. We use a fragmentpageradapter derivative, which will keep every loaded fragment in memory. If this becomes too memory intensive, it may be best to switch to fragmentstatepageadapter
    private ViewPager mViewPager;                       //the viewpager that will host the section contents

    //VIEWS OF THE ACTIVITIES
    static Button cameraBig;                            //big camera button
    //views of fragment_main
    static ImageView imageView;
    Bitmap imageToShow;
    static TextView infoText;
    static ListView listView;
    public static CustomList resultsAdapter;            //array adapter for the result of the classification
    static TextView noneOfTheAbove;
    //views of fragment_history
    static TextView accuracyTop1Text;
    static TextView accuracyTop5Text;
    static TextView avgScoreText;
    //views of fragment_dishes
    static ListView dishesListView;
    public static ArrayAdapter<String> adapterDishes;   //array adapter for the list of all dishes

    //UTILITY
    static RecognizedClass[] myItems;
    static double[] topScores;
    static String[] topLabels;
    static Context context;                                     //context of the application, used in static functions
    static boolean done = false;                                //true: the user has chosen if the food is correct or not (update statistics)
    static int chosen;                                          //index of the correct food in Top5 predictions
    static NumberFormat formatter = new DecimalFormat("#0.00"); //format the output


    //INTENTS
    CNNResultReceiver cnnResultReceiver;
    IntentFilter statusIntentFilter;

    //STATIC BUTTONS
    static FloatingActionButton camera;
    static FloatingActionButton clear;
    static FloatingActionButton fab;
    static ProgressDialog progressDialog;

    //SHARED PREFERENCES
    static SharedPreferences sharedPref;
    static SharedPreferences.Editor editor;

    /*
    * Dialog when the user says that the food is in the list of Top5 predictions
    * */
    static DialogInterface.OnClickListener dialogClickListener = new DialogInterface.OnClickListener() {
        @Override
        public void onClick(DialogInterface dialog, int which) {
            switch (which){
                case DialogInterface.BUTTON_POSITIVE:
                    //update:
                    //  - total number of classifications
                    //  - number of correct classifications in top5
                    //  - sum of the scores of the correct classifications
                    int correctClassifications = Utility.incrementSharedPrefInt(sharedPref,editor,Utility.CORRECT_CLASSIFICATIONS,1);
                    int totalClassifications = Utility.incrementSharedPrefInt(sharedPref,editor,Utility.TOTAL_CLASSIFICATIONS,1);
                    double scoresSum = Utility.incrementSharedPrefFloat(sharedPref,editor,Utility.SCORES_SUM,topScores[chosen]);

                    accuracyTop5Text.setText("Accuracy Top5: " + formatter.format( ((double)correctClassifications/(double)totalClassifications)*100) );
                    avgScoreText.setText("Avg Score         : " + formatter.format( (scoresSum/(double)correctClassifications)*100) );
                    done = true;
                    break;

                case DialogInterface.BUTTON_NEGATIVE:
                    break;
            }
        }
    };

    /*
    * Dialog when the user says that food is not in the list, update statistics
    * */
    static DialogInterface.OnClickListener dialogNotInTheList = new DialogInterface.OnClickListener(){
        @Override
        public void onClick(DialogInterface dialog, int which) {
            switch (which){
                case DialogInterface.BUTTON_POSITIVE:
                    //update:
                    //  - total number of classifications
                    int correctClassifications = Utility.incrementSharedPrefInt(sharedPref,editor,Utility.CORRECT_CLASSIFICATIONS,0);
                    int totalClassifications = Utility.incrementSharedPrefInt(sharedPref,editor,Utility.TOTAL_CLASSIFICATIONS,1);

                    accuracyTop5Text.setText(formatter.format(((double)correctClassifications/(double)totalClassifications)*100) + "%");
                    done = true;
                    break;

                case DialogInterface.BUTTON_NEGATIVE:
                    break;
            }
        }
    };

    /*
    * Dialog when the user is shown the Top1 prediction because its score is higher than the threshold
    * */
    static DialogInterface.OnClickListener dialogTop1Prediction = new DialogInterface.OnClickListener() {
        @Override
        public void onClick(DialogInterface dialogInterface, int which) {
            switch (which){
                case DialogInterface.BUTTON_POSITIVE:
                    //update:
                    //  - total number of classifications
                    //  - number of correct classifications in top5
                    //  - number of correct classifications in top1
                    //  - total number of Top1 predictions
                    //  - sum of the scores of the correct classifications
                    int totalTop1Classifications = Utility.incrementSharedPrefInt(sharedPref,editor,Utility.TOTAL_TOP1_CLASSIFICATIONS,1);
                    int correctClassifications = Utility.incrementSharedPrefInt(sharedPref,editor,Utility.CORRECT_CLASSIFICATIONS,1);
                    int totalClassifications = Utility.incrementSharedPrefInt(sharedPref,editor,Utility.CORRECT_CLASSIFICATIONS,1);
                    int correctTop1Classifications = Utility.incrementSharedPrefInt(sharedPref,editor,Utility.CORRECT_TOP1_CLASSIFICATION,1);
                    double scoresSum = Utility.incrementSharedPrefFloat(sharedPref,editor,Utility.SCORES_SUM,topScores[0]);

                    accuracyTop1Text.setText(formatter.format(((double)correctTop1Classifications/(double)totalTop1Classifications)*100) + "%");
                    accuracyTop5Text.setText(formatter.format(((double)correctClassifications/(double)totalClassifications)*100) + "%");
                    avgScoreText.setText(formatter.format((scoresSum/(double)correctClassifications)*100) + "%");
                    done = true;
                    break;

                case DialogInterface.BUTTON_NEGATIVE:
                    //update:
                    //  - total number of classifications
                    //  - total number of Top1 predictions
                    int totalTop1Classifications2 = Utility.incrementSharedPrefInt(sharedPref,editor,Utility.TOTAL_TOP1_CLASSIFICATIONS,1);
                    int correctClassifications2 = Utility.incrementSharedPrefInt(sharedPref,editor,Utility.CORRECT_CLASSIFICATIONS,0);
                    int totalClassifications2 = Utility.incrementSharedPrefInt(sharedPref,editor,Utility.TOTAL_CLASSIFICATIONS,1);
                    int correctTop1Classifications2 = Utility.incrementSharedPrefInt(sharedPref,editor,Utility.CORRECT_TOP1_CLASSIFICATION,0);

                    accuracyTop1Text.setText(formatter.format(((double)correctTop1Classifications2/(double)totalTop1Classifications2)*100) + "%");
                    accuracyTop5Text.setText(formatter.format(((double)correctClassifications2/(double)totalClassifications2)*100) + "%");
                    break;
            }
        }
    };

    /*
    * Broadcast receiver for receiving status updates from the IntentService
    */
    private class CNNResultReceiver extends BroadcastReceiver {
        // Prevents instantiation
        private CNNResultReceiver() {
        }

        // Called when the BroadcastReceiver gets an Intent it's registered to receive
        @Override
        public void onReceive(Context context, Intent intent) {
            //handle intent received
            if (!intent.hasExtra(Utility.SCORES_EXTRA)) {
                Toast.makeText(getApplicationContext(), "Network unreachable", Toast.LENGTH_SHORT).show();
                progressDialog.dismiss();
                return;
            }
            Bundle extras = intent.getExtras();
            topScores = extras.getDoubleArray(Utility.SCORES_EXTRA);
            topLabels = extras.getStringArray(Utility.LABELS_EXTRA);
            myItems = new RecognizedClass[5];
            for (int i = 0; i < 5; i++) {
                myItems[i] = new RecognizedClass(topLabels[i], topScores[i]);
            }

            done = false;

            //if the score of the top 1 is higher than the threshold, show the dish
            if (myItems[0].getScore() >= Utility.TOP1_THRESHOLD) {
                Intent intentDishes = new Intent(getApplicationContext(), DishView.class);
                intentDishes.putExtra(Utility.LABEL_EXTRA, myItems[0].getLabel());
                intentDishes.putExtra(Utility.TOP1_CALL, true);
                startActivity(intentDishes);
            }

            //if the scores are between the thresholds, show the list
            imageView.setImageBitmap(imageToShow);
            imageView.setVisibility(View.VISIBLE);
            infoText.setVisibility(View.VISIBLE);
            infoText.setText("Is the food among these?");
            listView.setVisibility(View.VISIBLE);
            noneOfTheAbove.setVisibility(View.VISIBLE);
            cameraBig.setVisibility(View.INVISIBLE);
            if (myItems[0].getScore() <= Utility.TOP5_THRESHOLD) { //TODO fix the check
                infoText.setText("This is not a food!");
                listView.setVisibility(View.INVISIBLE);
                cameraBig.setVisibility(View.INVISIBLE);
            }
            //Update the resultsAdapter in order to show the result on the activity
            resultsAdapter = new CustomList(MainActivity.this, topLabels, topScores);
            listView.setAdapter(resultsAdapter);
            resultsAdapter.notifyDataSetChanged();

            //close the progress dialog when classification is over
            progressDialog.dismiss();
        }
    }

    /*
    * Initialize the list of recognized classes
    * */
    public RecognizedClass[] initializeList() {
        RecognizedClass[] list = new RecognizedClass[5];
        topLabels = new String[5];
        topScores = new double[5];
        for (int i = 0; i < 5; i++) {
            list[i] = new RecognizedClass("nothing", 0.0);
            topScores[i] = 0.0;
            topLabels[i] = "nothing";
        }
        return list;
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            Log.i("MainActivity", "Permission: " + permissions[0] + "was " + grantResults[0]);
            //resume the activity
            return;
        } else {
            //stop the activity
            finish();
            System.exit(0);
        }
    }

    public boolean isStoragePermissionGranted() {
        if (Build.VERSION.SDK_INT >= 23) {
            if (checkSelfPermission(Manifest.permission.WRITE_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED) {
                Log.i("MainActivity", "Permission is granted");
                return true;
            } else {
                Log.i("MainActivity", "Permission is revoked");
                ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE}, Utility.PERMISSION_REQUEST);
                return false;
            }
        } else { //permission is automatically granted on sdk<23 upon installation
            Log.i("MainActivity", "Permission is granted");
            return true;
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        context = getApplicationContext();

        //Initialization of the Shared Preferences
        sharedPref = PreferenceManager.getDefaultSharedPreferences(context);
        editor = sharedPref.edit();

        //Ask for permissions
        isStoragePermissionGranted();

        // Initialization of the intent for the BroadcastReceiver
        statusIntentFilter = new IntentFilter(Utility.BROADCAST_ACTION);
        cnnResultReceiver = new CNNResultReceiver();
        // Registers the cnnResultReceiver to receive its intent
        LocalBroadcastManager.getInstance(this).registerReceiver(cnnResultReceiver, statusIntentFilter);

        Toolbar toolbar = (Toolbar) findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);
        mSectionsPagerAdapter = new SectionsPagerAdapter(getSupportFragmentManager());  //create the resultsAdapter that will return a fragment for each of the three primary sections of the activity

        // Set up the ViewPager with the sections resultsAdapter.
        mViewPager = (ViewPager) findViewById(R.id.container);
        mViewPager.setAdapter(mSectionsPagerAdapter);
        mViewPager.setCurrentItem(1); //set the middle page as main page

        myItems = initializeList(); //initialize the list of recognized classes to nothing
        adapterDishes = new ArrayAdapter<String>(getApplicationContext(), R.layout.item_list_dishes, Utility.capitalizeLabels());

        //TAKE PHOTO BUTTON
        camera = (FloatingActionButton) findViewById(R.id.camera);
        camera.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) { //Take Picture Intent
                Intent takePictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                if (takePictureIntent.resolveActivity(getPackageManager()) != null) {
                    startActivityForResult(takePictureIntent, Utility.PHOTO_CODE);
                }
            }
        });

        //CHANGE MODE BUTTON
        fab = (FloatingActionButton) findViewById(R.id.fab);
        fab.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                int mode = sharedPref.getInt(Utility.MODE, 0);
                if (mode == 0) {
                    if (isNetworkAvailable()) {
                        editor.putInt(Utility.MODE, 1);
                        Snackbar.make(view, "Online Mode", Snackbar.LENGTH_LONG)
                                .setAction("Action", null).show();
                    } else {
                        Snackbar.make(view, "No Internet Connection", Snackbar.LENGTH_LONG)
                                .setAction("Action", null).show();
                    }
                } else {
                    editor.putInt(Utility.MODE, 0);
                    Snackbar.make(view, "Offline Mode", Snackbar.LENGTH_LONG)
                            .setAction("Action", null).show();
                }
                editor.commit();
            }
        });

        //CLEAR BUTTON
        clear = (FloatingActionButton) findViewById(R.id.clear);
        clear.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                listView.setVisibility(View.INVISIBLE);
                imageView.setVisibility(View.INVISIBLE);
                cameraBig.setVisibility(View.VISIBLE);
                infoText.setVisibility(View.INVISIBLE);
                noneOfTheAbove.setVisibility(View.INVISIBLE);
            }
        });
    }

    private boolean isNetworkAvailable() {
        ConnectivityManager connectivityManager
                = (ConnectivityManager) getSystemService(Context.CONNECTIVITY_SERVICE);
        NetworkInfo activeNetworkInfo = connectivityManager.getActiveNetworkInfo();
        return activeNetworkInfo != null && activeNetworkInfo.isConnected();
    }

    /*
    * Function to crop the captured image
    * */
    public void cropCapturedImage(Uri picUri) {
        Intent cropIntent = new Intent("com.android.camera.action.CROP");
        cropIntent.setDataAndType(picUri, "image/*");
        cropIntent.putExtra("crop", "true");
        cropIntent.putExtra("aspectX", 1);
        cropIntent.putExtra("aspectY", 1);
        //indicate output X and Y
        cropIntent.putExtra("outputX", 256);
        cropIntent.putExtra("outputY", 256);
        //retrieve data on return
        cropIntent.putExtra("return-data", true);
        //start the activity - we handle returning in onActivityResult
        startActivityForResult(cropIntent, Utility.CROP_CODE);
    }

    @Override
    public void onActivityResult(int requestCode, int resultCode, Intent data) {
        if (resultCode == RESULT_OK) { //if operation succeded
            if (requestCode == Utility.PHOTO_CODE) { //take photo case
                Bundle extras = data.getExtras();
                Bitmap imageBitmap = (Bitmap) extras.get("data");
                File dir = new File(Utility.TEMP_DIR_PATH);
                if (!dir.exists())
                    dir.mkdirs();
                File file = new File(dir, Utility.TEMP_IMAGE_NAME);
                FileOutputStream fOut = null;
                try {
                    fOut = new FileOutputStream(file);
                    imageBitmap.compress(Bitmap.CompressFormat.PNG, 0, fOut);
                    fOut.flush();
                    fOut.close();
                } catch (Exception e) {
                    e.printStackTrace();
                }

                //TODO uncomment the last function to crop the photo and comment the try catch statement
                imageToShow = Bitmap.createScaledBitmap(imageBitmap, 227, 227, true);
                try {
                    // show the progress dialog
                    progressDialog = ProgressDialog.show(this, "Recognizing", "Analysis in progress...", true);
                    // Launch the Background Service
                    Intent serviceIntent = new Intent(this, CNNService.class);
                    serviceIntent.putExtra(Utility.BITMAP_EXTRA, imageToShow);
                    this.startService(serviceIntent);
                } catch (Exception e) {
                    e.printStackTrace();
                }
                //cropCapturedImage(Uri.fromFile(file));
            }


            if (requestCode == Utility.CROP_CODE) { //crop photo case
                if (data != null) {
                    Bundle extras = data.getExtras();                         // get the returned data
                    Bitmap bmp = extras.getParcelable("data");                // get the cropped bitmap
                    imageToShow = Bitmap.createScaledBitmap(bmp, 227, 227, true); //make the foto squared

                    try {
                        // show the progress dialog
                        progressDialog = ProgressDialog.show(this, "Recognizing", "Analysis in progress...", true);
                        // Launch the Background Service
                        Intent serviceIntent = new Intent(this, CNNService.class);
                        serviceIntent.putExtra(Utility.BITMAP_EXTRA, imageToShow);
                        this.startService(serviceIntent);
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }
            }
        }
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        // Inflate the menu; this adds items to the action bar if it is present.
        getMenuInflater().inflate(R.menu.menu_main, menu);
        return true;
    }

    /*
    * Handle action bar items clicks here. The action bar will automatically handle clicks on the Home/Up button, so long as you specify a parent activity in AndroidManifest.xml
    * */
    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        int id = item.getItemId();
        if (id == R.id.action_settings) {
            return true;
        }
        if (id == R.id.dishes_fragment) {
            mViewPager.setCurrentItem(0);
            return true;
        }
        if (id == R.id.main_fragment) {
            mViewPager.setCurrentItem(1);
            return true;
        }
        if (id == R.id.history_fragment) {
            mViewPager.setCurrentItem(2);
            return true;
        }
        return super.onOptionsItemSelected(item);
    }

    /*
    * A placeholder fragment containing a simple view.
    * */
    public static class PlaceholderFragment extends Fragment {
        private final static String ARG_SECTION_NUMBER = "section_number";  //the fragment argument representing the section number for this fragment

        public PlaceholderFragment() {
        }

        /**
         * Returns a new instance of this fragment for the given section
         * number.
         */
        public static PlaceholderFragment newInstance(int sectionNumber) {
            PlaceholderFragment fragment = new PlaceholderFragment();
            Bundle args = new Bundle();
            args.putInt(ARG_SECTION_NUMBER, sectionNumber);
            fragment.setArguments(args);
            return fragment;
        }

        @Override
        public void onActivityResult(int requestCode, int resultCode, Intent data) {
            super.onActivityResult(requestCode, resultCode, data);
        }

        @Override
        public View onCreateView(LayoutInflater inflater, ViewGroup container, Bundle savedInstanceState) {

            View rootView = null;

            resultsAdapter = new CustomList(getActivity(), topLabels, topScores);

            //FIRST SECTION: all the dishes
            if (getArguments().getInt(ARG_SECTION_NUMBER) == 1) {
                rootView = inflater.inflate(R.layout.fragment_dishes, container, false);
                dishesListView = (ListView) rootView.findViewById(R.id.dishesListView);
                dishesListView.setAdapter(adapterDishes);
                dishesListView.setOnItemClickListener(new AdapterView.OnItemClickListener() {
                    @Override
                    public void onItemClick(AdapterView<?> parent, View view, int position, long l) {
                        //call the activity to show the dish
                        Intent intent = new Intent(getContext(), DishView.class);
                        intent.putExtra(Utility.LABEL_EXTRA, Utility.labels[position]);
                        startActivity(intent);
                    }
                });
                dishesListView.setVisibility(View.VISIBLE);
            }
            //SECOND SECTION: main section, recognition of food
            if (getArguments().getInt(ARG_SECTION_NUMBER) == 2) {
                rootView = inflater.inflate(R.layout.fragment_main, container, false);

                cameraBig = (Button) rootView.findViewById(R.id.camera_big);
                cameraBig.setOnClickListener(new View.OnClickListener() {
                    @Override
                    public void onClick(View view) {
                        Intent takePictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                        getActivity().startActivityForResult(takePictureIntent, Utility.PHOTO_CODE);
                    }
                });

                imageView = (ImageView) rootView.findViewById(R.id.image);
                listView = (ListView) rootView.findViewById(R.id.listView);
                infoText = (TextView) rootView.findViewById(R.id.infoText);
                noneOfTheAbove = (TextView) rootView.findViewById(R.id.noneOfTheAbove);
                noneOfTheAbove.setOnClickListener(new View.OnClickListener() {
                    @Override
                    public void onClick(View view) {
                        if (!done) {
                            AlertDialog.Builder builder = new AlertDialog.Builder(getContext());
                            builder.setMessage("Are you sure the food is not in the list?").setPositiveButton("Yes", dialogNotInTheList)
                                    .setNegativeButton("No", dialogNotInTheList).show();
                        }
                    }
                });
                listView.setAdapter(resultsAdapter);
                //on click on the item start the activity show the dish
                listView.setOnItemClickListener(new AdapterView.OnItemClickListener() {
                    @Override
                    public void onItemClick(AdapterView<?> parent, View view, int position, long l) {
                        Intent intent = new Intent(getContext(), DishView.class);
                        intent.putExtra(Utility.LABEL_EXTRA, topLabels[position]);
                        startActivity(intent);
                    }
                });
                //on long click on the item ask if it is the correct prediction
                listView.setOnItemLongClickListener(new AdapterView.OnItemLongClickListener() {
                    @Override
                    public boolean onItemLongClick(AdapterView<?> adapterView, View view, int i, long l) {
                        if (!done) {
                            AlertDialog.Builder builder = new AlertDialog.Builder(getContext());
                            builder.setMessage("Is this the food in the image?").setPositiveButton("Yes", dialogClickListener)
                                    .setNegativeButton("No", dialogClickListener).show();
                        }
                        return true;
                    }
                });
                listView.setVisibility(View.INVISIBLE);
                noneOfTheAbove.setVisibility(View.INVISIBLE);
            }
            //THIRD SECTION: history
            if (getArguments().getInt(ARG_SECTION_NUMBER) == 3){
                rootView = inflater.inflate(R.layout.fragment_history, container, false);

                accuracyTop1Text = (TextView) rootView.findViewById(R.id.accuracy_top1);
                accuracyTop5Text = (TextView) rootView.findViewById(R.id.accuracy_top5);
                avgScoreText = (TextView) rootView.findViewById(R.id.score_correct_class);

                int correctClassifications = sharedPref.getInt(Utility.CORRECT_CLASSIFICATIONS, 0);
                int totalClassifications = sharedPref.getInt(Utility.TOTAL_CLASSIFICATIONS,0);
                int correctTop1Classifications = sharedPref.getInt(Utility.CORRECT_TOP1_CLASSIFICATION,0);
                int totalTop1Classifications = sharedPref.getInt(Utility.TOTAL_TOP1_CLASSIFICATIONS,0);
                double scoresSum = sharedPref.getFloat(Utility.SCORES_SUM,0.0f);

                accuracyTop1Text.setText(formatter.format(((double)correctTop1Classifications/(double)totalTop1Classifications)*100) + "%");
                accuracyTop5Text.setText(formatter.format(((double)correctClassifications/(double)totalClassifications)*100) + "%");
                avgScoreText.setText(formatter.format((scoresSum/(double)correctClassifications)*100) + "%");
            }
            return rootView;
        }
    }

    /*
    * A FragmentPageAdapter that returns a fragment corresponding to one of the sections/tabs/pages.
    * */
    public class SectionsPagerAdapter extends FragmentPagerAdapter {

        public SectionsPagerAdapter(FragmentManager fm) {
            super(fm);
        }

        /*
         * Called to instantiate the fragment for the given page. Return a PlaceholderFragment (defined as a static inner class below)
         */
        @Override
        public Fragment getItem(int position) {
            return PlaceholderFragment.newInstance(position + 1);
        }

        @Override
        public int getCount() {
            return 3;   //show 3 total pages
        }

        @Override
        public CharSequence getPageTitle(int position) {
            switch (position) {
                case 0:
                    return "SECTION 1";
                case 1:
                    return "SECTION 2";
                /*case 2:
                    return "SECTION 3";*/
            }
            return null;
        }
    }
}