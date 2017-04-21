package thesis.myapplication;

import android.app.AlertDialog;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.preference.PreferenceManager;
import android.support.design.widget.CollapsingToolbarLayout;
import android.support.design.widget.FloatingActionButton;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.ShareActionProvider;
import android.support.v7.widget.Toolbar;
import android.view.View;
import android.webkit.WebView;
import android.webkit.WebViewClient;

import junit.framework.Assert;

import org.apache.commons.lang3.text.WordUtils;

public class DishView extends AppCompatActivity {
    private ShareActionProvider shareActionProvider;

    public static int getDrawable(Context context, String name) {
        Assert.assertNotNull(context);
        Assert.assertNotNull(name);

        return context.getResources().getIdentifier(name, "drawable", context.getPackageName());
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_dishview);
        Toolbar toolbar = (Toolbar) findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);
        CollapsingToolbarLayout collapsingToolbar = (CollapsingToolbarLayout) findViewById(R.id.toolbar_layout);

        WebView webView = (WebView) findViewById(R.id.webView);

        Intent intent = getIntent();
        final String food = intent.getStringExtra(Utility.LABEL_EXTRA);

        //set name and corresponding image on the collapsing toolbar
        collapsingToolbar.setTitle(WordUtils.capitalize(food));
        collapsingToolbar.setBackgroundResource(getDrawable(getApplicationContext(),food.replace(" ","_")));
        collapsingToolbar.setExpandedTitleTextAppearance(R.style.toolbar_text);
        webView.setWebViewClient(new WebViewClient() {
            public boolean shouldOverrideUrlLoading(WebView view, String url) {
                view.loadUrl(url);
                return true;
            }});
        webView.loadUrl("https://en.wikipedia.org/wiki/"+food.replace(" ","_"));


        FloatingActionButton fab = (FloatingActionButton) findViewById(R.id.fab);
        fab.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent sharingIntent = new Intent(android.content.Intent.ACTION_SEND);
                sharingIntent.setType("text/plain");
                String shareBody = "Ehi, I'm just eating "+food;
                sharingIntent.putExtra(android.content.Intent.EXTRA_SUBJECT, "Subject Here");
                sharingIntent.putExtra(android.content.Intent.EXTRA_TEXT, shareBody);

                startActivity(Intent.createChooser(sharingIntent, "Share via"));
            }
        });
        if(intent.getBooleanExtra(Utility.TOP1_CALL, false) == true){
            AlertDialog.Builder builder = new AlertDialog.Builder(this);
            builder.setMessage("Is this the food in the image?").setPositiveButton("Yes", MainActivity.dialogTop1Prediction)
                    .setNegativeButton("No", MainActivity.dialogTop1Prediction).show();
        }
    }
}
