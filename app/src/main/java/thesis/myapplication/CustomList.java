package thesis.myapplication;

import android.app.Activity;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ArrayAdapter;
import android.widget.ImageView;
import android.widget.TextView;

public class CustomList extends ArrayAdapter<String>{

    private final Activity context;
    private final String[] labels;
    private final double[] scores;
    public CustomList(Activity context, String[] labels, double[] scores) {
        super(context, R.layout.list_single, labels);
        this.context = context;
        this.labels = labels;
        this.scores = scores;

    }
    @Override
    public View getView(int position, View view, ViewGroup parent) {
        LayoutInflater inflater = context.getLayoutInflater();
        View rowView= inflater.inflate(R.layout.list_single, null, true);
        TextView dishScore = (TextView) rowView.findViewById(R.id.dish_score);
        TextView dishClass = (TextView) rowView.findViewById(R.id.dish_class);
        dishClass.setText(labels[position]);

        dishScore.setText((int)(scores[position]*100)+"%");
        return rowView;
    }
}