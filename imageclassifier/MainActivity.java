package com.example.imageclassifier;

import android.annotation.SuppressLint;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.view.Window;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;


import com.example.imageclassifier.ml.ModelUnquant;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Objects;

public class MainActivity extends AppCompatActivity {
     ImageView Image;
     Button predict;
     Button view;
     TextView result;
    Bitmap bitmap;

    public MainActivity() {
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        Objects.requireNonNull(getSupportActionBar()).hide();
        setContentView(R.layout.activity_main);
        String[] labels=new String[7] ;
        try {
            BufferedReader bufferedReader=new BufferedReader(new InputStreamReader(getAssets().open("labels.txt")));
            String line= bufferedReader.readLine();
            int cnt=0;
            while (line!=null && cnt < 7){

                labels[cnt]=line;
                cnt++;

            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }


        Image=findViewById(R.id.Image);
        view = findViewById(R.id.view);
        predict=findViewById(R.id.predict);
        result=findViewById(R.id.result);
        view.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent=new Intent(Intent.ACTION_GET_CONTENT);
                intent.setType("image/*");
                startActivityForResult(intent,100);
            }
        });
        predict.setOnClickListener(new View.OnClickListener() {
            @SuppressLint("SetTextI18n")
            @Override
            public void onClick(View view) {
                try {
                    ModelUnquant model = ModelUnquant.newInstance(MainActivity.this);

                    // Creates inputs for reference.
                    TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);
                    bitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true);

                    // Debug: Log the bitmap dimensions
                    Log.d("DEBUG", "Bitmap width: " + bitmap.getWidth() + ", height: " + bitmap.getHeight());

                    // Debug: Check if the bitmap is null
                    if (bitmap == null) {
                        Log.e("ERROR", "Bitmap is null.");
                        return;
                    }

                    // Debug: Check pixel values in the bitmap
                    int pixel = bitmap.getPixel(0, 0);
                    int red = Color.red(pixel);
                    int green = Color.green(pixel);
                    int blue = Color.blue(pixel);
                    Log.d("DEBUG", "Pixel 0,0: R=" + red + ", G=" + green + ", B=" + blue);

                    // Normalize pixel values to [0, 1]
                    float[] floatValues = new float[224 * 224 * 3];
                    for (int i = 0; i < 224; i++) {
                        for (int j = 0; j < 224; j++) {
                            int pixelValue = bitmap.getPixel(i, j);
                            floatValues[(i * 224 + j) * 3 + 0] = (Color.red(pixelValue) - 127) / 128.0f;
                            floatValues[(i * 224 + j) * 3 + 1] = (Color.green(pixelValue) - 127) / 128.0f;
                            floatValues[(i * 224 + j) * 3 + 2] = (Color.blue(pixelValue) - 127) / 128.0f;
                        }
                    }

                    // Load the normalized pixel values into the TensorBuffer
                    inputFeature0.loadArray(floatValues, inputFeature0.getShape());

                    // Runs model inference and gets result.
                    ModelUnquant.Outputs outputs = model.process(inputFeature0);
                    TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();
                    int predictedIndex = getMax(outputFeature0.getFloatArray());
                    String predictedLabel = labels[predictedIndex];

                    // Display the result as text.
                    result.setText(predictedLabel);


                    // Releases model resources if no longer used.
                    model.close();
                } catch (IOException e) {
                    // TODO Handle the exception
                }
            }
        });

    }
    int getMax(float[] arr){
        int max=0;
        for(int i=0;i<arr.length;i++){
            if (arr[i]>arr[max]) max=i;
        }
        return max;
    }


    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        if(requestCode==100){
            if(data!=null){
                Uri uri= data.getData();
                try {
                    bitmap= MediaStore.Images.Media.getBitmap(this.getContentResolver(),uri);
                    Image.setImageBitmap(bitmap);
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
            }
        }
        super.onActivityResult(requestCode, resultCode, data);
    }
}