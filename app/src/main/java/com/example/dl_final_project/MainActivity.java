package com.example.dl_final_project;

import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;

import android.net.Uri;
import android.os.Build;
import android.os.Bundle;

import androidx.annotation.Nullable;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.media.ThumbnailUtils;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import com.example.dl_final_project.ml.Model;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class MainActivity extends AppCompatActivity {

    TextView result;
    ImageView imageView;
    Button picture, gallery;
    int imageSize = 32;
    private ThumbnailUtils thumbnailUtils;
    boolean False;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        result = findViewById(R.id.result);
        imageView = findViewById(R.id.imageView);
        //picture = findViewById(R.id.button);
        gallery = findViewById(R.id.button2);

/*        picture.setOnClickListener(new View.OnClickListener() {
            @RequiresApi(api = Build.VERSION_CODES.M)
            @Override
            public void onClick(View view) {
                // Launch camera if we have permission
                if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                    Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                    System.out.println("lunch Camera");
                    startActivityForResult(cameraIntent, 1);
                } else {
                    System.out.println("CAMERA DOES NOT LUNCHING");

                    //Request camera permission if we don't have it.
                    requestPermissions(new String[]{Manifest.permission.CAMERA}, 100);
                }
            }
        });
*/
        gallery.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                System.out.println("lunch Gallery");
                Intent galleryIntent = new Intent(Intent.ACTION_PICK,MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                startActivityForResult(galleryIntent,2);
            }
        });
    }

    @Override
    public void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        // Processing Image
        if (resultCode == RESULT_OK) {
           /* if(requestCode == 1){
                System.out.println("START setImageBitmap using Camera");

                //get the image from Camera and resizing it
                Bitmap image = (Bitmap) data.getExtras().get("Data");
                //resize our bitmap
                int dimension = Math.min(image.getWidth(),image.getHeight()); // we get the smallest dimension of our image
                //rescale our image to fit these dimensiioins
                image = thumbnailUtils.extractThumbnail(image,dimension,dimension);

                System.out.println("before setImageBitmap using Camera");

                imageView.setImageBitmap(image);

                System.out.println("after setImageBitmap using Camera");

                //this prepares our Bitmap image to be used for classification from our model
                image = Bitmap.createScaledBitmap(image,imageSize,imageSize,False);

                classifyImage(image);
            }else {*/
                if(requestCode == 2){
                    //get the image from Gallery and resizing it
                    Uri dat = data.getData();
                    Bitmap  image = null;
                    try {
                        image = MediaStore.Images.Media.getBitmap(this.getContentResolver(),dat);
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                    //System.out.println("before setImageBitmap using GALLERY");
                    imageView.setImageBitmap(image);
                    //System.out.println("AFTER setImageBitmap using GALLERY");

                    //this prepares our Bitmap image to be used for classification from our model
                    image = Bitmap.createScaledBitmap(image,imageSize,imageSize,False);

                    classifyImage(image);
                }
            }
        super.onActivityResult(requestCode, resultCode, data);
    }

    private void classifyImage(Bitmap image) {
        try {
            Model model = Model.newInstance(getApplicationContext());

            System.out.println("FIRST THE CLASSIFICATION METHOD");

            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 32, 32, 3}, DataType.FLOAT32);
            //byteBuffer is what is going to contain the pixel values from our Bitmap
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3); //specify how large it should be
            byteBuffer.order(ByteOrder.nativeOrder());

            //System.out.println("SECOND THE CLASSIFICATION METHOD");

            //array of those pixel values
            int[] intValues = new int[imageSize * imageSize];
            image.getPixels(intValues,0, image.getWidth(), 0, 0, image.getWidth(), image.getHeight());

            //System.out.println("THIRD THE CLASSIFICATION METHOD");

            int pixel = 0;  //to keep track of the pixel number
            //iterate over each pixel and extract R, G, B values. Add those values individually to the bytebuffer.
            for (int i =0 ; i< imageSize ; i++){
                for (int j =0; j< imageSize ; j++){
                    //get the value of the pixels
                    int val = intValues[pixel++];  //RGB
                    byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f/1));
                    byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f/1));
                    byteBuffer.putFloat((val & 0xFF) * (1.f/1));
                }
            }
            //System.out.println("FOURTH THE CLASSIFICATION METHOD");

            inputFeature0.loadBuffer(byteBuffer);

            //System.out.println("FIFTH THE CLASSIFICATION METHOD");

            // Runs model inference and gets result.
            Model.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            //System.out.println("SIXTH THE CLASSIFICATION METHOD");

            //the idea : the position with the highest confidence is what our model thought that image was of
            float[] confidences = outputFeature0.getFloatArray();
            //find the index of the class with the biggest confidence.
            int maxPos = 0;
            float maxConfidence = 0;
            for(int i=0; i< confidences.length; i++){
                if(confidences[i] > maxConfidence){
                    maxConfidence = confidences[i];
                    maxPos = i;
                }
            }

            String[] classes = {"Apple","Banana", "Orange"};
            result.setText(classes[maxPos]);
            //System.out.println("outputFeature0 : "+classes[maxPos]);

            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
        }

    }
}