package com.example.bharath.helloubuntu;

import android.content.Context;
import android.os.Bundle;
import android.os.Looper;
import android.os.Message;
import android.support.design.widget.FloatingActionButton;
import android.support.design.widget.Snackbar;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.util.Log;
import android.view.Surface;
import android.view.SurfaceView;
import android.view.View;
import android.view.Menu;
import android.view.MenuItem;
import android.view.Window;
import android.view.WindowManager;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import android.os.Handler;

import com.hanks.htextview.HTextView;

import java.util.logging.LogRecord;

public class MainActivity extends AppCompatActivity{

    private CameraBridgeViewBase mOpenCvCameraView;
    Mat mRgba,mRgbaF,mRgbaT,t;
    info.hoang8f.widget.FButton detect;
    CascadeClassifier face_cascade;
    MatOfRect faces=null;
    com.hanks.htextview.HTextView title;
    int flag=0,count=0;
    Handler mHandler;
    private int mCameraId = 0; //add this one
    @Override
    protected void onCreate(Bundle savedInstanceState) {

        requestWindowFeature(Window.FEATURE_NO_TITLE);
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN,
                WindowManager.LayoutParams.FLAG_FULLSCREEN);

        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);


        android.support.design.widget.FloatingActionButton fab=(android.support.design.widget.FloatingActionButton)findViewById(R.id.fab);

        detect=(info.hoang8f.widget.FButton)findViewById(R.id.detect);

        title = (com.hanks.htextview.HTextView) findViewById(R.id.title);

        timer();

        fab.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                //detect(t);
                swapCamera();
            }
        });

        detect.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                detect(t);
            }
        });

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.HelloOpenCvView);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);

        //


        mOpenCvCameraView.setCvCameraViewListener(new CameraBridgeViewBase.CvCameraViewListener() {

            @Override
            public void onCameraViewStarted(int width, int height) {


                mRgba = new Mat(height, width, CvType.CV_8UC4);
                mRgbaF = new Mat(height, width, CvType.CV_8UC4);
                mRgbaT = new Mat(width, width, CvType.CV_8UC4);  // NOTE width,width is NOT a typo

            }


            @Override
            public void onCameraViewStopped() {

            }


            @Override
            public Mat onCameraFrame(Mat inputFrame) {

                mRgba = inputFrame;
                // Rotate mRgba 90 degrees
                Core.transpose(mRgba, mRgbaT);
                Imgproc.resize(mRgbaT, mRgbaF, mRgbaF.size(), 0, 0, 0);
                Core.flip(mRgbaF, mRgba, -1);
                t=mRgba;
                if(flag==1) {
                    for (Rect rect : faces.toArray()) {
                        Imgproc.rectangle(mRgba, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height),
                                new Scalar(0, 255, 0), 5);
                    }
                }
                return mRgba;

            }

        });



    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        // Inflate the menu; this adds items to the action bar if it is present.
        getMenuInflater().inflate(R.menu.menu_main, menu);
        return true;
    }

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Toast.makeText(getApplicationContext(),"OpenCV loaded successfully!",Toast.LENGTH_SHORT).show();
                    load_cascade();
                    mOpenCvCameraView.enableView();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    private void swapCamera() {
        mCameraId = mCameraId^1; //bitwise not operation to flip 1 to 0 and vice versa
        mOpenCvCameraView.disableView();
        mOpenCvCameraView.setCameraIndex(mCameraId);
        mOpenCvCameraView.enableView();
    }

    @Override
    public void onResume()
    {
        super.onResume();
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_1_0, this, mLoaderCallback);
    }

    @Override
    public void onPause()
    {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    public void load_cascade(){


        try {
            InputStream is = getResources().openRawResource(R.raw.face);
            File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
            File mCascadeFile = new File(cascadeDir, "face.xml");
            FileOutputStream os = new FileOutputStream(mCascadeFile);

            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = is.read(buffer)) != -1) {
                os.write(buffer, 0, bytesRead);
            }
            is.close();
            os.close();

            face_cascade = new CascadeClassifier(mCascadeFile.getAbsolutePath());
            face_cascade.load(mCascadeFile.getAbsolutePath());
            if(face_cascade.empty())
            {
                Toast.makeText(getApplicationContext(),"Error loading cascade file.",Toast.LENGTH_SHORT).show();
                return;
            }
            else
            {
                Toast.makeText(getApplicationContext(),"Loaded from "+mCascadeFile.getAbsolutePath(),Toast.LENGTH_SHORT).show();
            }
        } catch (IOException e) {
            e.printStackTrace();
            Toast.makeText(getApplicationContext(),"Error : "+e ,Toast.LENGTH_SHORT).show();
        }
    }

    public void detect(Mat t){

        Mat mRgba=new Mat();
        Mat mGrey=new Mat();

        faces = new MatOfRect();
        //MatOfRect eyes = new MatOfRect();

        t.copyTo(mRgba);
        t.copyTo(mGrey);
        Imgproc.cvtColor(mRgba, mGrey, Imgproc.COLOR_BGR2GRAY);
        Imgproc.equalizeHist(mGrey, mGrey);

        face_cascade.detectMultiScale(mGrey, faces, 1.3, 5);


        if(faces.toArray().length > 0)
        {
            flag=1;
        }
        //Imgproc.rectangle(t,new Point(faces.));

        this.runOnUiThread(new Runnable() {
            public void run() {
                Toast.makeText(MainActivity.this, "Faces count : " + faces.toArray().length, Toast.LENGTH_SHORT).show();
            }
        });



    }

    public void timer()
    {

        new Thread(new Runnable() {
            @Override
            public void run() {
                // TODO Auto-generated method stub
                while (true) {
                    try {
                        Thread.sleep(5000);
                        mHandler.post(new Runnable() {

                            @Override
                            public void run() {
                                // TODO Auto-generated method stub
                                if(count%2==0)
                                {
                                    title.animateText("HAAR based face detection");
                                }
                                else
                                {
                                    title.animateText("OpenCV is used on android");
                                }
                                count++;
                                title.animate();

                            }
                        });
                    } catch (Exception e) {
                        // TODO: handle exception
                    }
                }
            }
        }).start();


    }


    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        // Handle action bar item clicks here. The action bar will
        // automatically handle clicks on the Home/Up button, so long
        // as you specify a parent activity in AndroidManifest.xml.
        int id = item.getItemId();

        //noinspection SimplifiableIfStatement
        if (id == R.id.action_settings) {
            return true;
        }

        return super.onOptionsItemSelected(item);
    }

}
