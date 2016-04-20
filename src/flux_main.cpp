#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv2/opencv.hpp>


#include <iostream>
#include <string>
#include <cstdarg>

#include "file_name_generator.hpp"
#include "window_manager.hpp"
#include "FluxTensorMethod.hpp"

using namespace cv;
using namespace std;

#define SHOW_ALL

const string input_path = "highway/input";
//const string input_path = "highway_high_res/1080p";
const string gt_frame_prefix = "highway/groundtruth/gt";
const int frame_num = 1700;

long flux_metrics[] = {0,0,0,0};
long subtract_metrics[] = {0,0,0,0};
long optical_flow_metrics[] = {0,0,0,0};

void test(Mat & result, Mat & gt_frame, long * metrics)
{
    const uchar * result_pixel_ptr;
    const uchar * gt_pixel_ptr;

    for(int row = 0; row < result.rows; ++row)
    {
    	//cout<<"row " << row <<endl;
        result_pixel_ptr = result.ptr(row);
        gt_pixel_ptr = gt_frame.ptr(row);
        for(int col = 0; col < result.cols; ++col)
        {
        	uchar gt = *gt_pixel_ptr++;
        	//bool gt_mask = (gt > 0) ? true : false;
        	bool gt_mask = (gt >= 170) ? true : false;
        	uchar res;
        	res = *result_pixel_ptr++;
        	if(gt == 85)
        		continue;
        	bool res_mask = (res == 255) ? true : false;

        	if(gt_mask && res_mask)
        		metrics[0]++;
        	else if(!gt_mask && !res_mask)
        		metrics[1]++;
        	else if(!gt_mask && res_mask)
        		metrics[2]++;
        	else if(gt_mask && !res_mask)
        		metrics[3]++;
        }
    }
}

void print_metrics(string method_name, long * metrics){

	cout << method_name << endl;

	long tp = metrics[0];
	long tn = metrics[1];
	long fp = metrics[2];
	long fn = metrics[3];

    double recall = (double)tp/(tp+fn);
    double specificity = (double) tn/(tn+fp);
    double FPR = (double) fp/(fp+tn);
    double FNR = (double) fn/(tp+fn);
    double PWC = (double) 100*(fn+fp)/(tp+fn+fp+tn);
    double precision = (double)tp/(tp+fp);
    double F1 = (2*precision*recall)/(precision+recall);

    cout << "tp: " <<tp<< endl;
    cout << "tn: " <<tn<< endl;
    cout << "fp: " <<fp<< endl;
    cout << "fn: " <<fn<< endl;
    cout << /*"Recall = " <<*/ recall << endl;
    cout << /*"Specificity = " <<*/ specificity << endl;
    cout << /*"FPR = " <<*/ FPR << endl;
    cout << /*"FNR = " <<*/ FNR << endl;
    cout << /*"PWC = " <<*/ PWC << endl;
    cout << /*"F1 = " <<*/ F1 << endl;
    cout << /*"Precision = " <<*/ precision << endl;
    cout << endl;
}

void subtract(Mat & input_frame, Mat & output_frame){

	static Mat prev_gray_frame;
	static bool first_frame = true;

	Mat gray_input_frame, sub_output_frame;
	cvtColor(input_frame, gray_input_frame, cv::COLOR_BGR2GRAY);

	if(first_frame){
		prev_gray_frame = gray_input_frame.clone();
		output_frame = gray_input_frame.clone();
		first_frame = false;
		return;
	}

	subtract(gray_input_frame, prev_gray_frame, sub_output_frame);
	output_frame = sub_output_frame.clone();
	threshold(sub_output_frame, output_frame, 20, 255,THRESH_BINARY);
	prev_gray_frame = gray_input_frame.clone();

}

void optical_flow(Mat & input_frame, Mat & output_frame){

	 static bool first_frame = true;
	 static UMat prevgray;

	 Mat gray, flow;
	 UMat  flowUmat;

	 cvtColor(input_frame, gray, cv::COLOR_BGR2GRAY);

	 if(first_frame){
		 first_frame = false;
		 gray.copyTo(prevgray);
		 gray.copyTo(output_frame);
		 return;
	 }

	 calcOpticalFlowFarneback(prevgray, gray, flowUmat, 0.4, 1, 12, 2, 8, 1.2, 0);
	 flowUmat.copyTo(flow);
	 gray.copyTo(prevgray);

	 Mat original = gray.clone();
	 original.setTo(Scalar(0,0,0));

	 for (int y = 0; y < original.rows; y += 5) {
		 for (int x = 0; x < original.cols; x += 5)
	     {
	              // get the flow from y, x position * 10 for better visibility
	              const Point2f flowatxy = flow.at<Point2f>(y, x)*5;
	                     // draw line at flow direction
	              line(original, Point(x, y), Point(cvRound(x + flowatxy.x), cvRound(y + flowatxy.y)), Scalar(255,255,255));
	                                                         // draw initial point
	              circle(original, Point(x, y), 1, Scalar(0, 0, 0), -1);
	      }
	 }

	 output_frame = original.clone();

}

int main(int argc, char** argv)
{
    FluxTensorMethod flux (5,5,5,5,30);


	FileNameGenerator input_file_name_generator(input_path+"/in", JPG);
	FileNameGenerator ground_truth_file_name_generator(gt_frame_prefix, PNG);

    initialize_windows();

    Mat input_frame,flux_output_frame, gt_frame, sub_output_frame, optical_flow_frame;

    string frame_name, gt_name;

    for(int frame_id = 1; frame_id < frame_num; frame_id++)
    {
        frame_name = input_file_name_generator.get_frame_name(frame_id);
        gt_name = ground_truth_file_name_generator.get_frame_name(frame_id);

        input_frame = imread(frame_name, 1);

        gt_frame = imread(gt_name, 0);

        flux.update(input_frame, flux_output_frame);
       // flux_output_frame = input_frame.clone();
          test(flux_output_frame, gt_frame, flux_metrics);

#ifdef SHOW_ALL
        subtract(input_frame, sub_output_frame);
        optical_flow(input_frame, optical_flow_frame);
        test(sub_output_frame, gt_frame, subtract_metrics);
        test(optical_flow_frame, gt_frame, optical_flow_metrics);
        update_windows(4, &input_frame, &flux_output_frame, &sub_output_frame, &optical_flow_frame);
#else
        update_windows(2, &input_frame, &flux_output_frame);
#endif

        if(waitKey(10) != -1)
            break;

    }

    print_metrics("flux tensor", flux_metrics);

#ifdef SHOW_ALL
    print_metrics("subtract", subtract_metrics);
    print_metrics("optical flow", optical_flow_metrics);
#endif

    return 0;

}


