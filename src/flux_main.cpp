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

// debug flag
//#define DEBUG



const string input_path = "highway/input";
const int frame_num = 1700;


int main(int argc, char** argv)
{
    //FluxTensorMethod flux (3,3,3,3);
	FluxTensorMethod flux (5,5,5,5);


	FileNameGenerator input_file_name_generator(input_path+"/in", JPG);

    initialize_windows();

    Mat input_frame, output_frame, temp_frame;
    string frame_name;

    for(int frame_id = 1; frame_id < frame_num; frame_id++)
    {
        frame_name = input_file_name_generator.get_frame_name(frame_id);
        input_frame = imread(frame_name, 1);

        flux.update(input_frame, output_frame);

        update_windows(2, &input_frame, &output_frame);
        if(waitKey(10) != -1)
            break;
    }

	return 0;
}
