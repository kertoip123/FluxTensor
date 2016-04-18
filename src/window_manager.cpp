#include <opencv2/highgui/highgui.hpp>
#include "window_manager.hpp"

void initialize_windows()
{
	namedWindow(INPUT, CV_WINDOW_AUTOSIZE);
	namedWindow(FLUX_TENSOR, CV_WINDOW_AUTOSIZE);

	moveWindow(INPUT, WINDOW_PADDING, WINDOW_PADDING);
	moveWindow(FLUX_TENSOR, WINDOW_PADDING + WINDOW_X_POS_OFFSET, WINDOW_PADDING);
}

void update_windows(int windows_num, ...)
{
    va_list args;
    va_start(args, windows_num);
    Mat *img;
    img = va_arg(args, Mat *);
	imshow(INPUT, *img);
	img = va_arg(args, Mat *);
	imshow(FLUX_TENSOR, *img);
}
