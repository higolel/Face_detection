#ifndef __TENSORFLOW_MTCNN_H__
#define __TENSORFLOW_MTCNN_H__

#include <tensorflow/c/c_api.h>
#include <cstdio>
#include <fstream>
#include <utility>
#include <vector>
#include <unistd.h>
#include <iostream>
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <sensor_msgs/image_encodings.h>
#include <face_plate_msgs/Face_pic.h>
#include <location/location.h>
#include <cmath>
#include <thread>
#include <queue>
#include <ctime>
#include <sys/time.h>

#define NMS_UNION 1
#define NMS_MIN 2

using namespace std;
using namespace cv;

struct face_landmark
{
	float x[5];
	float y[5];
};

struct face_box
{
	float x0;
	float y0;
	float x1;
	float y1;

	/* confidence score */
	float score;

	/*regression scale */

	float regress[4];

	/* padding stuff*/
	float px0;
	float py0;
	float px1;
	float py1;

	face_landmark landmark;
};

struct scale_window
{
	int h;
	int w;
	float scale;
};


void Nms_boxes(std::vector<face_box> &input, float threshold, int type, std::vector<face_box> &output);

void Regress_boxes(std::vector<face_box> &rects);

void Square_boxes(std::vector<face_box> &rects);

void Padding(int img_h, int img_w, std::vector<face_box> &rects);

void Process_boxes(std::vector<face_box> &input, int img_h, int img_w, std::vector<face_box> &rects);

void generate_bounding_box(const float * confidence_data, int confidence_size,
               const float * reg_data, float scale, float threshold,
               int feature_h, int feature_w, std::vector<face_box>&  output, bool transposed);


void set_input_buffer(std::vector<cv::Mat>& input_channels, float* input_data, const int height, const int width);

void Compute_pyramid_list(int height, int width, int min_size, float factor,std::vector<scale_window> &list);

void cal_landmark(std::vector<face_box>& box_list);

void set_box_bound(std::vector<face_box>& box_list, int img_h, int img_w);

void Run_pnet(TF_Session *sess, TF_Graph *graph, cv::Mat &img, scale_window &win, std::vector<face_box> &box_list);

void Run_rnet(TF_Session *sess, TF_Graph *graph, cv::Mat& img, std::vector<face_box> &pnet_boxes, std::vector<face_box> &output_boxes);

void Run_onet(TF_Session *sess, TF_Graph *graph, cv::Mat& img, std::vector<face_box> &rnet_boxes, std::vector<face_box> &output_boxes);

void Copy_one_patch(const cv::Mat& img, face_box &input_box, float *input_data, int input_height, int input_width);

void Generate_bounding_box_tf(const float *confidence_data, int confidence_size, const float *reg_data, float scale, float threshold, int feature_h, int feature_w, std::vector<face_box>&  output, bool transposed);
#endif
