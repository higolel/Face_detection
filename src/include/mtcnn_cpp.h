#ifndef __MTCNN_CPP_H__
#define __MTCNN_CPP_H__

#include "tensorflow_mtcnn.h"
#include "base64_mat.h"

std::string image_, model_pb_, out_image_, scripts_, video_, cam_;
std::string video_sub_ = "cam/realmonitor?channel=1&subtype=0";

image_transport::Publisher pub_image_raw_;
ros::Publisher pub_face_pic_message_;
queue<cv::Mat> que;

int Pub_image_raw();
TF_Session *Load_graph(const char *model_fname, TF_Graph **p_graph);
int Load_file(const std::string &fname, std::vector<char> &buf);
void Mtcnn_detect(TF_Session *sess, TF_Graph *graph, cv::Mat& img, std::vector<face_box> &face_list);
cv::Mat Image_intercept(cv::Mat frame);
cv::Mat Rect_face_pic(cv::Mat cp_frame, face_box box);
void Pub_face_pic_message(cv::Mat cp_frame, cv::Mat frame_face);
void sleep_ms(unsigned int millisec);
//void Python_face_attribute();

void Display();
void Receive();

#endif
