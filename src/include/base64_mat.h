#ifndef __BASE64_MAT_H__
#define __BASE64_MAT_H__
#include "tensorflow_mtcnn.h"

std::string base64Decode(const char *Data, int DataByte);
std::string base64Encode(const unsigned char *Data, int DataByte);
std::string Mat_to_Base64(cv::Mat img, std::string imgType);
cv::Mat Base_to_Mat(std::string &base64_data);

#endif
