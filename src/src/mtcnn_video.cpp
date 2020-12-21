#include "mtcnn_cpp.h"

// --------------------------------------------------------
/// \概要:	msleep
///
/// \参数:	image
// --------------------------------------------------------
void sleep_ms(unsigned int millisec)
{
	struct timeval tval;
	tval.tv_sec = millisec / 1000;
	tval.tv_usec = (millisec * 1000) % 1000000;
	select(0, NULL, NULL, NULL, &tval);
}

// --------------------------------------------------------
/// \概要:	图像截取函数
///
/// \参数:	middle_frame
///
/// \返回:	cv::Mat
// --------------------------------------------------------
cv::Mat Image_intercept(cv::Mat frame)
{
	int weight = frame.cols, height = frame.rows;
	cv::Rect rect(weight / 9, height / 4, weight / 9 * 7, height / 2);
	cv::Mat rect_frame = frame(rect);

	return rect_frame;
}

#if 0
// --------------------------------------------------------
/// \概要:	调用python 人脸属性函数
// --------------------------------------------------------
void Python_face_attribute()
{
	//	Py_SetPythonHome((wchar_t *)L"/home/lel/anaconda2/");
	Py_Initialize();
	if(!Py_IsInitialized())
	{
		return;
	}

	//	PyRun_SimpleString("from deepface import DeepFace");
	PyRun_SimpleString("import sys");
	PyRun_SimpleString("import string");
	std::string strPath = std::string("sys.path.append('") + scripts_ + std::string("')");
	std::cout << strPath << std::endl;
	PyRun_SimpleString(strPath.c_str());

	PyObject *pModule = PyImport_ImportModule("face_attribute");
	PyObject *pFunc = PyObject_GetAttrString(pModule, "FaceAttribute");
	PyObject *pArgsT = PyTuple_New(1);
	PyTuple_SetItem(pArgsT, 0, Py_BuildValue("s", out_image_.c_str()));

	PyObject *pReturn = PyObject_CallObject(pFunc, pArgsT);

	int ok = 0;
	PyArg_Parse(pReturn, "i", &ok);
	ROS_INFO("ok = %d", ok);

	//	Py_Finalize();
}
#endif

// --------------------------------------------------------
/// \概要:	切割人脸图片
///
/// \参数:	cp_frame
/// \参数:	box
///
/// \返回:	cv::Mat
// --------------------------------------------------------
cv::Mat Rect_face_pic(cv::Mat cp_frame, face_box box)
{
	static int couter = 0;

	int weight = cp_frame.cols, height = cp_frame.rows;
	int face_weight = int(box.x1 - box.x0), face_height = int(box.y1 - box.y0);

	int first_point_x0 = std::max(int(box.x0 + (weight / 9) + (face_weight / 2) - 413 / 2), 0);
	int first_point_y0 = std::max(int(box.y0 + (height / 4) + (face_height / 2) - 626 / 2), 0);
	int first_point_x1 = std::min(int(box.x1 + (weight / 9) - (face_weight / 2) + 413 / 2), cp_frame.cols);
	int first_point_y1 = std::min(int(box.y1 + (height / 4) - (face_height / 2) + 626 / 2), cp_frame.rows);

	cv::Rect rect(first_point_x0, first_point_y0, (first_point_x1 - first_point_x0), (first_point_y1 - first_point_y0));
	cv::Mat rect_frame = cv::Mat(cp_frame, rect);

#if 0
	cv::imwrite(out_image_ + std::to_string(couter++) + std::string(".jpg"), rect_frame);
	cv::imwrite(out_image_ + std::to_string(couter) + std::string(".jpg"), rect_frame);
	cv::imwrite(out_image_ + std::string("_bk") + std::to_string(couter++) + std::string(".jpg"), cp_frame);
	if(couter >= 10)
		couter = 0;
	cv::imwrite(out_image_, rect_frame);
#endif

	return rect_frame;
}

// --------------------------------------------------------
/// \概要:	接收线程
// --------------------------------------------------------
void Receive()
{
	cv::VideoCapture cap;
	cap.open(video_);
	if(!cap.isOpened())
		std::cerr << "open video failed!" << std::endl;
	else
		std::cout << "open video success!" << std::endl;

	cv::Mat frame;
	bool isSuccess = true;
	while(1)
	{
		isSuccess = cap.read(frame);
		if(!isSuccess)
		{
			std::cerr << "video ends!" << endl;
			break;
		}

		que.push(frame);
		if(que.size() > 1)
			// 注意 pop和front的区别
			que.pop();
		else
			sleep_ms(750);
	}
}

// --------------------------------------------------------
/// \概要:	发送人脸图片信息
///
/// \参数:	cp_frame
/// \参数:	box
// --------------------------------------------------------
void Pub_face_pic_message(cv::Mat cp_frame, cv::Mat frame_face)
{
	face_plate_msgs::Face_pic face_pic_msg;
	face_pic_msg.vin = "as00030";
	face_pic_msg.deviceId = "030人脸";
	face_pic_msg.pictureType = 1;
	face_pic_msg.sex = 1;
	face_pic_msg.age = 25;
	face_pic_msg.facialExpression = 0;
	face_pic_msg.race = 1;
	face_pic_msg.hat = 1;
	face_pic_msg.bmask = 1;
	face_pic_msg.eyeglass = 1;
	struct timeval tv;
	gettimeofday(&tv, NULL);
	face_pic_msg.capTime = (tv.tv_sec * 1000 + tv.tv_usec / 1000);
//	std::cout << "capTime" << face_pic_msg.capTime << std::endl;

	face_pic_msg.facePicture = Mat_to_Base64(frame_face, std::string("jpg"));
	face_pic_msg.faceScenePicture = Mat_to_Base64(cp_frame, "jpg");

	pub_face_pic_message_.publish(face_pic_msg);
}

// --------------------------------------------------------
/// \概要:	显示线程
// --------------------------------------------------------
void Display()
{
	TF_Graph *graph = TF_NewGraph();
	TF_Status *status = TF_NewStatus();

	TF_Session *sess = Load_graph(model_pb_.c_str(), &graph);

	if(sess == nullptr)
		cerr << "sess is failed!" << endl;
	cv::Mat frame;

	while(1)
	{
		sleep_ms(1500);
		if(!que.empty())
		{
			frame = que.front();
			que.pop();
		}
		else
			continue;

		vector<face_box> face_info;

		int weight = frame.cols, height = frame.rows;

		cv::Mat frame_intercept = Image_intercept(frame);
		Mtcnn_detect(sess, graph, frame_intercept, face_info);
	//	Mtcnn_detect(sess, graph, frame, face_info);
		for(unsigned int i = 0; i < face_info.size(); i++)
		{
			face_box &box = face_info[i];
			cv::Mat cp_frame = frame;
			cv::Mat frame_face = Rect_face_pic(cp_frame, box);
		//	Python_face_attribute();

			Pub_face_pic_message(cp_frame, frame_face);

#if 0
			// 画人脸框
			cv::rectangle(frame, cv::Point(box.x0 + weight / 9, box.y0 + height / 4), cv::Point(box.x1 + weight / 9, box.y1 + height / 4), cv::Scalar(0, 255, 0), 2);
		//	cv::rectangle(frame, cv::Point(box.x0, box.y0), cv::Point(box.x1, box.y1), cv::Scalar(0, 255, 0), 2);

			// 画人脸标记点
			for(int y = 0; y < 5; y++)
				cv::circle(frame, cv::Point(box.landmark.x[y] + weight / 9, box.landmark.y[y] + height / 4), 1, cv::Scalar(0, 0, 255), 3);
			//	cv::circle(frame, cv::Point(box.landmark.x[y], box.landmark.y[y]), 1, cv::Scalar(0, 0, 255), 3);
#endif
		}

		cv::rectangle(frame, cv::Point(weight / 9, height / 4), cv::Point(weight / 9 * 8, height / 4 * 3), cv::Scalar(0, 0, 255), 2);

#if 0
		sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", frame).toImageMsg();
		pub_image_raw_.publish(msg);
#endif
	}

	TF_CloseSession(sess, status);
	TF_DeleteSession(sess, status);
	TF_DeleteGraph(graph);
	TF_DeleteStatus(status);
}

// --------------------------------------------------------
/// \概要:	从model_pb加载文件到model_buf 里
///
/// \参数:	model_pb
/// \参数:	model_buf
///
/// \返回:	int
// --------------------------------------------------------
int Load_file(const std::string &fname, std::vector<char> &buf)
{
	std::ifstream fs(fname, std::ios::binary | std::ios::in);

	if(!fs.good())
	{
		std::cerr<<fname<<" does not exist"<<std::endl;
		return -1;
	}

	fs.seekg(0, std::ios::end);
	int fsize=fs.tellg();

	fs.seekg(0, std::ios::beg);
	buf.resize(fsize);
	fs.read(buf.data(),fsize);

	fs.close();

	return 0;
}

// --------------------------------------------------------
/// \概要:	加载表graph
///
/// \参数:	model_fname
/// \参数:	p_graph
///
/// \返回:	sess
// --------------------------------------------------------
TF_Session *Load_graph(const char *model_fname, TF_Graph **p_graph)
{
	TF_Status *status = TF_NewStatus();

	TF_Graph* graph = TF_NewGraph();

	std::vector<char> model_buf;

	Load_file(model_fname, model_buf);

	// TF_Buffer 是一个结构体，查看c_api.h文件, 同理，以下的函数也可查看c_api.h
	TF_Buffer graph_def = {model_buf.data(), model_buf.size(), nullptr};

	TF_ImportGraphDefOptions *import_opts = TF_NewImportGraphDefOptions();
	TF_ImportGraphDefOptionsSetPrefix(import_opts, "");
	TF_GraphImportGraphDef(graph, &graph_def, import_opts, status);

	if(TF_GetCode(status) != TF_OK)
	{
		printf("load graph failed!\n Error: %s\n", TF_Message(status));
		return nullptr;
	}

	TF_SessionOptions* sess_opts = TF_NewSessionOptions();
	TF_Session* session = TF_NewSession(graph, sess_opts, status);
	assert(TF_GetCode(status) == TF_OK);

	TF_DeleteStatus(status);

	*p_graph = graph;

	return session;
}

void Mtcnn_detect(TF_Session *sess, TF_Graph *graph, cv::Mat& img, std::vector<face_box> &face_list)
{
	cv::Mat working_img;

	float alpha = 0.0078125;
	float mean = 255.0 / 2.0;

	/*
	 *	1--bit_depth---比特数---代表8bite,16bites,32bites,64bites---举个例子吧--比如说,如
	 如果你现在创建了一个存储--灰度图片的Mat对象,这个图像的大小为宽100,高100,那么,现在这张
	 灰度图片中有10000个像素点，它每一个像素点在内存空间所占的空间大小是8bite,8位--所以它对
	 应的就是CV_8

	 *	2--S|U|F
	 S--代表---signed int----有符号整形
	 U--代表--unsigned int---无符号整形
	 F--代表--float----------单精度浮点型
	 *	3--C<number_of_channels>----代表---一张图片的通道数,比如:
	 1--灰度图片--grayImg----是--单通道图像
	 2--RGB彩色图像----------是--3通道图像
	 3--带Alph通道的RGB图像--是--4通道图像
	 */
	img.convertTo(working_img, CV_32FC3);

	// 均值归一化
	working_img = (working_img - mean) * alpha;
	// 转置矩阵
	working_img = working_img.t();

	cv::cvtColor(working_img, working_img, cv::COLOR_BGR2RGB);

	int image_height = working_img.rows;
	int image_width = working_img.cols;

	int min_size = 40;
//	int min_size = 20;
	float scale_factor = 0.709;

	std::vector<scale_window> win_list;
	std::vector<face_box> total_pnet_boxes;
	std::vector<face_box> total_rnet_boxes;
	std::vector<face_box> total_onet_boxes;

	// 计算图像金字塔
	Compute_pyramid_list(image_height, image_width, min_size, scale_factor, win_list);

	for(unsigned int i = 0; i < win_list.size(); i++)
	{
		std::vector<face_box> boxes;

		Run_pnet(sess, graph, working_img, win_list[i], boxes);
		total_pnet_boxes.insert(total_pnet_boxes.end(), boxes.begin(), boxes.end());
	}

	std::vector<face_box> pnet_boxes;
	Process_boxes(total_pnet_boxes, image_height, image_width, pnet_boxes);

	// RNet
	std::vector<face_box> rnet_boxes;
	Run_rnet(sess, graph, working_img, pnet_boxes, total_rnet_boxes);
	Process_boxes(total_rnet_boxes, image_height, image_width, rnet_boxes);

	//ONet
	Run_onet(sess, graph, working_img, rnet_boxes, total_onet_boxes);

	// 计算landmark
	for(unsigned int i = 0; i < total_onet_boxes.size(); i++)
	{
		face_box &box = total_onet_boxes[i];

		float h = box.x1 - box.x0 + 1;
		float w = box.y1 - box.y0 + 1;

		for(int j = 0; j < 5; j++)
		{
			box.landmark.x[j] = box.x0 + w * box.landmark.x[j] - 1;
			box.landmark.y[j] = box.y0 + h * box.landmark.y[j] - 1;
		}
	}

	//Get Final Result
	Regress_boxes(total_onet_boxes);
	Nms_boxes(total_onet_boxes, 0.7, NMS_MIN, face_list);

	//switch x and y, since working_img is transposed
	for(unsigned int i = 0; i < face_list.size(); i++)
	{
		face_box &box = face_list[i];

		std::swap(box.x0, box.y0);
		std::swap(box.x1, box.y1);

		for(int l = 0; l < 5; l++)
			std::swap(box.landmark.x[l], box.landmark.y[l]);
	}
}


int main(int argc, char *argv[])
{
	ros::init(argc, argv, "mtcnn_cpp");
	ros::NodeHandle nh_("~");
	ros::Time time = ros::Time::now();
	ros::Rate loop_rate(10);
	nh_.param("/image", image_, std::string("test.jpg"));
	nh_.param("/model_pb", model_pb_, std::string("./models/mtcnn_frozen_model.pb"));
	nh_.param("/out_image", out_image_, std::string("./new.jpg"));
//	nh_.param("scripts", scripts_, std::string("scripts"));
	nh_.param("video", video_, std::string("test.mp4"));
	nh_.param("cam", cam_, std::string("right_front"));

	video_ = video_ + video_sub_;
//	cout << "video path: " << video_ << endl;
//	cout << "model_pb " << model_pb_ << endl;

#if 0
	image_transport::ImageTransport it(nh_);
	pub_image_raw_ = it.advertise("/camera/image_raw_" + cam_, 1);
#endif
	pub_face_pic_message_ = nh_.advertise<face_plate_msgs::Face_pic>("/face_pic_msg", 1);

	// 是指使用的显存
	TF_Graph *graph = TF_NewGraph();
	TF_Status *status = TF_NewStatus();
	TF_SessionOptions *session_opts = TF_NewSessionOptions();

	// 设置最大显存为当前gpu的三分之一, 该数组通过python 指令获得
//	uint8_t config[16] = {0x32, 0xe, 0x9, 0x1d, 0x5a, 0x64, 0x3b, 0xdf, 0x4f, 0xd5, 0x3f, 0x20, 0x1, 0x2a, 0x1, 0x30};
	// 设置最大显存为当前gpu的四分之一, 该数组通过python 指令获得
	uint8_t config[16] = {0x32, 0xe, 0x9, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0xd0, 0x3f, 0x20, 0x1, 0x2a, 0x1, 0x30};
	TF_SetConfig(session_opts, (void *)config, 16, status);
	TF_Session *sess = TF_NewSession(graph, session_opts, status);

	std::thread thread_1(Receive);
	std::thread thread_2(Display);

	thread_1.join();
	thread_2.join();

	return 0;
}


