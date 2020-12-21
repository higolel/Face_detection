#include "tensorflow_mtcnn.h"

// --------------------------------------------------------
/// \概要:	计算图像金字塔
///
/// \参数:	image_height
/// \参数:	image_width
/// \参数:	min_size
/// \参数:	scale_scale_factor
/// \参数:	img_win_list
// --------------------------------------------------------
void  Compute_pyramid_list(int image_height, int image_width, int min_size, float scale_factor,std::vector<scale_window> &img_win_list)
{
	double m = 12.0 / min_size;
	int min_layer = std::min(image_height, image_width) * m;
	int factor_count = 0;
	double scale = 0.0;

	while(min_layer >= 12)
	{
		scale = m * std::pow(scale_factor, factor_count);
		min_layer *= scale_factor;

		scale_window img_win;
		img_win.h = std::ceil(image_height * scale);
		img_win.w = std::ceil(image_width * scale);
		img_win.scale = scale;
		img_win_list.push_back(img_win);

		factor_count++;
	}
}

// --------------------------------------------------------
/// \概要:	虚拟解除定位器 根据TF_NewTensor函数要求不写任何内容
///
/// \参数:	data
/// \参数:	len
/// \参数:	arg
// --------------------------------------------------------
static void dummy_deallocator(void *data, size_t len, void *arg)
{
}

// --------------------------------------------------------
/// \概要:	运行pnet网络
///
/// \参数:	sess
/// \参数:	graph
/// \参数:	img
/// \参数:	win_list
/// \参数:	box_list
// --------------------------------------------------------
void Run_pnet(TF_Session *sess, TF_Graph *graph, cv::Mat &img, scale_window &win_list, std::vector<face_box> &box_list)
{
	cv::Mat  resized;
	int scale_h = win_list.h;
	int scale_w = win_list.w;
	float scale = win_list.scale;
	float pnet_threshold = 0.6;

	cv::resize(img, resized, cv::Size(scale_w, scale_h),0,0);

	/* tensorflow related*/
	TF_Status *status = TF_NewStatus();

	std::vector<TF_Output> input_names;
	std::vector<TF_Tensor*> input_values;

	TF_Operation* input_name = TF_GraphOperationByName(graph, "pnet/input");

	input_names.push_back({input_name, 0});

	const int64_t dim[4] = {1, scale_h, scale_w, 3};

	// 图片转张量
	TF_Tensor *input_tensor = TF_NewTensor(TF_FLOAT, dim, 4, resized.ptr(), sizeof(float) * scale_w * scale_h * 3, dummy_deallocator, nullptr);

	input_values.push_back(input_tensor);

	std::vector<TF_Output> output_names;

	TF_Operation *output_name = TF_GraphOperationByName(graph, "pnet/conv4-2/BiasAdd");
	output_names.push_back({output_name, 0});

	output_name = TF_GraphOperationByName(graph, "pnet/prob1");
	output_names.push_back({output_name, 0});

	std::vector<TF_Tensor*> output_values(output_names.size(), nullptr);

	TF_SessionRun(sess, nullptr, input_names.data(), input_values.data(), input_names.size(), output_names.data(), output_values.data(), output_names.size(), nullptr, 0, nullptr, status);

	assert(TF_GetCode(status) == TF_OK);

	// 取回前向传播结果
	const float *conf_data = (const float *)TF_TensorData(output_values[1]);
	const float *reg_data = (const float *)TF_TensorData(output_values[0]);

	// 返回 dim_index 维度中张量的长度。
	int feature_h = TF_Dim(output_values[0],1);
	int feature_w = TF_Dim(output_values[0],2);

	int conf_size = feature_h * feature_w * 2;
//	std::cout << "conf_size: " << conf_size << std::endl;
//	std::cout << "data_size: " << sizeof(*conf_data) << std::endl;

	// 侯选框
	std::vector<face_box> candidate_boxes;

	Generate_bounding_box_tf(conf_data, conf_size, reg_data, scale, pnet_threshold, feature_h, feature_w, candidate_boxes, true);

	Nms_boxes(candidate_boxes, 0.5, NMS_UNION, box_list);

	TF_DeleteStatus(status);
	TF_DeleteTensor(output_values[0]);
	TF_DeleteTensor(output_values[1]);
	TF_DeleteTensor(input_tensor);

}


// --------------------------------------------------------
/// \概要:	生成边界框
///
/// \参数:	confidence_data
/// \参数:	confidence_size
/// \参数:	reg_data
/// \参数:	resize_scale
/// \参数:	pnet_threshold
/// \参数:	feature_h
/// \参数:	feature_w
/// \参数:	output
/// \参数:	transposed
// --------------------------------------------------------
void Generate_bounding_box_tf(const float *confidence_data, int confidence_size, const float *reg_data, float scale, float threshold, int feature_h, int feature_w, std::vector<face_box>&  output, bool transposed)
{

	int stride = 2;
	int cellSize = 12;

	int img_h= feature_h;
	int img_w = feature_w;

//	std::cout << "img_h: " << img_h << "img_w: " << img_w << std::endl;
//	std::cout << "pnet_threshold: " << threshold << std::endl;
//	std::cout << "confidence_data[0]: " << confidence_data[0] << std::endl;

	for(int y = 0; y < img_h; y++)
	{
		for(int x = 0; x < img_w; x++)
		{
			int line_size = img_w * 2;

			float score = confidence_data[line_size * y + 2 * x + 1];

			if(score>= threshold)
			{
				float top_x = (int)((x * stride + 1) / scale);
				float top_y = (int)((y * stride + 1) / scale);
				float bottom_x = (int)((x * stride + cellSize) / scale);
				float bottom_y = (int)((y * stride + cellSize) / scale);

				face_box box;

				box.x0 = top_x;
				box.y0 = top_y;
				box.x1 = bottom_x;
				box.y1 = bottom_y;
				box.score = score;

				int c_offset = (img_w * 4) * y + 4 * x;

				if(transposed)
				{
					box.regress[1] = reg_data[c_offset];
					box.regress[0] = reg_data[c_offset + 1];
					box.regress[3] = reg_data[c_offset + 2];
					box.regress[2] = reg_data[c_offset + 3];
				}
				else
				{
					box.regress[0] = reg_data[c_offset];
					box.regress[1] = reg_data[c_offset + 1];
					box.regress[2] = reg_data[c_offset + 2];
					box.regress[3] = reg_data[c_offset + 3];
				}

				output.push_back(box);
			}
		}
	}
//	std::cout << "output size: " << output.size() << std::endl;
}

// --------------------------------------------------------
/// \概要:	极大值抑制
///
/// \参数:	input
/// \参数:	threshold
/// \参数:	type
/// \参数:	output
// --------------------------------------------------------
void Nms_boxes(std::vector<face_box> &input, float threshold, int type, std::vector<face_box> &output)
{
	// 按照score 降序排列
	std::sort(input.begin(), input.end(),
			[](const face_box &a, const face_box &b) {
			return a.score > b.score;
			});

	int box_num = input.size();

	std::vector<int> merged(box_num, 0);

	for(int i = 0; i < box_num; i++)
	{
		if(merged[i])
			continue;

		output.push_back(input[i]);

		float h_0 = input[i].y1 - input[i].y0 + 1;
		float w_0 = input[i].x1 - input[i].x0 + 1;

		float area_0 = h_0 * w_0;


		for(int j = i + 1; j < box_num; j++)
		{
			if(merged[j])
				continue;

			float inner_x0 = std::max(input[i].x0, input[j].x0);
			float inner_y0 = std::max(input[i].y0, input[j].y0);

			float inner_x1 = std::min(input[i].x1, input[j].x1);
			float inner_y1 = std::min(input[i].y1, input[j].y1);

			float inner_h = inner_y1 - inner_y0 + 1;
			float inner_w = inner_x1 - inner_x0 + 1;

			if(inner_h <= 0 || inner_w <= 0)
				continue;

			float inner_area = inner_h * inner_w;

			float h_1 = input[j].y1 - input[j].y0 + 1;
			float w_1 = input[j].x1 - input[j].x0 + 1;

			float area_1 = h_1 * w_1;

			float score;

			if(type == NMS_UNION)
			{
				score = inner_area / (area_0 + area_1 - inner_area);
			}
			else
			{
				score = inner_area / std::min(area_0, area_1);
			}

			if(score > threshold)
				merged[j] = 1;
		}
	}
}
// --------------------------------------------------------
/// \概要:	处理生成的候选框
///
/// \参数:	input
/// \参数:	img_h
/// \参数:	img_w
/// \参数:	rects
// --------------------------------------------------------
void Process_boxes(std::vector<face_box> &input, int img_h, int img_w, std::vector<face_box> &rects)
{
	Nms_boxes(input, 0.7, NMS_UNION, rects);
	Regress_boxes(rects);
	Square_boxes(rects);
	Padding(img_h, img_w, rects);
}
// --------------------------------------------------------
/// \概要:	回归框
///
/// \参数:	rects
// --------------------------------------------------------
void Regress_boxes(std::vector<face_box> &rects)
{
	for(unsigned int i=0;i<rects.size();i++)
	{
		face_box &box=rects[i];

		float h = box.y1 - box.y0 + 1;
		float w = box.x1 - box.x0 + 1;

		box.x0 = box.x0 + w * box.regress[0];
		box.y0 = box.y0 + h * box.regress[1];
		box.x1 = box.x1 + w * box.regress[2];
		box.y1 = box.y1 + h * box.regress[3];
	}
}

// --------------------------------------------------------
/// \概要:	设置框尺寸，转换正方形尺寸
///
/// \参数:	rects
// --------------------------------------------------------
void Square_boxes(std::vector<face_box> &rects)
{
	for(unsigned int i = 0; i < rects.size(); i++)
	{
		float h = rects[i].y1 - rects[i].y0 + 1;
		float w = rects[i].x1 - rects[i].x0 + 1;

		float max_length = std::max(h, w);

		rects[i].x0 = rects[i].x0 + (w - max_length) * 0.5;
		rects[i].y0 = rects[i].y0 + (h - max_length) * 0.5;
		rects[i].x1 = rects[i].x0 + max_length - 1;
		rects[i].y1 = rects[i].y0 + max_length - 1;
	}
}

// --------------------------------------------------------
/// \概要:	填充
///
/// \参数:	img_h
/// \参数:	img_w
/// \参数:	rects
// --------------------------------------------------------
void Padding(int img_h, int img_w, std::vector<face_box> &rects)
{
	for(unsigned int i = 0; i < rects.size(); i++)
	{
		rects[i].px0 = std::max(rects[i].x0, 1.0f);
		rects[i].py0 = std::max(rects[i].y0, 1.0f);
		rects[i].px1 = std::min(rects[i].x1, (float)img_w);
		rects[i].py1 = std::min(rects[i].y1, (float)img_h);
	}
}

// --------------------------------------------------------
/// \概要:	运行rnet网络
///
/// \参数:	sess
/// \参数:	graph
/// \参数:	img
/// \参数:	pnet_boxes
/// \参数:	output_boxes
// --------------------------------------------------------
void Run_rnet(TF_Session *sess, TF_Graph *graph, cv::Mat& img, std::vector<face_box> &pnet_boxes, std::vector<face_box> &output_boxes)
{
	int batch = pnet_boxes.size();
	int input_channel = 3, input_height = 24, input_width = 24;
	float rnet_threshold = 0.7;

	// 准备输入图像数据
	int input_size = batch * input_height * input_width * input_channel;
	std::vector<float> input_buffer(input_size);
	float *input_data = input_buffer.data();

	for(int i = 0; i < batch; i++)
	{
		int patch_size = input_width * input_height * input_channel;
		Copy_one_patch(img, pnet_boxes[i], input_data, input_height, input_width);
		input_data += patch_size;
	}

	// tensorflow
	TF_Status *status = TF_NewStatus();

	std::vector<TF_Output> input_names;
	std::vector<TF_Tensor*> input_values;

	TF_Operation *input_name = TF_GraphOperationByName(graph, "rnet/input");
	input_names.push_back({input_name, 0});

	const int64_t dim[4] = {batch, input_height, input_width, input_channel};
	TF_Tensor *input_tensor = TF_NewTensor(TF_FLOAT, dim, 4, input_buffer.data(), sizeof(float) * input_size, dummy_deallocator, nullptr);
	input_values.push_back(input_tensor);

	std::vector<TF_Output> output_names;

	TF_Operation *output_name = TF_GraphOperationByName(graph, "rnet/conv5-2/conv5-2");
	output_names.push_back({output_name, 0});

	output_name = TF_GraphOperationByName(graph, "rnet/prob1");
	output_names.push_back({output_name, 0});

	std::vector<TF_Tensor*> output_values(output_names.size(), nullptr);

	TF_SessionRun(sess, nullptr, input_names.data(), input_values.data(), input_names.size(), output_names.data(), output_values.data(), output_names.size(), nullptr, 0, nullptr, status);

	assert(TF_GetCode(status) == TF_OK);

	// 取回前向传播结果
	const float *conf_data = (const float *)TF_TensorData(output_values[1]);
	const float *reg_data = (const float *)TF_TensorData(output_values[0]);

	for(int i = 0; i < batch; i++)
	{
		if(conf_data[1] > rnet_threshold)
		{
			face_box output_box;

			face_box &input_box = pnet_boxes[i];

			output_box.x0 = input_box.x0;
			output_box.y0 = input_box.y0;
			output_box.x1 = input_box.x1;
			output_box.y1 = input_box.y1;

			output_box.score = *(conf_data + 1);

			output_box.regress[0] = reg_data[1];
			output_box.regress[1] = reg_data[0];
			output_box.regress[2] = reg_data[3];
			output_box.regress[3] = reg_data[2];

			output_boxes.push_back(output_box);
		}
		conf_data += 2;
		reg_data += 4;
	}

	TF_DeleteStatus(status);
	TF_DeleteTensor(output_values[0]);
	TF_DeleteTensor(output_values[1]);
	TF_DeleteTensor(input_tensor);
}

// --------------------------------------------------------
/// \概要:	拷贝一个片段
///
/// \参数:	img
/// \参数:	input_box
/// \参数:	input_data
/// \参数:	input_height
/// \参数:	input_width
// --------------------------------------------------------
void Copy_one_patch(const cv::Mat& img, face_box &input_box, float *input_data, int input_height, int input_width)
{
	cv::Mat resized(input_height, input_width, CV_32FC3, input_data);

	cv::Mat chop_img = img(cv::Range(input_box.py0, input_box.py1), cv::Range(input_box.px0, input_box.px1));

	int pad_top = std::abs(input_box.py0 - input_box.y0);
	int pad_left = std::abs(input_box.px0 - input_box.x0);
	int pad_bottom = std::abs(input_box.py1 - input_box.y1);
	int pad_right = std::abs(input_box.px1 - input_box.x1);

	cv::copyMakeBorder(chop_img, chop_img, pad_top, pad_bottom, pad_left, pad_right, cv::BORDER_CONSTANT, cv::Scalar(0));

	cv::resize(chop_img, resized, cv::Size(input_width, input_height), 0, 0);
}

// --------------------------------------------------------
/// \概要:	运行onet网络
///
/// \参数:	sess
/// \参数:	graph
/// \参数:	working_img
/// \参数:	rnet_boxes
/// \参数:	total_onet_boxes
// --------------------------------------------------------
void Run_onet(TF_Session *sess, TF_Graph *graph, cv::Mat& img, std::vector<face_box> &rnet_boxes, std::vector<face_box> &output_boxes)
{
	int batch = rnet_boxes.size();
	int input_channel = 3, input_height = 48, input_width = 48;

	float onet_threshold = 0.9;
//	float onet_threshold = 0.7;

	// 准备输入图像数据
	int input_size = batch * input_height * input_width * input_channel;

	std::vector<float> input_buffer(input_size);

	float *input_data = input_buffer.data();

	for(int i = 0; i < batch; i++)
	{
		int patch_size = input_width * input_height * input_channel;
		Copy_one_patch(img, rnet_boxes[i], input_data, input_height, input_width);
		input_data += patch_size;
	}

	// tensorflow
	TF_Status *status= TF_NewStatus();

	std::vector<TF_Output> input_names;
	std::vector<TF_Tensor*> input_values;

	TF_Operation *input_name = TF_GraphOperationByName(graph, "onet/input");
	input_names.push_back({input_name, 0});

	const int64_t dim[4] = {batch, input_height, input_width, input_channel};

	TF_Tensor *input_tensor = TF_NewTensor(TF_FLOAT, dim, 4, input_buffer.data(), sizeof(float) * input_size, dummy_deallocator, nullptr);

	input_values.push_back(input_tensor);
	std::vector<TF_Output> output_names;

	TF_Operation *output_name = TF_GraphOperationByName(graph, "onet/conv6-2/conv6-2");
	output_names.push_back({output_name, 0});

	output_name = TF_GraphOperationByName(graph, "onet/conv6-3/conv6-3");
	output_names.push_back({output_name, 0});

	output_name = TF_GraphOperationByName(graph, "onet/prob1");
	output_names.push_back({output_name, 0});

	std::vector<TF_Tensor*> output_values(output_names.size(), nullptr);

	TF_SessionRun(sess, nullptr, input_names.data(), input_values.data(), input_names.size(), output_names.data(), output_values.data(), output_names.size(), nullptr, 0, nullptr, status);

	assert(TF_GetCode(status) == TF_OK);

	// 取回前向传播结果
	const float *conf_data = (const float *)TF_TensorData(output_values[2]);
	const float *reg_data = (const float *)TF_TensorData(output_values[0]);
	const float *points_data = (const float *)TF_TensorData(output_values[1]);

	for(int i = 0; i < batch; i++)
	{
		if(conf_data[1] > onet_threshold)
		{
			face_box output_box;

			face_box &input_box = rnet_boxes[i];

			output_box.x0 = input_box.x0;
			output_box.y0 = input_box.y0;
			output_box.x1 = input_box.x1;
			output_box.y1 = input_box.y1;

			output_box.score = conf_data[1];

			output_box.regress[0] = reg_data[1];
			output_box.regress[1] = reg_data[0];
			output_box.regress[2] = reg_data[3];
			output_box.regress[3] = reg_data[2];

			/*Note: switched x,y points value too..*/
			for (int j = 0; j<5; j++){
				output_box.landmark.x[j] = *(points_data + j + 5);
				output_box.landmark.y[j] = *(points_data + j);
			}

			output_boxes.push_back(output_box);
		}
		conf_data += 2;
		reg_data += 4;
		points_data += 10;
	}

	TF_DeleteStatus(status);
	TF_DeleteTensor(output_values[0]);
	TF_DeleteTensor(output_values[1]);
	TF_DeleteTensor(output_values[2]);
	TF_DeleteTensor(input_tensor);
}

