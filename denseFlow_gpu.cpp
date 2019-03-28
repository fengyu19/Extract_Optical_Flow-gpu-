#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/gpu/gpu.hpp"

#include <stdio.h>
#include <iostream>
#include <string>
#include <algorithm>
#include <sys/types.h>
#include <vector>
#include <dirent.h>
#include <sys/stat.h>
using namespace std;
using namespace cv;
using namespace cv::gpu;

static void convertFlowToImage(const Mat &flow_x, const Mat &flow_y, Mat &img_x, Mat &img_y,
       double lowerBound, double higherBound) {
	#define CAST(v, L, H) ((v) > (H) ? 255 : (v) < (L) ? 0 : cvRound(255*((v) - (L))/((H)-(L))))
	for (int i = 0; i < flow_x.rows; ++i) {
		for (int j = 0; j < flow_y.cols; ++j) {
			float x = flow_x.at<float>(i,j);
			float y = flow_y.at<float>(i,j);
			img_x.at<uchar>(i,j) = CAST(x, lowerBound, higherBound);
			img_y.at<uchar>(i,j) = CAST(y, lowerBound, higherBound);
		}
	}
	#undef CAST
}

static void drawOptFlowMap(const Mat& flow, Mat& cflowmap, int step,double, const Scalar& color){
    for(int y = 0; y < cflowmap.rows; y += step)
        for(int x = 0; x < cflowmap.cols; x += step)
        {
            const Point2f& fxy = flow.at<Point2f>(y, x);
            line(cflowmap, Point(x,y), Point(cvRound(x+fxy.x), cvRound(y+fxy.y)),
                 color);
            circle(cflowmap, Point(x,y), 2, color, -1);
        }
}

void GetFileNames(string path, vector<string>& filenames)
{
	DIR *pDir;
	struct dirent* ptr;
	if(!(pDir = opendir(path.c_str())))
		return;
	while((ptr = readdir(pDir)) != 0)
	{
		string str = "DIR";
		if(strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0)
		{
			string v_name = ptr->d_name;
			if(v_name.find(str) < v_name.length())
				continue;
			else{
				filenames.push_back(path + ptr->d_name);
			}
		}
	}
	closedir(pDir);
	
}

void PorcessOneVideo(string videoFile, int device_id, int proposed_imageNum, int bound){
	VideoCapture capture(videoFile);
	if(!capture.isOpened()) {
		printf("Could not initialize capturing..\n");
	}

	// true skip steps
	double fps = capture.get(CV_CAP_PROP_FPS);
	int fps_1 = int(fps);
	if(fps_1 == 29)
		fps_1 = 30;

	long nFrame = capture.get(CV_CAP_PROP_FRAME_COUNT);
	int duration = nFrame/fps_1;
	if(duration > 30)
		duration = 30;

	int step = fps_1 * duration/proposed_imageNum;
	cout<<step<<endl;

	// mkdir directory to save the optical flow output image x, y
	string dirName = videoFile + "DIR";
	mkdir(dirName.c_str(), S_IRWXU);

	int frame_num = 0;
	Mat image, prev_image, prev_grey, grey, frame, flow_x, flow_y;
	GpuMat frame_0, frame_1, flow_u, flow_v;

	setDevice(device_id);
	OpticalFlowDual_TVL1_GPU alg_tvl1;
	// we just want to capture #proposed_imageNum of pics
	int k = 0;
	while(true) {
		capture >> frame;
		if(frame.empty())
			break;
		if(frame_num == 0) {
			image.create(frame.size(), CV_8UC3);
			grey.create(frame.size(), CV_8UC1);
			prev_image.create(frame.size(), CV_8UC3);
			prev_grey.create(frame.size(), CV_8UC1);

			frame.copyTo(prev_image);
			cvtColor(prev_image, prev_grey, CV_BGR2GRAY);

			frame_num++;

			int step_t = step;
			while(step_t > 1){
				capture >> frame;
				step_t--;
			}
			continue;
		}
		frame.copyTo(image);
		cvtColor(image, grey, CV_BGR2GRAY);

		frame_0.upload(prev_grey);
		frame_1.upload(grey);

		alg_tvl1(frame_0, frame_1, flow_u, flow_v);

		flow_u.download(flow_x);
		flow_v.download(flow_y);
		
		// output optical flow
		if(k < proposed_imageNum){
		k = k+1;
		Mat imgX(flow_x.size(), CV_8UC1);
		Mat imgY(flow_y.size(), CV_8UC1);
		convertFlowToImage(flow_x, flow_y, imgX, imgY, -bound, bound);
		char tmp[20];
		sprintf(tmp, "_%05d.jpg", int(frame_num));

		imwrite(dirName + '/' +'X' + tmp, imgX);
		imwrite(dirName + '/' +'Y' + tmp, imgY);
		}
		else
			break;
		//cout<<dirName<<tmp<<endl;
		std::swap(prev_grey, grey);
		std::swap(prev_image, image);
		frame_num = frame_num + 1;

		int step_t = step;
		while (step_t > 1) {
			capture >> frame;
			step_t--;
		}
	}
}


int main(int argc, char** argv){
	// IO operation
const char* keys =
		{
			"{ f  | vidFile      | ex2.avi | filename of video }"
			"{ start  | start    | flow_x | filename of flow x component }"
			"{ end  | end    | flow_y | filename of flow x component }"
		//	"{ i  | imgFile      | flow_i | filename of flow image}"
			"{ b  | bound | 15 | specify the maximum of optical flow}"
		//	"{ t  | type | 0 | specify the optical flow algorithm }"
			"{ d  | device_id    | 0  | set gpu id}"
			"{ s  | step  | 1 | specify the step for frame sampling}"
		};

	CommandLineParser cmd(argc, argv, keys);
	string vidFile = cmd.get<string>("vidFile");
	int start = cmd.get<int>("start");
	int end = cmd.get<int>("end");
	//string imgFile = cmd.get<string>("imgFile");
	int bound = cmd.get<int>("bound");
    int type  = cmd.get<int>("type");
    int device_id = cmd.get<int>("device_id");
    int step = cmd.get<int>("step");

	vector<string> file_names;
	sort(file_names.begin(), file_names.end());
	string path = vidFile;

	GetFileNames(path, file_names);
	for(int i = start; i<end; i++)
	{
		cout<<"start: "<<file_names[i]<<endl;
		PorcessOneVideo(file_names[i], device_id, step, bound);
		
	}
	return 0;
}
