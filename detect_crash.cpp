#include <iostream>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <vector>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <string>
#include <opencv2/nonfree/features2d.hpp>

using namespace std;
using namespace boost::program_options;
using namespace boost::filesystem;
using namespace cv;

namespace fs = boost::filesystem;

int	frames2skip, frameDiffGaussMask, damageDetectMaskSize, frameDmgDialMask, fastHessianThs, minNrOfTrackedFeatures, minNrOfAcceptedFeatures, featurePointDistClusterThs,
	ransacProjThreshold, ransacProjThs_1_to, try2MergeXclusters, try2MergeBeforeAfterSteps, try2MergeChangeBase;

float	frameDiffAvg, frameDiffThs, windowHorizontal, windowVertical, damageSensitivity, verificationLevels, verifDamageMaskDilate, featurePointDistThs, 
	featurePointAngeleMedian, featurePointAngeleThs, displayFrameDifference, displayDamages, displayAllFeatures, displayFeatures, displayVerifiedDamages, exitWhenDone;

void extract_videos(vector<boost::filesystem::path>& videos, fs::path full_path)
{
	// videos path
	//fs::path full_path(fs::current_path() / "videos");
	fs::recursive_directory_iterator it(full_path), endit;

	// extracting valid videos from the "videos" directory
	vector<string> valid_video_extensions{".mp4", ".avi", ".mov", ".wmv", ".MP4", ".AVI", ".MOV", ".WMV"};

	while (it != endit)
	{

		if (fs::is_regular_file(*it) && (find(begin(valid_video_extensions), end(valid_video_extensions), it->path().extension()) != end(valid_video_extensions)))
			videos.push_back(it->path());
		++it;
	}

	if (videos.empty())
        {
                cout << "No videos found!" << endl;
                exit(1);
        }
}

bool keep_frame(Mat previousFrame, Mat currentFrame, int max_pix, int& stopAt, float& max_pix_fin, float& max_pix_avg)
{
        Mat grayPreviousFrame, grayCurrentFrame, imgDifference;
        cvtColor(previousFrame, grayPreviousFrame, CV_BGR2GRAY);
        cvtColor(currentFrame, grayCurrentFrame, CV_BGR2GRAY);
        GaussianBlur(grayPreviousFrame, grayPreviousFrame, Size(frameDiffGaussMask, frameDiffGaussMask), 0);
        GaussianBlur(grayCurrentFrame, grayCurrentFrame, Size(frameDiffGaussMask, frameDiffGaussMask), 0);
        absdiff(grayPreviousFrame, grayCurrentFrame, imgDifference);
        float avg_count, temp_avg;
        avg_count = sum(imgDifference)[0] / max_pix;
        temp_avg = avg_count + max_pix_avg;
        max_pix_fin = temp_avg / stopAt;
        if (max_pix_fin*frameDiffAvg < avg_count)
        {
                max_pix_avg = temp_avg;
                stopAt++;
                return true;
        }
        else
	{
		return false;
	}

}

void find_damage_mask(Mat currentFrame, int frame_height, int frame_width, int maxCol, int minCol, int CD1, int CD2, int minRow, 
			int maxRow, Mat frameDmgDialMaskElement, Mat& dmgMaskDamage, int& dmgSum)
{
	Mat imgFrameRGB2 = currentFrame.clone();
	Mat imgFrameHSV;
	cvtColor(imgFrameRGB2, imgFrameHSV, CV_RGB2HSV);

	dmgSum = 0;
	for (int i = minRow; i < maxRow; i++)
	{
		for (int j = minCol; j < maxCol; j++)
		{
			float subVal = (imgFrameHSV.at<Vec3b>(i, j)[2]);
			for (int v = (i - CD1); v <= (i + CD1); v++)
			{
				for (int c = (j - CD2); c <= (j + CD2); c++)
				{
					if (abs((imgFrameHSV.at<Vec3b>(v,c)[2]) - subVal) > damageSensitivity)
					{
						//cout << damageSensitivity << endl;
						dmgMaskDamage.at<uchar>(i, j) = 1;
						dmgSum++;
						//cout << "girdi" << endl;
						goto stop;
					}
				}
			}
		}
		stop:;
	}

	if (frameDmgDialMask > 0) dilate(dmgMaskDamage, dmgMaskDamage, frameDmgDialMaskElement);
}

void find_the_features(Mat previousFrame, int minRow, int maxRow, int minCol, int maxCol, Mat& Tform, Mat& InvTform, Mat currentFrame, float maxDiststance, 
			vector<DMatch>& matches)
{
	// previous frame
	Mat img_objectRGB1 = previousFrame.clone();
	Mat img_objectHSV1;
	cvtColor(img_objectRGB1, img_objectHSV1, CV_RGB2HSV);
	vector<Mat>chHSVout(3);
	split(img_objectHSV1, chHSVout);
	Mat img_frame1 = chHSVout[2].clone();
	equalizeHist(img_frame1, img_frame1);
	FastFeatureDetector detector(fastHessianThs);
	vector<KeyPoint> keypoints_frame1, keypoints_frame11;
	detector.detect(img_frame1, keypoints_frame1);

	int length = keypoints_frame1.size();
	for (int num = 0; num < length; num++)
	{
		Point2f temp = keypoints_frame1[num].pt;
		if (temp.y > minRow && temp.y < maxRow)
			if (temp.x > minCol && temp.x < maxCol)
				keypoints_frame11.push_back(keypoints_frame1[num]);
	}

	SurfDescriptorExtractor extractor;
	Mat descriptors_frame1;
	extractor.compute(img_frame1, keypoints_frame11, descriptors_frame1);
	Tform = Mat::eye(3, 3, CV_64F);
	InvTform = Mat::eye(3, 3, CV_64F);

	// current frame
	Mat img_objectRGB2 = currentFrame.clone();
	Mat img_objectHSV2;
	cvtColor(img_objectRGB2, img_objectHSV2, CV_RGB2HSV);
	vector<Mat> chHSVout2(3);
	split(img_objectHSV2, chHSVout2);
	Mat img_frame2 = chHSVout2[2].clone();
	equalizeHist(img_frame2, img_frame2);
	vector<KeyPoint> keypoints_frame2, keypoints_frame22;
	detector.detect(img_frame2, keypoints_frame2);

	length = keypoints_frame2.size();
	for (int num = 0; num < length; num++)
	{
		Point2f temp = keypoints_frame2[num].pt;
		if (temp.y > minRow && temp.y < maxRow)
			if (temp.x > minCol && temp.x < maxCol)
				keypoints_frame22.push_back(keypoints_frame2[num]);
	}

	Mat descriptors_frame2;
	extractor.compute(img_frame2, keypoints_frame22, descriptors_frame2);

	// match the features
	FlannBasedMatcher matcher;
	vector<DMatch> matchesFrame12;
	matcher.match(descriptors_frame1, descriptors_frame2, matchesFrame12);

	length = matchesFrame12.size();
	float dist;
	vector<float> trueDistRaw;
	vector<float> trueDistSortedRaw;
	vector<float> vectorDist;
	vector<Point2f> frame1_obj;
	vector<Point2f> frame2_scene;
	for (int num = 0; num < length; num++)
	{
		Point2f P_end = keypoints_frame11[matchesFrame12[num].queryIdx].pt;
		Point2f P_start = keypoints_frame22[matchesFrame12[num].trainIdx].pt;
		dist = sqrt(pow((P_end.x - P_start.x), 2) + pow((P_end.y - P_start.y), 2));
		if (dist < maxDiststance)
		{
			trueDistRaw.push_back(dist);
			trueDistSortedRaw.push_back(dist);
			matches.push_back(matchesFrame12[num]);
			vectorDist.push_back(matchesFrame12[num].distance);
			frame1_obj.push_back(keypoints_frame11[matchesFrame12[num].queryIdx].pt);
			frame2_scene.push_back(keypoints_frame22[matchesFrame12[num].trainIdx].pt);
		}
	}
}

void process_video(cv::VideoCapture capVideo, fs::path full_path, fs::path videoName, string timestamp)
{
	// REDUCE VIDEO FRAMES
	string vidReduced = full_path.string() + "/" + videoName.stem().string() + "_" + timestamp + ".avi";

	int ex = static_cast<int>(capVideo.get(CV_CAP_PROP_FOURCC));
        char EXT[] = {(char)(ex & 0XFF) , (char)((ex & 0XFF00) >> 8),(char)((ex & 0XFF0000) >> 16),(char)((ex & 0XFF000000) >> 24), 0};

        int frame_width = (int)capVideo.get(CV_CAP_PROP_FRAME_WIDTH);
        int frame_height = (int)capVideo.get(CV_CAP_PROP_FRAME_HEIGHT);
	int max_pix = frame_width * frame_height;
        Size S = Size(frame_width, frame_height);

	cv::VideoWriter capVideoGood;
        capVideoGood.open(vidReduced, ex, capVideo.get(CV_CAP_PROP_FPS), S, true);

        if (!capVideoGood.isOpened())
        {
        	cout << "!!! Output video could not be opened" << std::endl;
                return;
        }

	if (frames2skip < 1) frames2skip = 1;
	int numberOfFrames = capVideo.get(CV_CAP_PROP_FRAME_COUNT);
	int noFrame = 1;
	vector<Mat> frameHeap;
	Mat previousFrame, currentFrame;
	frameHeap.push_back(previousFrame);
	frameHeap.push_back(currentFrame);
	capVideo.read(frameHeap[0]);
	frameHeap[1] = frameHeap[0].clone();

	// keep_frame values
	int stopAt = 1;
	float max_pix_avg = 0;
	float max_pix_fin;

	// find_damage_mask VALUES
	int d_damageUp = frame_height * (1 - windowVertical) / 2;
	int d_damageDown = frame_height * (1 + windowVertical) / 2;
	int d_damageLeft = frame_width * (1 - windowHorizontal) / 2;
	int d_damageRight = frame_width * (1 + windowHorizontal) / 2;
	damageSensitivity *= 255;
	float rowC = 0.6;
	float colC = 0.6;
	int minRowCenter = frame_height * (0.5 - rowC / 2);
	int maxRowCenter = frame_height * (0.5 + rowC / 2);
	int minColCenter = frame_width * (0.5 - colC / 2);
	int maxColCenter = frame_width * (0.5 + colC / 2);
	int CD1 = (abs(damageDetectMaskSize) - 1) / 2;
	int CD2 = (abs(damageDetectMaskSize) - 1) / 2;
	int minRow = CD1 + d_damageUp;
	int maxRow = d_damageDown - CD1 - 1;
	int minCol = CD2 + d_damageLeft;
	int maxCol = d_damageRight - CD2 - 1;
	if (minRow < CD1) minRow = CD1;
	if (maxRow > frame_height - CD1 - 1) maxRow = frame_height - CD1 - 1;
	if (minCol < CD2) minCol = CD2;
	if (maxCol > frame_width - CD2 - 1) maxCol = frame_width - CD2 - 1;
	Mat wrapLimitMask = Mat(frame_height, frame_width, CV_8U, Scalar::all(0));
	int rmove = (maxRow - minRow) * 0.25;
	int cmove = (maxCol - minCol) * 0.25;
	int minRow2 = minRow + rmove;
	int maxRow2 = maxRow - rmove;
	int minCol2 = minCol + cmove;
	int maxCol2 = maxCol - cmove;
	wrapLimitMask(Rect_<int>(minCol2, minRow2, (maxCol2 - minCol2), (maxRow2 - minRow2))) = 1;
	int dmgMaskSum;
	Mat frameDmgDialMaskElement = getStructuringElement(MORPH_ELLIPSE, Size(2 * frameDmgDialMask + 1, 2 * frameDmgDialMask + 1), 
					Point(frameDmgDialMask, frameDmgDialMask));

	// find_the_features VALUES
	int transCounter;
	float maxDiststance;
	if (frame_height >= frame_width) maxDiststance = frame_height * featurePointDistThs;
	else maxDiststance = frame_width * featurePointDistThs;
	double angleBinVal[360] = {};
	for (int i = 360, j = 0; i > 0; i--, j++) angleBinVal[j] = float(i - 180.0);

	while (noFrame < numberOfFrames)
	{
		if ((noFrame-1) % frames2skip != 0)
                {
                	noFrame++;
                        capVideo.read(frameHeap[1]);
                        continue;
                }
		if (keep_frame(frameHeap[0], frameHeap[1], max_pix, stopAt, max_pix_fin, max_pix_avg))
		{
			Mat dmgMaskDamage(frame_height, frame_width, CV_8U, Scalar::all(0));
			find_damage_mask(frameHeap[1], frame_height, frame_width, maxCol, minCol, CD1, CD2, minRow, maxRow, 
						frameDmgDialMaskElement, dmgMaskDamage, dmgMaskSum);
			Mat Tform, InvTform;
			vector<DMatch> matches;
			find_the_features(frameHeap[0], minRow, maxRow, minCol, maxCol, Tform, InvTform, frameHeap[1], 
						maxDiststance, matches);
			//capVideoGood.write(frameHeap[1]);
		}
		frameHeap[0] = frameHeap[1].clone();
		capVideo.read(frameHeap[1]);
		noFrame++;
	}

}

void process_videos(vector<boost::filesystem::path> videos, fs::path full_path)
{

	chrono::milliseconds sec = chrono::duration_cast<chrono::seconds>(chrono::system_clock::now().time_since_epoch());
        string timestamp = std::to_string(sec.count());

	int length = videos.size();
	for (int videoNum = 0; videoNum < length; videoNum++)
	{
		try
		{
			cv::VideoCapture capVideo;
			capVideo.open(videos[videoNum].string());
			process_video(capVideo, full_path, videos[videoNum], timestamp);
		}
		catch (exception& e)
		{
			cout << "\nError reading video file" << endl;
			cerr << e.what() << "\n";
			continue;
		}

	}
}


int main(int argc, char* argv[])
{

	chrono::milliseconds sec = chrono::duration_cast<chrono::seconds>(chrono::system_clock::now().time_since_epoch());
        string timestamp = std::to_string(sec.count());

	// Parameters are given by the user as arguments. All parameters have default value.
	options_description desc("Allowed options");
	desc.add_options()
	("frames2skip", value<int>(&frames2skip)->default_value(2))
	("frameDiffGaussMask", value<int>(&frameDiffGaussMask)->default_value(5))
	("frameDiffAvg", value<float>(&frameDiffAvg)->default_value(0.5))
	("frameDiffThs", value<float>(&frameDiffThs)->default_value(15.0))
	("frameDmgDialMask", value<int>(&frameDmgDialMask)->default_value(1))
	("windowHorizontal", value<float>(&windowHorizontal)->default_value(0.7))
	("windowVertical", value<float>(&windowVertical)->default_value(0.7))
	("damageSensitivity", value<float>(&damageSensitivity)->default_value(0.1))
	("damageDetectMaskSize", value<int>(&damageDetectMaskSize)->default_value(5))
	("verificationLevels", value<float>(&verificationLevels)->default_value(3.0))
	("verifDamageMaskDilate", value<float>(&verifDamageMaskDilate)->default_value(1.0))
	("fastHessianThs", value<int>(&fastHessianThs)->default_value(10))
	("featurePointDistThs", value<float>(&featurePointDistThs)->default_value(0.25))
	("minNrOfTrackedFeatures", value<int>(&minNrOfTrackedFeatures)->default_value(5))
	("minNrOfAcceptedFeatures", value<int>(&minNrOfAcceptedFeatures)->default_value(10))
	("featurePointDistClusterThs", value<int>(&featurePointDistClusterThs)->default_value(0))
	("featurePointAngeleMedian", value<float>(&featurePointAngeleMedian)->default_value(5.0))
	("featurePointAngeleThs", value<float>(&featurePointAngeleThs)->default_value(20.0))
	("ransacProjThreshold", value<int>(&ransacProjThreshold)->default_value(5))
	("ransacProjThs_1_to", value<int>(&ransacProjThs_1_to)->default_value(15))
	("try2MergeXclusters", value<int>(&try2MergeXclusters)->default_value(5))
	("try2MergeBeforeAfterSteps", value<int>(&try2MergeBeforeAfterSteps)->default_value(7))
	("try2MergeChangeBase", value<int>(&try2MergeChangeBase)->default_value(5))
	("displayFrameDifference", value<float>(&displayFrameDifference)->default_value(0.0))
	("displayDamages", value<float>(&displayDamages)->default_value(0.0))
	("displayAllFeatures", value<float>(&displayAllFeatures)->default_value(0.0))
	("displayFeatures", value<float>(&displayFeatures)->default_value(0.0))
	("displayVerifiedDamages", value<float>(&displayVerifiedDamages)->default_value(0.0))
	("exitWhenDone", value<float>(&exitWhenDone)->default_value(1.0));

	variables_map opts;
	store(parse_command_line(argc, argv, desc), opts);

	frames2skip = opts["frames2skip"].as<int>();
        frameDiffGaussMask = opts["frameDiffGaussMask"].as<int>();
        frameDiffAvg = opts["frameDiffAvg"].as<float>();
        windowHorizontal = opts["windowHorizontal"].as<float>();
        windowVertical = opts["windowVertical"].as<float>();
        damageSensitivity = opts["damageSensitivity"].as<float>();
	damageDetectMaskSize = opts["damageDetectMaskSize"].as<int>();
	frameDmgDialMask = opts["frameDmgDialMask"].as<int>();
	fastHessianThs = opts["fastHessianThs"].as<int>();
	featurePointDistThs = opts["featurePointDistThs"].as<float>();
	minNrOfTrackedFeatures = opts["minNrOfTrackedFeatures"].as<int>();
	minNrOfAcceptedFeatures = opts["minNrOfAcceptedFeatures"].as<int>();
	featurePointDistClusterThs = opts["featurePointDistClusterThs"].as<int>();
	featurePointAngeleMedian = opts["featurePointAngeleMedian"].as<float>();
	featurePointAngeleThs = opts["featurePointAngeleThs"].as<float>();
	ransacProjThreshold = opts["ransacProjThreshold"].as<int>();
	ransacProjThs_1_to = opts["ransacProjThs_1_to"].as<int>();
	try2MergeXclusters = opts["try2MergeXclusters"].as<int>();
	try2MergeBeforeAfterSteps = opts["try2MergeBeforeAfterSteps"].as<int>();
	try2MergeChangeBase = opts["try2MergeChangeBase"].as<int>();

	vector<boost::filesystem::path> videos;
	fs::path full_path(fs::current_path() / "videos");
	extract_videos(videos, full_path);
	process_videos(videos, full_path);

}
