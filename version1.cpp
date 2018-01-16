/*************************************************************************
    > File Name: main.cpp
    > Author: QYQ
    > Mail: qiuyeqiang@hotmail.com
    > Created Time: 2017年11月22日 星期三 13时06分35秒
 ************************************************************************/

#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdio.h>

#include <pcl/io/pcd_io.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/extract_indices.h>

#include <pcl/segmentation/supervoxel_clustering.h>

// The segmentation class this example is for
#include <pcl/segmentation/lccp_segmentation.h>

using namespace cv;
typedef  pcl::PointXYZRGBA PointT;
typedef pcl::LCCPSegmentation<PointT>::SupervoxelAdjacencyList SuperVoxelAdjacencyList;

cv::Mat org,dst,img;
std::vector<uint32_t> color;
pcl::PointCloud<PointT>::Ptr bkcloud(new pcl::PointCloud<PointT>);
pcl::PointCloud<PointT>::Ptr curcloud(new pcl::PointCloud<PointT>);
pcl::PointCloud<PointT>::Ptr precloud(new pcl::PointCloud<PointT>);
pcl::visualization::CloudViewer vi("3D");


void colorInitiate()
{
    color.resize(6);
    uint32_t rgb = ((uint32_t)(255) << 16 | (uint32_t)(0) << 8 | (uint32_t)(0));
    color[0]=rgb;
    rgb = ((uint32_t)(255) << 16 | (uint32_t)(255) << 8 | (uint32_t)(0));
    color[1]=rgb;
    rgb = ((uint32_t)(255) << 16 | (uint32_t)(0) << 8 | (uint32_t)(255));
    color[2]=rgb;
    rgb = ((uint32_t)(0) << 16 | (uint32_t)(255) << 8 | (uint32_t)(0));
    color[3]=rgb;
    rgb = ((uint32_t)(0) << 16 | (uint32_t)(255) << 8 | (uint32_t)(255));
    color[4]=rgb;
    rgb = ((uint32_t)(0) << 16 | (uint32_t)(0) << 8 | (uint32_t)(255));
    color[5]=rgb;

}

void lccp(pcl::PointCloud<PointT>::Ptr input_cloud_ptr)
{
    ///  Default values of parameters before parsing
     // Supervoxel Stuff
     float voxel_resolution = 0.0075f;
     float seed_resolution = 0.03f;
     float color_importance = 0.0f;
     float spatial_importance = 1.0f;
     float normal_importance = 4.0f;
     bool use_single_cam_transform = false;
     bool use_supervoxel_refinement = false;

     // LCCPSegmentation Stuff
     float concavity_tolerance_threshold = 10;
     float smoothness_threshold = 0.1;
     uint32_t min_segment_size = 0;
     bool use_extended_convexity = false;
     bool use_sanity_criterion = false;

     float normals_scale = seed_resolution / 2.0;

     // Segmentation Stuff
     unsigned int k_factor = 0;
     if (use_extended_convexity)
       k_factor = 1;

     /// Preparation of Input: Supervoxel Oversegmentation

     pcl::SupervoxelClustering<PointT> super (voxel_resolution, seed_resolution);
     super.setUseSingleCameraTransform (use_single_cam_transform);
     super.setInputCloud (input_cloud_ptr);
     super.setColorImportance (color_importance);
     super.setSpatialImportance (spatial_importance);
     super.setNormalImportance (normal_importance);
     std::map<uint32_t, pcl::Supervoxel<PointT>::Ptr> supervoxel_clusters;

     PCL_INFO ("Extracting supervoxels\n");
     super.extract (supervoxel_clusters);

     if (use_supervoxel_refinement)
     {
       PCL_INFO ("Refining supervoxels\n");
       super.refineSupervoxels (2, supervoxel_clusters);
     }
     std::stringstream temp;
     temp << "  Nr. Supervoxels: " << supervoxel_clusters.size () << "\n";
     PCL_INFO (temp.str ().c_str ());

     PCL_INFO ("Getting supervoxel adjacency\n");
     std::multimap<uint32_t, uint32_t> supervoxel_adjacency;
     super.getSupervoxelAdjacency (supervoxel_adjacency);

     /// Get the cloud of supervoxel centroid with normals and the colored cloud with supervoxel coloring (this is used for visulization)
     pcl::PointCloud<pcl::PointNormal>::Ptr sv_centroid_normal_cloud = pcl::SupervoxelClustering<PointT>::makeSupervoxelNormalCloud (supervoxel_clusters);

     /// The Main Step: Perform LCCPSegmentation

     PCL_INFO ("Starting Segmentation\n");
     pcl::LCCPSegmentation<PointT> lccp;
     lccp.setConcavityToleranceThreshold (concavity_tolerance_threshold);
     lccp.setSanityCheck (use_sanity_criterion);
     lccp.setSmoothnessCheck (true, voxel_resolution, seed_resolution, smoothness_threshold);
     lccp.setKFactor (k_factor);
     lccp.setInputSupervoxels (supervoxel_clusters, supervoxel_adjacency);
     lccp.setMinSegmentSize (min_segment_size);
     lccp.segment ();

     PCL_INFO ("Interpolation voxel cloud -> input cloud and relabeling\n");
     pcl::PointCloud<pcl::PointXYZL>::Ptr sv_labeled_cloud = super.getLabeledCloud ();
     pcl::PointCloud<pcl::PointXYZL>::Ptr lccp_labeled_cloud = sv_labeled_cloud->makeShared ();
     lccp.relabelCloud (*lccp_labeled_cloud);
     SuperVoxelAdjacencyList sv_adjacency_list;
     lccp.getSVAdjacencyList (sv_adjacency_list);  // Needed for visualization

     for(size_t i = 0;i<lccp_labeled_cloud->size();i++)
     {
         int j = (lccp_labeled_cloud->at(i).label)%6 ;
         uint32_t rgb = color[j];
         input_cloud_ptr->at(i).rgb = *reinterpret_cast<float *>(&rgb);
     }
}

void on_mouse(int event,int x,int y,int flags,void *ustc)//event鼠标事件代号，x,y鼠标坐标，flags拖拽和键盘操作的代号
{
    static Point pre_pt(-1,-1);//初始坐标
    static Point cur_pt(-1,-1);//实时坐标
    char temp[16];
    if (event == CV_EVENT_LBUTTONDOWN)//左键按下，读取初始坐标，并在图像上该点处划圆
    {
        org.copyTo(img);//将原始图片复制到img中
        sprintf(temp,"(%d,%d)",x,y);
        pre_pt = Point(x-45,y-45);
        cur_pt = Point(x+45,y+45);
        cv::rectangle(img,pre_pt,cur_pt,cv::Scalar(0,255,0,0),1,8,0);

        pcl::PointCloud<PointT>::Ptr result(new pcl::PointCloud<PointT>);
        for(int i=x-45;i<=x+45;i++)
            for(int j=y-45;j<=y+45;j++)
            {
                if(std::fabs(bkcloud->at(i,j).z - curcloud->at(i,j).z) > 0.01)
                {
                    result->push_back(curcloud->at(i,j));
                }
            }
        pcl::io::savePCDFile("result.pcd",*result);
        lccp(result);
        vi.showCloud(result);

        imshow("img",img);
    }
}
int main()
{
    org = imread("Large_scene_segmentation_data/experiment1/pic/99.png");
    org.copyTo(img);

    pcl::io::loadPCDFile("Large_scene_segmentation_data/experiment1/pcd/background.pcd",*bkcloud);
    pcl::io::loadPCDFile("Large_scene_segmentation_data/experiment1/pcd/99.pcd",*curcloud);

    for(int i=0;i<bkcloud->size();i++)
        {
            if(std::fabs(bkcloud->at(i).z - curcloud->at(i).z) > 0.01)
            {
                precloud->push_back(curcloud->at(i));
            }
        }

    colorInitiate();
    vi.showCloud(precloud);
    namedWindow("img");//定义一个img窗口
    while(!vi.wasStopped()){
        setMouseCallback("img",on_mouse,0);//调用回调函数
        imshow("img",img);
        cv::waitKey(0);
    }
    return 0;
}

