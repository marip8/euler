#include <eigen_stl_containers/eigen_stl_vector_container.h>
#include <eigen_conversions/eigen_msg.h>
#include <euler_perception_demo/LocalizePart.h>
#include <geometry_msgs/PoseStamped.h>
#include <pcl/common/pca.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
//#include <pcl/filters/extract_indices.h>
#include <pcl/filters/project_inliers.h>
#include <pcl_ros/point_cloud.h>
//#include <pcl/common/centroid.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/String.h>
#include <std_msgs/Bool.h>
#include <std_srvs/Trigger.h>

struct DemoParams
{
  int min_cluster_size;
  double cluster_tolerance;
  double plane_fit_tolerance;
  double z_offset;
};

class PerceptionDemo
{
public:

  PerceptionDemo(const ros::NodeHandle& nh,
                 const DemoParams& params,
                 const bool debug)
    : nh_(nh)
    , params_(params)
    , debug_(debug)
  {
    server_ = nh_.advertiseService("localize_part", &PerceptionDemo::callback, this);

    if(debug_)
    {
      const static std::string DEBUG_POSE_TOPIC = "debug_pose";
      debug_pub_ = nh_.advertise<geometry_msgs::PoseStamped>(DEBUG_POSE_TOPIC, 1, true);

      const static std::string DEBUG_CLOUD_TOPIC = "debug_cloud";
      debug_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>(DEBUG_CLOUD_TOPIC, 1, true);
    }
  }

private:

  template<typename T>
  bool processCloud(const typename pcl::PointCloud<T>::ConstPtr cloud,
                    const pcl::PointIndicesConstPtr cluster,
                    pcl::PointCloud<T>& out,
                    Eigen::Vector3d& z_axis)
  {
    // Plane fit the data
    pcl::SACSegmentation<T> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(params_.plane_fit_tolerance);
    seg.setInputCloud(cloud);
    seg.setIndices(cluster);

    pcl::ModelCoefficientsPtr coefficients (new pcl::ModelCoefficients);
    pcl::PointIndicesPtr inliers (new pcl::PointIndices ());
    seg.segment(*inliers, *coefficients);

    if(inliers->indices.size() == 0)
    {
      ROS_ERROR_STREAM("Failed to fit plane to cluster");
      return false;
    }

    // Project inliers onto the plane
    pcl::ProjectInliers<T> projector;
    projector.setInputCloud(cloud);
    projector.setIndices(inliers);
    projector.setModelType(pcl::SACMODEL_PLANE);
    projector.setModelCoefficients(coefficients);

    // Save the result into the output cloud
    projector.filter(out);

    // Z-axis computed in plane segmentation step
    z_axis = Eigen::Vector3d(coefficients->values[0], coefficients->values[1], coefficients->values[2]);
    z_axis.normalize();

    return true;
  }

  template<typename T>
  geometry_msgs::Pose computePickPose(const typename pcl::PointCloud<T>::ConstPtr cloud,
                                      const Eigen::Vector3d& z_axis)
  {
    // Get the center and principal axes of the projected, plane-fit cluster
    pcl::PCA<T> pca;
    pca.setInputCloud(cloud);

    Eigen::Matrix3d evecs = pca.getEigenVectors().template cast<double>();

    // Y-axis can be estimated from the 2nd eigenvector
    Eigen::Vector3d y_axis(evecs(0, 1), evecs(1, 1), evecs(2, 1));
    y_axis.normalize();

    // Cross product to obtain X axis
    Eigen::Vector3d x_axis = y_axis.cross(z_axis);

    Eigen::Affine3d pose;
    pose.matrix().col(0).head<3>() = x_axis;
    pose.matrix().col(1).head<3>() = y_axis;
    pose.matrix().col(2).head<3>() = z_axis;
    pose.matrix().col(3) = pca.getMean().template cast<double>();

    // Check that the Z-axis of the pose is aligned with the Z-axis of the camera
    double dp = pose.matrix().col(2).head<3>().dot(Eigen::Vector3d::UnitZ());

    // Flip the pose about the X-axis if it is not aligned
    if(dp < 0)
    {
      pose.rotate(Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX()));
    }

    // Translate the pose along the Z-axis towards the camera
    pose.translate(Eigen::Vector3d(0.0f, 0.0f, -params_.z_offset));

    geometry_msgs::Pose out;
    tf::poseEigenToMsg(pose, out);

    return out;
  }

  bool callback(euler_perception_demo::LocalizePartRequest& req,
                euler_perception_demo::LocalizePartResponse& res)
  {
    auto fail_cb = [this, &res](const std::string& msg)
    {
      ROS_ERROR_STREAM(msg);
      res.success = false;
      res.message = msg;
    };

    // Convert the point cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>());
    pcl::fromROSMsg(req.cloud, *cloud);

    // Create a search tree
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>());
    tree->setInputCloud(cloud);

    // Cluster the data
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(params_.cluster_tolerance);
    ec.setMinClusterSize(params_.min_cluster_size);
    ec.setMaxClusterSize(std::numeric_limits<int>::max());
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud);
    ec.extract(cluster_indices);

    // Make sure there are at least the desired number of clusters
    ROS_INFO_STREAM(cluster_indices.size() << " cluster(s) identified");
    if(cluster_indices.size() < 1)
    {
      fail_cb("No clusters found");
      return true;
    }

    // Grab the largest cluster
    pcl::PointIndicesPtr cluster (new pcl::PointIndices(cluster_indices[0]));

    pcl::PointCloud<pcl::PointXYZ>::Ptr processed_cloud (new pcl::PointCloud<pcl::PointXYZ>());
    Eigen::Vector3d z_axis;
    if(!processCloud<pcl::PointXYZ>(cloud, cluster, *processed_cloud, z_axis))
    {
      fail_cb("Failed to process cluster into pose");
      return true;
    }

    // Show the points that fit the plane
    if(debug_)
    {
      sensor_msgs::PointCloud2 processed_cloud_msg;
      pcl::toROSMsg(*processed_cloud, processed_cloud_msg);
      processed_cloud_msg.header.frame_id = req.cloud.header.frame_id;
      processed_cloud_msg.header.stamp = ros::Time::now();
      debug_cloud_pub_.publish(processed_cloud_msg);
    }

    res.pose.header.frame_id = req.cloud.header.frame_id;
    res.pose.header.stamp = ros::Time::now();
    res.pose.pose = computePickPose<pcl::PointXYZ>(processed_cloud, z_axis);
    res.success = true;
    res.message = "Successfully determined largest cluster";

    if(debug_)
    {
      debug_pub_.publish(res.pose);
    }

    return true;
  }

  ros::NodeHandle nh_;

  ros::ServiceServer server_;

  DemoParams params_;

  bool debug_;

  ros::Publisher debug_pub_;

  ros::Publisher debug_cloud_pub_;
};

int main(int argc, char **argv)
{
  ros::init(argc, argv, "perception_demo");
  ros::NodeHandle nh, pnh("~");

  DemoParams params;
  pnh.param<double>("cluster_tolerance", params.cluster_tolerance, 0.025f);
  pnh.param<double>("plane_fit_tolerance", params.plane_fit_tolerance, 0.010f);
  pnh.param<int>("min_cluster_size", params.min_cluster_size, 100);
  pnh.param<double>("z_offset", params.z_offset, 0.025f);

  bool debug;
  pnh.param<bool>("debug", debug, false);

  PerceptionDemo demo(nh, params, debug);

  ros::spin();

  return 0;
}
