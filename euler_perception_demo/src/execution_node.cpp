#include <ros/ros.h>
#include <moveit/move_group_interface/move_group_interface.h>
#include <std_srvs/Trigger.h>
#include <euler_perception_demo/LocalizePart.h>

class Executor
{
public:

  Executor(const ros::NodeHandle& nh,
           const std::string& group)
    : nh_(nh)
    , move_group_(group)
  {
    client_ = nh_.serviceClient<euler_perception_demo::LocalizePart>("localize_part");
    server_ = nh_.advertiseService("pick", &Executor::callback, this);
  }

  bool callback(std_srvs::TriggerRequest& req,
                std_srvs::TriggerResponse& res)
  {
    const static std::string cloud_topic = "cloud";
    sensor_msgs::PointCloud2ConstPtr cloud_msg = ros::topic::waitForMessage<sensor_msgs::PointCloud2>(cloud_topic, ros::Duration(3.0));

    if(!cloud_msg)
    {
      res.success = false;
      res.message = "Failed to get point cloud from topic '" + cloud_topic + "'";
      return true;
    }

    euler_perception_demo::LocalizePart srv;
    srv.request.cloud = *cloud_msg;

    if(!client_.call(srv))
    {
      res.success = false;
      res.message = "Failed to localize part";
    }
    else
    {
      move_group_.setPoseTarget(srv.response.pose);

      moveit::planning_interface::MoveItErrorCode code;

      moveit::planning_interface::MoveGroupInterface::Plan plan;
      code = move_group_.plan(plan);
      if(code.val != code.SUCCESS)
      {
        res.success = false;
        res.message = "MoveIt planning failed";
        return true;
      }
      else
      {
        code = move_group_.execute(plan);

        if(code.val != code.SUCCESS)
        {
          res.success = false;
          res.message = "MoveIt execution failed";
          return true;
        }
      }
    }

    res.success = true;
    res.message = "Success";

    return true;
  }

private:

  ros::NodeHandle nh_;

  moveit::planning_interface::MoveGroupInterface move_group_;

  ros::ServiceClient client_;

  ros::ServiceServer server_;
};

int main(int argc, char **argv)
{
  ros::init(argc, argv, "execution_node");
  ros::NodeHandle pnh, nh;

  std::string group;
  pnh.param<std::string>("planning_group", group, "manipulator");

  Executor executor(nh, group);

  ros::spin();

  return 0;
}
