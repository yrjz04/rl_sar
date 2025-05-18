/*
 * @Author: yrjz yrjz04@outlook.com
 * @Date: 2025-05-18 19:49:05
 * @LastEditors: yrjz yrjz04@outlook.com
 * @LastEditTime: 2025-05-18 19:49:09
 * @FilePath: \rl_sar\src\subscribe_pointcloud.cpp
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
/**
 * @file subscribe_pointcloud.cpp
 * @brief Subscribe the pointcloud published from DDS topic
 * @date 2023-11-23
 */

#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/common/time/time_tool.hpp>
#include <unitree/idl/ros2/PointCloud2_.hpp>

#define TOPIC_CLOUD "rt/utlidar/cloud"

using namespace unitree::robot;
using namespace unitree::common;

void Handler(const void *message)
{
  const sensor_msgs::msg::dds_::PointCloud2_ *cloud_msg = (const sensor_msgs::msg::dds_::PointCloud2_ *)message;
  std::cout << "Received a raw cloud here!"
            << "\n\tstamp = " << cloud_msg->header().stamp().sec() << "." << cloud_msg->header().stamp().nanosec()
            << "\n\tframe = " << cloud_msg->header().frame_id()
            << "\n\tpoints number = " << cloud_msg->width()
            << std::endl
            << std::endl;
}

int main(int argc, const char **argv)
{
  if (argc < 2)
  {
    std::cout << "Usage: " << argv[0] << " networkInterface" << std::endl;
    exit(-1);
  }

  ChannelFactory::Instance()->Init(0, argv[1]);

  ChannelSubscriber<sensor_msgs::msg::dds_::PointCloud2_> subscriber(TOPIC_CLOUD);
  subscriber.InitChannel(Handler);

  while (true)
  {
    sleep(10);
  }

  return 0;
}