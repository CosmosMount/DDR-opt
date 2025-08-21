#include <ros/ros.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/Pose.h>
#include <std_msgs/Int32MultiArray.h>
#include <vector>
#include <sstream>

// 回调函数，用于接收并打印椅子的访问顺序
void chairOrderCallback(const std_msgs::Int32MultiArray::ConstPtr& msg)
{
    std::stringstream ss;
    for (size_t i = 0; i < msg->data.size(); ++i) {
        ss << msg->data[i] << (i == msg->data.size() - 1 ? "" : ", ");
    }
    ROS_INFO("[Validation Node] Received Chair Order: [ %s ]", ss.str().c_str());
}

// 回调函数，用于接收并打印目标的访问顺序
void targetOrderCallback(const std_msgs::Int32MultiArray::ConstPtr& msg)
{
    std::stringstream ss;
    for (size_t i = 0; i < msg->data.size(); ++i) {
        ss << msg->data[i] << (i == msg->data.size() - 1 ? "" : ", ");
    }
    ROS_INFO("[Validation Node] Received Target Order: [ %s ]", ss.str().c_str());
}

int main(int argc, char** argv)
{
    // 初始化ROS节点
    ros::init(argc, argv, "mission_validation_node");
    ros::NodeHandle nh;

    ROS_INFO("Starting Mission Validation Node...");

    // 创建发布者，用于发布物品和目标的位置
    ros::Publisher items_pub = nh.advertise<geometry_msgs::PoseArray>("/mission/items", 1, true); // Latching publisher
    ros::Publisher targets_pub = nh.advertise<geometry_msgs::PoseArray>("/mission/targets", 1, true); // Latching publisher

    // 创建订阅者，用于接收规划结果
    ros::Subscriber chair_order_sub = nh.subscribe("/mission/results/chair_order", 10, chairOrderCallback);
    ros::Subscriber target_order_sub = nh.subscribe("/mission/results/target_order", 10, targetOrderCallback);

    // 等待一秒，确保所有连接都已建立
    ros::Duration(1.0).sleep();

    // --- 创建并发布物品位置 ---
    geometry_msgs::PoseArray items_msg;
    items_msg.header.stamp = ros::Time::now();
    items_msg.header.frame_id = "world";

    // 添加三个示例物品位置
    geometry_msgs::Pose item1, item2, item3;
    item1.position.x = 0.0; item1.position.y = 0.0; item1.orientation.w = 1.0;
    item2.position.x = 2.0; item2.position.y = -2.0; item2.orientation.w = 1.0;
    item3.position.x = -3.0; item3.position.y = 3.0; item3.orientation.w = 1.0;
    
    items_msg.poses.push_back(item1);
    items_msg.poses.push_back(item2);
    items_msg.poses.push_back(item3);

    items_pub.publish(items_msg);
    ROS_INFO("Published 3 item positions to /mission/items");

    // --- 创建并发布目标位置 ---
    geometry_msgs::PoseArray targets_msg;
    targets_msg.header.stamp = ros::Time::now();
    targets_msg.header.frame_id = "world";

    // 添加三个示例目标位置
    geometry_msgs::Pose target1, target2, target3;
    target1.position.x = -6.0; target1.position.y = -6.0; target1.orientation.w = 1.0;
    target2.position.x = -8.9; target2.position.y = -8.9; target2.orientation.w = 1.0;
    target3.position.x = 5.0; target3.position.y = 5.0; target3.orientation.w = 1.0;

    targets_msg.poses.push_back(target1);
    targets_msg.poses.push_back(target2);
    targets_msg.poses.push_back(target3);

    targets_pub.publish(targets_msg);
    ROS_INFO("Published 3 target positions to /mission/targets");
    
    ROS_INFO("Mission data published. Waiting for results...");

    // 保持节点运行，以接收回调函数的消息
    ros::spin();

    return 0;
}