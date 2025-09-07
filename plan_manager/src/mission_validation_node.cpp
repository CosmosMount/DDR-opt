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

    // 保留offset变量，并通过参数读取
    float sim_2_rviz_offset_1 = 4.5;
    float sim_2_rviz_offset_2 = 2.0;

    // 读取item参数
    double item1_x, item1_y, item2_x, item2_y, item3_x, item3_y, item4_x, item4_y;
    nh.param("item1_x", item1_x, -2.0);
    nh.param("item1_y", item1_y, 4.0);
    nh.param("item2_x", item2_x, -6.5);
    nh.param("item2_y", item2_y, -3.0);
    nh.param("item3_x", item3_x, 0.0);
    nh.param("item3_y", item3_y, -11.0);
    nh.param("item4_x", item4_x, 8.0);
    nh.param("item4_y", item4_y, -5.0);

    geometry_msgs::Pose item1, item2, item3, item4;
    item1.position.x = item1_x + sim_2_rviz_offset_1; item1.position.y = item1_y + sim_2_rviz_offset_2; item1.orientation.w = 1.0;
    item2.position.x = item2_x + sim_2_rviz_offset_1; item2.position.y = item2_y + sim_2_rviz_offset_2; item2.orientation.w = 1.0;
    item3.position.x = item3_x + sim_2_rviz_offset_1; item3.position.y = item3_y + sim_2_rviz_offset_2; item3.orientation.w = 1.0;
    item4.position.x = item4_x + sim_2_rviz_offset_1; item4.position.y = item4_y + sim_2_rviz_offset_2; item4.orientation.w = 1.0;

    items_msg.poses.push_back(item1);
    items_msg.poses.push_back(item2);
    items_msg.poses.push_back(item3);
    items_msg.poses.push_back(item4);

    items_pub.publish(items_msg);
    ROS_INFO("Published 4 item positions to /mission/items");


    // --- 创建并发布目标位置 ---
    geometry_msgs::PoseArray targets_msg;
    targets_msg.header.stamp = ros::Time::now();
    targets_msg.header.frame_id = "world";

    // 读取target参数
    double target1_x, target1_y, target2_x, target2_y, target3_x, target3_y, target4_x, target4_y;
    nh.param("target1_x", target1_x, -1.0);
    nh.param("target1_y", target1_y, -3.5);
    nh.param("target2_x", target2_x, 2.5);
    nh.param("target2_y", target2_y, -4.5);
    nh.param("target3_x", target3_x, -1.0);
    nh.param("target3_y", target3_y, -5.5);
    nh.param("target4_x", target4_x, 2.5);
    nh.param("target4_y", target4_y, -2.5);

    geometry_msgs::Pose target1, target2, target3, target4;
    target1.position.x = target1_x + sim_2_rviz_offset_1; target1.position.y = target1_y + sim_2_rviz_offset_2; target1.orientation.w = 1.0;
    target2.position.x = target2_x + sim_2_rviz_offset_1; target2.position.y = target2_y + sim_2_rviz_offset_2; target2.orientation.w = 1.0;
    target3.position.x = target3_x + sim_2_rviz_offset_1; target3.position.y = target3_y + sim_2_rviz_offset_2; target3.orientation.w = 1.0;
    target4.position.x = target4_x + sim_2_rviz_offset_1; target4.position.y = target4_y + sim_2_rviz_offset_2; target4.orientation.w = 1.0;

    targets_msg.poses.push_back(target1);
    targets_msg.poses.push_back(target2);
    targets_msg.poses.push_back(target3);
    targets_msg.poses.push_back(target4);

    targets_pub.publish(targets_msg);
    ROS_INFO("Published 4 target positions to /mission/targets");
    
    ROS_INFO("Mission data published. Waiting for results...");

    // 保持节点运行，以接收回调函数的消息
    ros::spin();

    return 0;
}