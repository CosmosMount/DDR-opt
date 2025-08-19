#ifndef _PLAN_MANAGER_HPP_
#define _PLAN_MANAGER_HPP_

// 核心 ROS 头文件
#include <ros/ros.h>

// 依赖的包和消息类型
#include "plan_env/sdf_map.h"
#include "visualizer/visualizer.hpp"
#include "front_end/jps_planner/jps_planner.h"
#include "back_end/optimizer.h"
#include "carstatemsgs/CarState.h"
#include "carstatemsgs/Polynome.h"
#include "geometry_msgs/PoseStamped.h"
#include "geometry_msgs/PoseArray.h"
#include "nav_msgs/Odometry.h"
#include "std_msgs/Bool.h"
#include "visualization_msgs/MarkerArray.h"
#include "tf/tf.h"

// C++ 标准库
#include <thread>
#include <vector>
#include <algorithm>
#include <limits>

// 本地模块
#include "plan_manager/hungarian.hpp"

enum StateMachine{
  INIT,
  IDLE,
  PLANNING,
  REPLAN,
  GOINGTOGOAL,
  EMERGENCY_STOP,
};

class PlanManager
{
private:
    ros::NodeHandle nh_;

    // 核心组件
    std::shared_ptr<SDFmap> sdfmap_;
    std::shared_ptr<Visualizer> visualizer_;
    std::shared_ptr<MSPlanner> msplanner_;
    std::shared_ptr<JPS::JPSPlanner> jps_planner_;

    // 任务与路径点管理
    std::vector<Eigen::Vector3d> chair_positions_;
    std::vector<Eigen::Vector3d> target_positions_;
    std::vector<std::pair<int, int>> chair_to_target_assignments_;
    std::vector<Eigen::Vector3d> ordered_waypoints_;
    int current_waypoint_idx_;

    // ROS 通信
    ros::Subscriber goal_sub_;
    ros::Subscriber current_state_sub_;
    ros::Timer main_thread_timer_;
    ros::Publisher cmd_pub_;
    ros::Publisher mpc_polynome_pub_;
    ros::Publisher emergency_stop_pub_;
    ros::Publisher record_pub_;
    ros::Publisher marker_pub_;

    // 状态变量
    ros::Time current_time_;
    Eigen::Vector3d current_state_XYTheta_;
    Eigen::Vector3d current_state_VAJ_;
    Eigen::Vector3d current_state_OAJ_;
    double plan_start_time_;
    Eigen::Vector3d plan_start_state_XYTheta;
    Eigen::Vector3d plan_start_state_VAJ;
    Eigen::Vector3d plan_start_state_OAJ;
    Eigen::Vector3d goal_state_;
    ros::Time Traj_start_time_;
    double Traj_total_time_;
    ros::Time loop_start_time_;
    bool have_geometry_;
    bool have_goal_;
    bool if_fix_final_;
    Eigen::Vector3d final_state_;
    double replan_time_;
    double max_replan_time_;
    double predicted_traj_start_time_;
    bool enable_tsp_optimization_ = true;  // 是否启用TSP优化
    StateMachine state_machine_ = StateMachine::INIT;

public:
    PlanManager(ros::NodeHandle nh);
    ~PlanManager();

    // 任务规划流程函数
    bool generateRandomPositions();
    void assignChairsToTargets();
    void visualizeChairsAndTargets();
    bool solvePath();
    
    // TSP求解辅助函数
    std::vector<int> TSPSolver(const std::vector<std::vector<double>>& dist_matrix);

    // ROS 回调与主循环
    void MainThread(const ros::TimerEvent& event);
    void GeometryCallback(const nav_msgs::Odometry::ConstPtr &msg);
    void goal_callback(const geometry_msgs::PoseStamped::ConstPtr &msg);
    
    // 辅助函数
    void printStateMachine();
    bool findJPSRoad();
    void MPCPathPub(const double& traj_start_time);
};

PlanManager::PlanManager(ros::NodeHandle nh) : nh_(nh) {
  
  sdfmap_ = std::make_shared<SDFmap>(nh);
  visualizer_ = std::make_shared<Visualizer>(nh);
  msplanner_ = std::make_shared<MSPlanner>(Config(ros::NodeHandle("~")), nh_, sdfmap_);
  jps_planner_ = std::make_shared<JPS::JPSPlanner>(sdfmap_, nh_);

  goal_sub_ = nh_.subscribe<geometry_msgs::PoseStamped>("/move_base_simple/goal", 1, &PlanManager::goal_callback, this);
  current_state_sub_ = nh_.subscribe<nav_msgs::Odometry>("odom", 1, &PlanManager::GeometryCallback, this);
  main_thread_timer_ = nh_.createTimer(ros::Duration(0.001), &PlanManager::MainThread, this);
  
  cmd_pub_ = nh_.advertise<carstatemsgs::CarState>("/simulation/PoseSub", 1);
  emergency_stop_pub_ = nh_.advertise<std_msgs::Bool>("/planner/emergency_stop", 1);
  record_pub_ = nh_.advertise<visualization_msgs::Marker>("/planner/calculator_time", 1);
  mpc_polynome_pub_ = nh_.advertise<carstatemsgs::Polynome>("traj", 1);
  marker_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("/planner/markers", 10, true);

  have_geometry_ = false;
  have_goal_ = false;
  current_waypoint_idx_ = -1;

  nh_.param<bool>("if_fix_final", if_fix_final_, false);
  if(if_fix_final_){
    nh_.param<double>("final_x", final_state_(0), 0.0);
    nh_.param<double>("final_y", final_state_(1), 0.0);
    nh_.param<double>("final_yaw", final_state_(2), 0.0);
  }

  nh_.param<double>("replan_time", replan_time_, 10000.0);
  nh_.param<double>("max_replan_time", max_replan_time_, 1.0);
  nh_.param<bool>("enable_tsp_optimization", enable_tsp_optimization_, true);

  ROS_INFO("Plan Manager initialized. TSP optimization: %s", enable_tsp_optimization_ ? "enabled" : "disabled");

  state_machine_ = StateMachine::IDLE;
  loop_start_time_ = ros::Time::now();
}

PlanManager::~PlanManager() {}

void PlanManager::printStateMachine() {
  if(state_machine_ == INIT) ROS_INFO("state_machine_ == INIT");
  if(state_machine_ == IDLE) ROS_INFO("state_machine_ == IDLE");
  if(state_machine_ == PLANNING) ROS_INFO("state_machine_ == PLANNING");
  if(state_machine_ == REPLAN) ROS_INFO("state_machine_ == REPLAN");
}

void PlanManager::GeometryCallback(const nav_msgs::Odometry::ConstPtr &msg) {
  bool first_time = !have_geometry_;

  have_geometry_ = true;
  current_state_XYTheta_ << msg->pose.pose.position.x, msg->pose.pose.position.y, tf::getYaw(msg->pose.pose.orientation);
  current_state_VAJ_ << 0.0, 0.0, 0.0;
  current_state_OAJ_ << 0.0, 0.0, 0.0;
  current_time_ = msg->header.stamp;

  if (first_time) {
    ROS_INFO("First odometry received. Initializing chair-target mission setup...");
    if (generateRandomPositions()) {
        visualizeChairsAndTargets();
        ROS_INFO("Initial mission setup complete. Send a goal in RViz to start execution.");
    } else {
        ROS_ERROR("Initial mission setup failed. Please check the map and restart.");
    }
  }
}

void PlanManager::goal_callback(const geometry_msgs::PoseStamped::ConstPtr &msg) {
  if (state_machine_ != StateMachine::IDLE) {
    ROS_WARN("Planner is busy. Ignoring new mission start signal.");
    return;
  }
  if (!have_geometry_) {
    ROS_ERROR("No odometry received. Cannot start mission.");
    return;
  }

  ROS_INFO("\n\n--- Received Mission Start Signal ---");
  
  if (!generateRandomPositions()) {
      ROS_ERROR("Mission aborted: Failed to generate all required chair/target positions.");
      visualizeChairsAndTargets(); // Call to clear any old markers
      state_machine_ = StateMachine::IDLE;
      return;
  }
  visualizeChairsAndTargets();
  
  assignChairsToTargets();
  
  if (chair_to_target_assignments_.empty()) {
      ROS_ERROR("Mission aborted: Failed to assign chairs to targets.");
      state_machine_ = StateMachine::IDLE;
      return;
  }

  if (solvePath()) {
    ROS_INFO("TSP solved. Starting multi-point traversal.");
    current_waypoint_idx_ = 0;
    goal_state_ = ordered_waypoints_[current_waypoint_idx_];
    have_goal_ = true;
  } else {
    ROS_ERROR("Failed to solve TSP. Aborting mission.");
    state_machine_ = StateMachine::IDLE;
    have_goal_ = false;
  }
  ROS_INFO("-------------------------------------\n\n");
}

bool PlanManager::generateRandomPositions() {
    srand(time(0)); 
    chair_positions_.clear();
    target_positions_.clear();
    std::vector<Eigen::Vector3d> all_generated_points;

    const double safe_dist_from_obstacle = 0.8;
    const double min_dist_between_points = 1.5;
    const int max_attempts = 200;

    if (!have_geometry_) {
        ROS_ERROR("Cannot generate positions without knowing robot's current location.");
        return false;
    }
    Eigen::Vector3d reference_pos = current_state_XYTheta_;
    ROS_INFO("Using robot's current position (%.1f, %.1f) as reachability reference.", reference_pos.x(), reference_pos.y());

    auto generate_points = [&](std::vector<Eigen::Vector3d>& point_vector, int num_points, const std::string& name) {
        ROS_INFO("Generating %d safe and reachable positions for %s...", num_points, name.c_str());
        for (int i = 0; i < num_points; ++i) {
            int attempts = 0;
            while (attempts < max_attempts) {
                Eigen::Vector3d candidate_pos((rand() % 180 - 90) / 10.0, 
                                              (rand() % 180 - 90) / 10.0,
                                              0.0);
                attempts++;

                if (sdfmap_->getDistanceReal(candidate_pos.head<2>()) < safe_dist_from_obstacle) {
                    continue;
                }

                bool self_collision = false;
                for (const auto& existing_pos : all_generated_points) {
                    if ((candidate_pos - existing_pos).head<2>().norm() < min_dist_between_points) {
                        self_collision = true;
                        break;
                    }
                }
                if (self_collision) {
                    continue;
                }

                if (!jps_planner_->plan(reference_pos, candidate_pos)) {
                    if (attempts % 50 == 0) ROS_WARN("Attempt %d for %s %d: Candidate (%.1f, %.1f) is not reachable.", attempts, name.c_str(), i, candidate_pos.x(), candidate_pos.y());
                    continue;
                }

                point_vector.push_back(candidate_pos);
                all_generated_points.push_back(candidate_pos);
                ROS_INFO("  - Generated %s %d at (%.1f, %.1f)", name.c_str(), i, candidate_pos.x(), candidate_pos.y());
                break; 
            }
            if (attempts >= max_attempts) {
                ROS_ERROR("Failed to generate a safe and reachable position for %s %d after %d attempts.", name.c_str(), i, max_attempts);
            }
        }
    };

    generate_points(chair_positions_, 3, "Chair");
    generate_points(target_positions_, 3, "Target");

    if (chair_positions_.size() != 3 || target_positions_.size() != 3) {
        ROS_ERROR("Failed to generate all required points. Aborting mission setup.");
        chair_positions_.clear();
        target_positions_.clear();
        return false;
    }
    return true;
}

void PlanManager::assignChairsToTargets() {
    int n = chair_positions_.size();
    if (n == 0) {
        ROS_WARN("No chair positions available for assignment. Aborting.");
        return;
    }

    ROS_INFO("Calculating cost matrix for chair-target assignment...");
    Eigen::MatrixXd cost_matrix(n, n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            bool path_found = jps_planner_->plan(chair_positions_[i], target_positions_[j]);
            if (path_found) {
                cost_matrix(i, j) = jps_planner_->getPathLength();
            } else {
                ROS_WARN("Cannot find a path from Chair %d [%.1f, %.1f] to Target %d [%.1f, %.1f]. Setting cost to infinity.", 
                         i, chair_positions_[i].x(), chair_positions_[i].y(), 
                         j, target_positions_[j].x(), target_positions_[j].y());
                cost_matrix(i, j) = std::numeric_limits<double>::max();
            }
        }
    }

    HungarianAlgorithm ha;
    std::vector<int> assignment_vec;
    double min_cost = ha.solve(cost_matrix, assignment_vec);

    chair_to_target_assignments_.clear();
    if (min_cost >= std::numeric_limits<double>::max()) {
        ROS_ERROR("Assignment failed. At least one target is unreachable from all chairs.");
        return;
    }

    ROS_INFO("Optimal assignment found with total estimated cost: %.2f", min_cost);
    for (int i = 0; i < n; ++i) {
        chair_to_target_assignments_.emplace_back(i, assignment_vec[i]);
        ROS_INFO("  - Assign Chair %d to Target %d", i, assignment_vec[i]);
    }
}

void PlanManager::visualizeChairsAndTargets() {
    ROS_INFO("Visualizing markers... Chairs: %zu, Targets: %zu", chair_positions_.size(), target_positions_.size());
    visualization_msgs::MarkerArray marker_array;
    ros::Time now = ros::Time::now();

    // The most robust way to clear and then draw markers is to use a single
    // MarkerArray message. The DELETEALL action will clear all markers previously
    // published by this node, and the subsequent ADD actions will draw the new ones.
    // This is processed atomically by RViz.
    visualization_msgs::Marker clear_marker;
    clear_marker.header.frame_id = "world";
    clear_marker.header.stamp = now;
    clear_marker.action = visualization_msgs::Marker::DELETEALL;
    marker_array.markers.push_back(clear_marker);

    // Add new markers for chairs
    for (size_t i = 0; i < chair_positions_.size(); ++i) {
        visualization_msgs::Marker marker;
        marker.header.frame_id = "world";
        marker.header.stamp = now;
        marker.ns = "chairs";
        marker.id = i;
        marker.type = visualization_msgs::Marker::CUBE;
        marker.action = visualization_msgs::Marker::ADD;
        marker.pose.position.x = chair_positions_[i].x();
        marker.pose.position.y = chair_positions_[i].y();
        marker.pose.position.z = 0.5;
        marker.scale.x = 0.5;
        marker.scale.y = 0.5;
        marker.scale.z = 0.5;
        marker.color.r = 1.0;
        marker.color.g = 0.0;
        marker.color.b = 0.0;
        marker.color.a = 0.8;
        marker.lifetime = ros::Duration(); // Persist until deleted
        marker_array.markers.push_back(marker);
    }

    // Add new markers for targets
    for (size_t i = 0; i < target_positions_.size(); ++i) {
        visualization_msgs::Marker marker;
        marker.header.frame_id = "world";
        marker.header.stamp = now;
        marker.ns = "targets";
        marker.id = i;
        marker.type = visualization_msgs::Marker::SPHERE;
        marker.action = visualization_msgs::Marker::ADD;
        marker.pose.position.x = target_positions_[i].x();
        marker.pose.position.y = target_positions_[i].y();
        marker.pose.position.z = 0.5;
        marker.scale.x = 0.5;
        marker.scale.y = 0.5;
        marker.scale.z = 0.5;
        marker.color.r = 0.0;
        marker.color.g = 1.0;
        marker.color.b = 0.0;
        marker.color.a = 0.8;
        marker.lifetime = ros::Duration(); // Persist until deleted
        marker_array.markers.push_back(marker);
    }

    marker_pub_.publish(marker_array);
    ROS_INFO("Published %zu markers to topic %s.", marker_array.markers.size(), marker_pub_.getTopic().c_str());
}

bool PlanManager::solvePath() {
    if (chair_to_target_assignments_.empty()) {
        ROS_ERROR("No assignments available for path planning.");
        return false;
    }

    if (!enable_tsp_optimization_) {
        // 简单策略：按原始顺序访问椅子-目标对
        ROS_INFO("TSP optimization disabled. Using original assignment order.");
        ordered_waypoints_.clear();
        ordered_waypoints_.push_back(current_state_XYTheta_); // 起始点
        
        for (const auto& assignment : chair_to_target_assignments_) {
            ordered_waypoints_.push_back(chair_positions_[assignment.first]);   // 椅子点
            ordered_waypoints_.push_back(target_positions_[assignment.second]); // 目标点
        }
        
        ROS_INFO("Simple task sequence generated with %zu waypoints.", ordered_waypoints_.size());
        return true;
    }

    // TSP优化策略
    ROS_INFO("Starting TSP optimization for chair visiting order...");

    // 构建椅子位置列表用于TSP优化
    std::vector<Eigen::Vector3d> chair_task_points;
    std::vector<int> chair_indices;
    for (const auto& assignment : chair_to_target_assignments_) {
        chair_task_points.push_back(chair_positions_[assignment.first]);
        chair_indices.push_back(assignment.first);
    }

    // 构建TSP距离矩阵：起始点 + 各个椅子之间的距离
    int n = chair_task_points.size() + 1; // +1 for starting position
    std::vector<std::vector<double>> tsp_dist_matrix(n, std::vector<double>(n, 0.0));
    
    ROS_INFO("Building TSP distance matrix for %d nodes...", n);
    
    // 计算起始点到各椅子的距离
    for (int j = 1; j < n; ++j) {
        bool path_found = jps_planner_->plan(current_state_XYTheta_, chair_task_points[j-1]);
        double len = path_found ? jps_planner_->getPathLength() : std::numeric_limits<double>::max();
        if (len == std::numeric_limits<double>::max()) {
            ROS_ERROR("TSP failed: Cannot find path from start to Chair %d.", chair_indices[j-1]);
            return false;
        }
        tsp_dist_matrix[0][j] = tsp_dist_matrix[j][0] = len;
    }
    
    // 计算椅子之间的距离（考虑椅子->目标->下一个椅子的总距离）
    for (int i = 1; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            // 从椅子i到椅子j的总距离 = 椅子i->目标i + 目标i->椅子j
            int chair_i_idx = chair_indices[i-1];
            int chair_j_idx = chair_indices[j-1];
            
            // 找到对应的目标
            int target_i_idx = -1, target_j_idx = -1;
            for (const auto& assignment : chair_to_target_assignments_) {
                if (assignment.first == chair_i_idx) target_i_idx = assignment.second;
                if (assignment.first == chair_j_idx) target_j_idx = assignment.second;
            }
            
            if (target_i_idx == -1 || target_j_idx == -1) {
                ROS_ERROR("TSP failed: Cannot find target assignments.");
                return false;
            }
            
            // 计算椅子i->目标i的距离
            bool path1_found = jps_planner_->plan(chair_positions_[chair_i_idx], target_positions_[target_i_idx]);
            double len1 = path1_found ? jps_planner_->getPathLength() : std::numeric_limits<double>::max();
            
            // 计算目标i->椅子j的距离
            bool path2_found = jps_planner_->plan(target_positions_[target_i_idx], chair_positions_[chair_j_idx]);
            double len2 = path2_found ? jps_planner_->getPathLength() : std::numeric_limits<double>::max();
            
            double total_len = len1 + len2;
            if (len1 == std::numeric_limits<double>::max() || len2 == std::numeric_limits<double>::max()) {
                total_len = std::numeric_limits<double>::max();
                ROS_WARN("TSP: Cannot find complete path from Chair %d to Chair %d via targets.", chair_i_idx, chair_j_idx);
            }
            
            tsp_dist_matrix[i][j] = tsp_dist_matrix[j][i] = total_len;
        }
    }

    // 使用智能TSP求解器（根据问题规模选择算法）
    std::vector<int> tsp_path;
    tsp_path = TSPSolver(tsp_dist_matrix);

    if (tsp_path.empty() || tsp_path[0] != 0) {
        ROS_ERROR("TSP solver failed to find valid solution.");
        return false;
    }
    
    // 根据TSP结果构建最终的waypoint序列
    ordered_waypoints_.clear();
    ordered_waypoints_.push_back(current_state_XYTheta_); // 起始点
    
    // 按TSP顺序添加椅子-目标对
    for (int i = 1; i < tsp_path.size(); ++i) {
        int chair_task_idx = tsp_path[i] - 1; // -1 because tsp_path[0] is start position
        int chair_idx = chair_indices[chair_task_idx];
        
        // 找到对应的目标
        int target_idx = -1;
        for (const auto& assignment : chair_to_target_assignments_) {
            if (assignment.first == chair_idx) {
                target_idx = assignment.second;
                break;
            }
        }
        
        if (target_idx != -1) {
            ordered_waypoints_.push_back(chair_positions_[chair_idx]);   // 椅子点
            ordered_waypoints_.push_back(target_positions_[target_idx]); // 目标点
            ROS_INFO("  Optimized waypoint pair %d: Chair %d -> Target %d", i, chair_idx, target_idx);
        }
    }

    ROS_INFO("TSP-optimized task sequence generated successfully with %zu waypoints.", ordered_waypoints_.size());
    return true;
}

void PlanManager::MainThread(const ros::TimerEvent& event) {
  if (!have_geometry_ || !have_goal_) return;

  // collision check
  if(have_geometry_){
    if(sdfmap_->getDistanceReal(Eigen::Vector2d(current_state_XYTheta_.x(), current_state_XYTheta_.y())) < 0.0){
      std_msgs::Bool emergency_stop;
      emergency_stop.data = true;
      emergency_stop_pub_.publish(emergency_stop);
      state_machine_ = EMERGENCY_STOP;
      ROS_INFO_STREAM("current_state_XYTheta_: " << current_state_XYTheta_.transpose());
      ROS_INFO_STREAM("Dis: " << sdfmap_->getDistanceReal(Eigen::Vector2d(current_state_XYTheta_.x(), current_state_XYTheta_.y())));
      ROS_ERROR("EMERGENCY_STOP!!! too close to obstacle!!!");
      return;
    }
  }
  
  if (state_machine_ == StateMachine::IDLE || 
      ((state_machine_ == StateMachine::PLANNING || state_machine_ == StateMachine::REPLAN) && 
       (ros::Time::now() - loop_start_time_).toSec() > replan_time_)) {
    
    loop_start_time_ = ros::Time::now();
    double current = loop_start_time_.toSec();
    // start new plan
    if (state_machine_ == StateMachine::IDLE) {
      state_machine_ = StateMachine::PLANNING;
      plan_start_time_ = -1;
      predicted_traj_start_time_ = -1;
      plan_start_state_XYTheta = current_state_XYTheta_;
      plan_start_state_VAJ = current_state_VAJ_;
      plan_start_state_OAJ = current_state_OAJ_;
    } 
    // Use predicted distance for replanning in planning state
    else if (state_machine_ == StateMachine::PLANNING || state_machine_ == StateMachine::REPLAN) {
      
      if (((current_state_XYTheta_ - goal_state_).head(2).squaredNorm() + fmod(fabs((plan_start_state_XYTheta - goal_state_)[2]), 2.0 * M_PI)*0.02 < 1.0) ||
          msplanner_->final_traj_.getTotalDuration() < max_replan_time_) {
        state_machine_ = StateMachine::GOINGTOGOAL;
        return;
      }

      state_machine_ = StateMachine::REPLAN;

      predicted_traj_start_time_ = current + max_replan_time_ - plan_start_time_;
      msplanner_->get_the_predicted_state(predicted_traj_start_time_, plan_start_state_XYTheta, plan_start_state_VAJ, plan_start_state_OAJ);

    } 
    
    ROS_INFO("\033[32;40m \n\n\n\n\n-------------------------------------start new plan------------------------------------------ \033[0m");
    
    visualizer_->finalnodePub(plan_start_state_XYTheta, goal_state_);
    ROS_INFO("init_state_: %.10f  %.10f  %.10f", plan_start_state_XYTheta(0), plan_start_state_XYTheta(1), plan_start_state_XYTheta(2));
    ROS_INFO("goal_state_: %.10f  %.10f  %.10f", goal_state_(0), goal_state_(1), goal_state_(2));
    std::cout<<"<arg name=\"start_x_\" value=\""<< plan_start_state_XYTheta(0) <<"\"/>"<<std::endl;
    std::cout<<"<arg name=\"start_y_\" value=\""<< plan_start_state_XYTheta(1) <<"\"/>"<<std::endl;
    std::cout<<"<arg name=\"start_yaw_\" value=\""<< plan_start_state_XYTheta(2) <<"\"/>"<<std::endl;
    std::cout<<"<arg name=\"final_x_\" value=\""<< goal_state_(0) <<"\"/>"<<std::endl;
    std::cout<<"<arg name=\"final_y_\" value=\""<< goal_state_(1) <<"\"/>"<<std::endl;
    std::cout<<"<arg name=\"final_yaw_\" value=\""<< goal_state_(2) <<"\"/>"<<std::endl;

    std::cout<<"plan_start_state_VAJ: "<<plan_start_state_VAJ.transpose()<<std::endl;
    std::cout<<"plan_start_state_OAJ: "<<plan_start_state_OAJ.transpose()<<std::endl;

    ROS_INFO("<arg name=\"start_x_\" value=\"%f\"/>", plan_start_state_XYTheta(0));
    ROS_INFO("<arg name=\"start_y_\" value=\"%f\"/>", plan_start_state_XYTheta(1));
    ROS_INFO("<arg name=\"start_yaw_\" value=\"%f\"/>", plan_start_state_XYTheta(2));
    ROS_INFO("<arg name=\"final_x_\" value=\"%f\"/>", goal_state_(0));
    ROS_INFO("<arg name=\"final_y_\" value=\"%f\"/>", goal_state_(1));
    ROS_INFO("<arg name=\"final_yaw_\" value=\"%f\"/>", goal_state_(2));

    ROS_INFO_STREAM("plan_start_state_VAJ: " << plan_start_state_VAJ.transpose());
    ROS_INFO_STREAM("plan_start_state_OAJ: " << plan_start_state_OAJ.transpose());

    // front end
    ros::Time astar_start_time = ros::Time::now();
    if(!findJPSRoad()){
      state_machine_ = EMERGENCY_STOP;
      ROS_ERROR("EMERGENCY_STOP!!! can not find astar road !!!");
      return;
    }
    ROS_INFO("\033[41;37m all of front end time:%f \033[0m", (ros::Time::now()-astar_start_time).toSec());

    // optimizer
    bool result = msplanner_->minco_plan(jps_planner_->flat_traj_);
    if(!result){
      return;
    }

    ROS_INFO("\033[43;32m all of plan time:%f \033[0m", (ros::Time::now().toSec()-current));

    // visualization
    msplanner_->mincoPathPub(msplanner_->final_traj_, plan_start_state_XYTheta, visualizer_->mincoPathPath);
    msplanner_->mincoPointPub(msplanner_->final_traj_, plan_start_state_XYTheta, visualizer_->mincoPointMarker, Eigen::Vector3d(239, 41, 41));
    
    // for replan
    if(plan_start_time_ < 0){
      Traj_start_time_ = ros::Time::now();
      plan_start_time_ = Traj_start_time_.toSec();
    }
    else{
      plan_start_time_ = current + max_replan_time_;
      Traj_start_time_ = ros::Time(plan_start_time_);
    }
    

    MPCPathPub(plan_start_time_);

    Traj_total_time_ = msplanner_->final_traj_.getTotalDuration();
  }

  if ((ros::Time::now() - Traj_start_time_).toSec() >= Traj_total_time_) {
    if (current_waypoint_idx_ >= 0) {
      current_waypoint_idx_++;
      if (current_waypoint_idx_ < ordered_waypoints_.size()) {
        ROS_INFO("Waypoint %d/%zu reached. Moving to the next one.", current_waypoint_idx_ + 1, ordered_waypoints_.size());
        goal_state_ = ordered_waypoints_[current_waypoint_idx_];
        have_goal_ = true;
        state_machine_ = StateMachine::IDLE;
      } else {
        ROS_INFO("All waypoints visited. Mission complete.");
        state_machine_ = StateMachine::IDLE;
        have_goal_ = false;
        current_waypoint_idx_ = -1;
      }
    } else {
      state_machine_ = StateMachine::IDLE;
      have_goal_ = false;
    }
  }
}

bool PlanManager::findJPSRoad(){

  ros::Time current = ros::Time::now();
  Eigen::Vector3d start_state;
  std::vector<Eigen::Vector3d> start_path;
  std::vector<Eigen::Vector3d> start_path_both_end;
  bool if_forward = true;
  if(plan_start_time_ > 0){
    start_path = msplanner_->get_the_predicted_state_and_path(predicted_traj_start_time_, predicted_traj_start_time_ + jps_planner_->jps_truncation_time_, plan_start_state_XYTheta, start_state, if_forward);
    u_int start_path_size = start_path.size();
    u_int start_path_i = 0;
    for(; start_path_i < start_path_size; start_path_i++){
      if(!jps_planner_->JPS_check_if_collision(start_path[start_path_i].head(2)))
        break;
    }
    if(start_path_i == 0){
      start_state = plan_start_state_XYTheta;
      start_path_both_end.push_back(start_path.front());
      start_path_both_end.push_back(start_state);
    }
    else if(start_path_i < start_path_size){
      start_path = std::vector<Eigen::Vector3d>(start_path.begin(), start_path.begin() + start_path_i);
      start_state = start_path.back();
      start_path_both_end.push_back(start_path.front());
      start_path_both_end.push_back(start_state);
    }
    else{
      start_path_both_end.push_back(start_path.front());
      start_path_both_end.push_back(start_state);
    }
  }
  else{
    start_state = plan_start_state_XYTheta;
  }

  jps_planner_->plan(start_state, goal_state_);
  
  jps_planner_->getKinoNodeWithStartPath(start_path, if_forward, plan_start_state_VAJ, plan_start_state_OAJ);


  visualization_msgs::Marker marker;
  marker.header.frame_id = "world";
  marker.header.stamp = ros::Time::now();
  marker.ns = "jps_planner";
  marker.id = 0;
  marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
  marker.action = visualization_msgs::Marker::ADD;
  marker.pose.position.x = 11;
  marker.pose.position.y = 8;
  marker.pose.position.z = 0;
  marker.pose.orientation.x = 0.0;
  marker.pose.orientation.y = 0.0;
  marker.pose.orientation.z = 0.0;
  marker.pose.orientation.w = 1.0;
  marker.scale.z = 0.5;
  marker.color.a = 1.0; // Don't forget to set the alpha!
  marker.color.r = 0.0;
  marker.color.g = 0.0;
  marker.color.b = 0.0;
  double search_time = (ros::Time::now()-current).toSec() * 1000.0;
  std::ostringstream out;
  out << std::fixed <<"JPS: \n"<< std::setprecision(2) << search_time<<" ms";
  marker.text = out.str();
  record_pub_.publish(marker);


  ROS_INFO("\033[40;36m jps_planner_ search time:%lf  \033[0m", (ros::Time::now()-current).toSec());

  return true;
}

void PlanManager::MPCPathPub(const double& traj_start_time){
  Eigen::MatrixXd initstate = msplanner_->get_current_iniState();
  Eigen::MatrixXd finState = msplanner_->get_current_finState();
  Eigen::MatrixXd finalInnerpoints = msplanner_->get_current_Innerpoints();
  Eigen::VectorXd finalpieceTime = msplanner_->get_current_finalpieceTime();
  Eigen::Vector3d iniStateXYTheta = msplanner_->get_current_iniStateXYTheta();

  carstatemsgs::Polynome polynome;
  polynome.header.frame_id = "world";
  polynome.header.stamp = ros::Time::now();
  polynome.init_p.x = initstate.col(0).x();
  polynome.init_p.y = initstate.col(0).y();
  polynome.init_v.x = initstate.col(1).x();
  polynome.init_v.y = initstate.col(1).y();
  polynome.init_a.x = initstate.col(2).x();
  polynome.init_a.y = initstate.col(2).y();
  polynome.tail_p.x = finState.col(0).x();
  polynome.tail_p.y = finState.col(0).y();
  polynome.tail_v.x = finState.col(1).x();
  polynome.tail_v.y = finState.col(1).y();
  polynome.tail_a.x = finState.col(2).x();
  polynome.tail_a.y = finState.col(2).y();

  if(plan_start_time_ < 0) polynome.traj_start_time = ros::Time::now();
  else polynome.traj_start_time = ros::Time(plan_start_time_);

  for(u_int i=0; i<finalInnerpoints.cols(); i++){
    geometry_msgs::Vector3 point;
    point.x = finalInnerpoints.col(i).x();
    point.y = finalInnerpoints.col(i).y();
    point.z = 0.0;
    polynome.innerpoints.push_back(point);
  }
  for(u_int i=0; i<finalpieceTime.size(); i++){
    polynome.t_pts.push_back(finalpieceTime[i]);
  }
  polynome.start_position.x = iniStateXYTheta.x();
  polynome.start_position.y = iniStateXYTheta.y();
  polynome.start_position.z = iniStateXYTheta.z();

  if(!msplanner_->if_standard_diff_){
    polynome.ICR.x = msplanner_->ICR_.x();
    polynome.ICR.y = msplanner_->ICR_.y();
    polynome.ICR.z = msplanner_->ICR_.z();
  }
  
  mpc_polynome_pub_.publish(polynome);
}

// TSP求解器实现
std::vector<int> PlanManager::TSPSolver(const std::vector<std::vector<double>>& dist_matrix) {
    int n = dist_matrix.size();
    
    // 动态规划求解TSP - Held-Karp算法
    // dp[mask][i] = 从起点出发，访问mask中的所有节点，最后到达节点i的最短距离
    std::vector<std::vector<double>> dp(1 << n, std::vector<double>(n, std::numeric_limits<double>::max()));
    std::vector<std::vector<int>> parent(1 << n, std::vector<int>(n, -1));
    
    // 初始化：从起点0出发到各个节点
    for (int i = 1; i < n; ++i) {
        dp[1 << i][i] = dist_matrix[0][i];
    }
    
    // 动态规划填表
    for (int mask = 1; mask < (1 << n); ++mask) {
        for (int u = 0; u < n; ++u) {
            if (!(mask & (1 << u)) || dp[mask][u] == std::numeric_limits<double>::max()) continue;
            
            for (int v = 1; v < n; ++v) {
                if (mask & (1 << v)) continue; // v已经访问过
                
                int new_mask = mask | (1 << v);
                double new_dist = dp[mask][u] + dist_matrix[u][v];
                
                if (new_dist < dp[new_mask][v]) {
                    dp[new_mask][v] = new_dist;
                    parent[new_mask][v] = u;
                }
            }
        }
    }
    
    // 找到最优解（不回到起点的TSP）
    int final_mask = (1 << n) - 1 - 1; // 除了起点0的所有节点
    double min_cost = std::numeric_limits<double>::max();
    int last_node = -1;
    
    for (int i = 1; i < n; ++i) {
        if (dp[final_mask][i] < min_cost) {
            min_cost = dp[final_mask][i];
            last_node = i;
        }
    }
    
    // 重构路径
    std::vector<int> path;
    int mask = final_mask;
    int curr = last_node;
    
    while (curr != -1) {
        path.push_back(curr);
        int prev = parent[mask][curr];
        mask ^= (1 << curr);
        curr = prev;
    }
    path.push_back(0); // 起点
    
    std::reverse(path.begin(), path.end());
    
    ROS_INFO("TSP exact solution found with cost: %.2f", min_cost);
    return path;
}


#endif