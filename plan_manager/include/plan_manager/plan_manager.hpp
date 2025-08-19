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
#include "plan_manager/branch_and_bound.hpp"

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
    StateMachine state_machine_ = StateMachine::INIT;

public:
    PlanManager(ros::NodeHandle nh);
    ~PlanManager();

    // 任务规划流程函数
    bool generateRandomPositions(int pairs);
    void visualizeChairsAndTargets();
    bool solvePath();
    bool solvePathWithBranchAndBound();
    
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
  nh_.param<double>("max_replan_time", max_replan_time_, 3.0);

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
    if (generateRandomPositions(6)) {
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
  
  if (!generateRandomPositions(6)) {
      ROS_ERROR("Mission aborted: Failed to generate all required chair/target positions.");
      visualizeChairsAndTargets(); // Call to clear any old markers
      state_machine_ = StateMachine::IDLE;
      return;
  }
  visualizeChairsAndTargets();

  if (solvePathWithBranchAndBound()) {
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

bool PlanManager::generateRandomPositions(int pairs) {
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

    generate_points(chair_positions_, pairs, "Chair");
    generate_points(target_positions_, pairs, "Target");

    if (chair_positions_.size() != pairs || target_positions_.size() != pairs) {
        ROS_ERROR("Failed to generate all required points. Aborting mission setup.");
        chair_positions_.clear();
        target_positions_.clear();
        return false;
    }
    return true;
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

bool PlanManager::solvePathWithBranchAndBound() {
    if (chair_positions_.empty() || target_positions_.empty()) {
        ROS_ERROR("No chair or target positions available for path planning.");
        return false;
    }

    ROS_INFO("Starting combined Branch and Bound optimization...");

    // 1. Build the full distance matrix
    int num_tasks = chair_positions_.size();
    int matrix_size = 1 + 2 * num_tasks; // 0:start, 1..n:chairs, n+1..2n:targets
    Eigen::MatrixXd all_dists(matrix_size, matrix_size);
    std::vector<Eigen::Vector3d> all_points;
    
    all_points.push_back(current_state_XYTheta_);
    all_points.insert(all_points.end(), chair_positions_.begin(), chair_positions_.end());
    all_points.insert(all_points.end(), target_positions_.begin(), target_positions_.end());

    for (int i = 0; i < matrix_size; ++i) {
        for (int j = i; j < matrix_size; ++j) {
            if (i == j) {
                all_dists(i, j) = 0;
                continue;
            }
            bool path_found = jps_planner_->plan(all_points[i], all_points[j]);
            double len = path_found ? jps_planner_->getPathLength() : std::numeric_limits<double>::max();
            if (len >= std::numeric_limits<double>::max()) {
                 ROS_WARN("Cannot find path between point %d and %d. Setting cost to infinity.", i, j);
            }
            all_dists(i, j) = all_dists(j, i) = len;
        }
    }

    // 2. Solve with the combined B&B solver
    BranchAndBoundCombined bnb_solver(all_dists, num_tasks);
    std::vector<int> best_path_indices;
    double optimal_cost = bnb_solver.solve(best_path_indices);

    if (optimal_cost >= std::numeric_limits<double>::max()) {
        ROS_ERROR("Combined B&B failed to find a valid solution.");
        return false;
    }

    ROS_INFO("Combined B&B solution found with cost: %.2f", optimal_cost);

    // 3. Reconstruct the waypoint list from the best path indices
    ordered_waypoints_.clear();
    for (int idx : best_path_indices) {
        ordered_waypoints_.push_back(all_points[idx]);
    }

    // Print the optimized path
    std::stringstream path_ss;
    for(size_t i = 0; i < best_path_indices.size(); ++i) {
        path_ss << best_path_indices[i] << (i == best_path_indices.size() - 1 ? "" : " -> ");
    }
    ROS_INFO("Optimal path sequence (indices): %s", path_ss.str().c_str());


    ROS_INFO("Combined B&B optimized task sequence generated successfully with %zu waypoints.", ordered_waypoints_.size());
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