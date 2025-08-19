#ifndef BRANCH_AND_BOUND_COMBINED_HPP
#define BRANCH_AND_BOUND_COMBINED_HPP

#include <vector>
#include <queue>
#include <limits>
#include <algorithm>
#include <Eigen/Eigen>
#include "hungarian.hpp"

// Node for the combined Assignment-Routing B&B
struct CombinedNode {
    // State
    int last_pos_idx; // Global index in the distance matrix (0=start, 1..n=chairs, n+1..2n=targets)
    int visited_chairs_mask;
    int assigned_targets_mask;
    int level; // How many pairs have been visited

    // Path & Cost
    double current_cost;
    double lower_bound;
    std::vector<int> path_indices; // Sequence of global indices

    bool operator>(const CombinedNode& other) const {
        return lower_bound > other.lower_bound;
    }
};

class BranchAndBoundCombined {
public:
    BranchAndBoundCombined(const Eigen::MatrixXd& dists, int num_tasks) 
        : all_dists_(dists), num_tasks_(num_tasks) {}

    double solve(std::vector<int>& best_path_indices) {
        std::priority_queue<CombinedNode, std::vector<CombinedNode>, std::greater<CombinedNode>> pq;

        // Initial node
        CombinedNode root;
        root.last_pos_idx = 0; // Start point
        root.visited_chairs_mask = 0;
        root.assigned_targets_mask = 0;
        root.level = 0;
        root.current_cost = 0;
        root.path_indices.push_back(0);
        root.lower_bound = calculateLowerBound(root);
        
        pq.push(root);

        double global_best_cost = std::numeric_limits<double>::max();

        while (!pq.empty()) {
            CombinedNode current = pq.top();
            pq.pop();

            if (current.lower_bound >= global_best_cost) {
                continue; // Prune
            }

            if (current.level == num_tasks_) {
                if (current.current_cost < global_best_cost) {
                    global_best_cost = current.current_cost;
                    best_path_indices = current.path_indices;
                }
                continue;
            }

            // Branching: from last target, go to an unvisited chair
            for (int i = 0; i < num_tasks_; ++i) {
                if (!((current.visited_chairs_mask >> i) & 1)) { // If chair i is not visited
                    
                    // Then, from that chair, go to an unassigned target
                    for (int j = 0; j < num_tasks_; ++j) {
                        if (!((current.assigned_targets_mask >> j) & 1)) { // If target j is not assigned
                            
                            CombinedNode child = current;
                            child.level++;

                            // Update path and cost
                            int chair_idx = 1 + i;
                            int target_idx = 1 + num_tasks_ + j;
                            child.current_cost += all_dists_(current.last_pos_idx, chair_idx);
                            child.current_cost += all_dists_(chair_idx, target_idx);
                            
                            // Update state
                            child.last_pos_idx = target_idx;
                            child.visited_chairs_mask |= (1 << i);
                            child.assigned_targets_mask |= (1 << j);
                            
                            // Update path history
                            child.path_indices.push_back(chair_idx);
                            child.path_indices.push_back(target_idx);

                            // Calculate bound and push to queue
                            child.lower_bound = calculateLowerBound(child);
                            if (child.lower_bound < global_best_cost) {
                                pq.push(child);
                            }
                        }
                    }
                }
            }
        }
        return global_best_cost;
    }

private:
    const Eigen::MatrixXd& all_dists_;
    int num_tasks_;

    double primMST(const std::vector<int>& nodes) {
        if (nodes.empty()) return 0.0;
        double mst_cost = 0.0;
        std::vector<double> key(nodes.size(), std::numeric_limits<double>::max());
        std::vector<bool> in_mst(nodes.size(), false);
        
        key[0] = 0.0;
        
        for (int count = 0; count < nodes.size(); ++count) {
            double min_key = std::numeric_limits<double>::max();
            int u = -1;

            for (int v_idx = 0; v_idx < nodes.size(); ++v_idx) {
                if (!in_mst[v_idx] && key[v_idx] < min_key) {
                    min_key = key[v_idx];
                    u = v_idx;
                }
            }
            
            if (u == -1) break;

            in_mst[u] = true;
            mst_cost += min_key;

            for (int v_idx = 0; v_idx < nodes.size(); ++v_idx) {
                double weight = all_dists_(nodes[u], nodes[v_idx]);
                if (!in_mst[v_idx] && weight < key[v_idx]) {
                    key[v_idx] = weight;
                }
            }
        }
        return mst_cost;
    }

    double calculateLowerBound(const CombinedNode& node) {
        double bound = node.current_cost;

        // 1. Hungarian cost for remaining assignments
        std::vector<int> unvisited_chairs;
        std::vector<int> unassigned_targets;
        for (int i = 0; i < num_tasks_; ++i) {
            if (!((node.visited_chairs_mask >> i) & 1)) unvisited_chairs.push_back(1 + i);
            if (!((node.assigned_targets_mask >> i) & 1)) unassigned_targets.push_back(1 + num_tasks_ + i);
        }

        if (!unvisited_chairs.empty()) {
            Eigen::MatrixXd sub_cost_matrix(unvisited_chairs.size(), unassigned_targets.size());
            for (size_t i = 0; i < unvisited_chairs.size(); ++i) {
                for (size_t j = 0; j < unassigned_targets.size(); ++j) {
                    sub_cost_matrix(i, j) = all_dists_(unvisited_chairs[i], unassigned_targets[j]);
                }
            }
            HungarianAlgorithm hungarian;
            std::vector<int> dummy_assignment;
            bound += hungarian.solve(sub_cost_matrix, dummy_assignment);
        }

        // 2. MST cost for connecting remaining nodes
        std::vector<int> mst_nodes;
        mst_nodes.push_back(node.last_pos_idx); // Current position
        for (int chair_idx : unvisited_chairs) mst_nodes.push_back(chair_idx);
        for (int target_idx : unassigned_targets) mst_nodes.push_back(target_idx);
        
        // Remove duplicates
        std::sort(mst_nodes.begin(), mst_nodes.end());
        mst_nodes.erase(std::unique(mst_nodes.begin(), mst_nodes.end()), mst_nodes.end());

        bound += primMST(mst_nodes);

        return bound;
    }
};

#endif // BRANCH_AND_BOUND_COMBINED_HPP
