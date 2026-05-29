#pragma once

#include "json_minimal.hpp"

#include <string>
#include <vector>

struct SolveRequest {
  std::string oop_range;
  std::string ip_range;
  std::vector<std::string> board;
  double starting_pot_bb = 0.0;
  double effective_stack_bb = 0.0;
  double chip_unit_bb = 100.0;
  std::string bet_sizes;
  std::string raise_sizes;
  int max_iterations = 10000;
  double target_exploitability_pct = 0.5;
  std::vector<std::string> action_path;
};

SolveRequest parse_solve_request(const jsonmini::Value& root);
std::string build_texassolver_config(const SolveRequest& req, int thread_num);
