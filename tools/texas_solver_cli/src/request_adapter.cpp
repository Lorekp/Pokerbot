#include "request_adapter.hpp"

#include <sstream>
#include <stdexcept>

namespace {

std::string require_string(const jsonmini::Value& obj, const char* key) {
  auto it = obj.object.find(key);
  if (it == obj.object.end() || it->second.kind != jsonmini::Value::String) {
    throw std::runtime_error(std::string("missing string field: ") + key);
  }
  return it->second.str;
}

double require_number(const jsonmini::Value& obj, const char* key, double fallback) {
  auto it = obj.object.find(key);
  if (it == obj.object.end()) return fallback;
  if (it->second.kind != jsonmini::Value::Number) {
    throw std::runtime_error(std::string("expected number: ") + key);
  }
  return it->second.number;
}

std::vector<std::string> read_string_array(const jsonmini::Value& obj, const char* key) {
  std::vector<std::string> out;
  auto it = obj.object.find(key);
  if (it == obj.object.end() || it->second.kind == jsonmini::Value::Null) return out;
  if (it->second.kind != jsonmini::Value::Array) {
    throw std::runtime_error(std::string("expected array: ") + key);
  }
  for (const auto& el : it->second.array) {
    if (el.kind != jsonmini::Value::String) continue;
    out.push_back(el.str);
  }
  return out;
}

int parse_raise_pct(const std::string& raise_sizes) {
  // "2.5x" -> 250 (percent of pot heuristic for TexasSolver raise line)
  if (raise_sizes.size() >= 2 && raise_sizes.back() == 'x') {
    double mult = std::stod(raise_sizes.substr(0, raise_sizes.size() - 1));
    return static_cast<int>(mult * 100.0);
  }
  return 250;
}

void append_bet_lines(std::ostringstream& oss, int raise_pct) {
  const char* streets[] = {"flop", "turn", "river"};
  const char* roles[] = {"oop", "ip"};
  for (const char* street : streets) {
    for (const char* role : roles) {
      oss << "set_bet_sizes " << role << ',' << street << ",bet,50\n";
      oss << "set_bet_sizes " << role << ',' << street << ",raise," << raise_pct << '\n';
      oss << "set_bet_sizes " << role << ',' << street << ",allin\n";
    }
  }
}

}  // namespace

SolveRequest parse_solve_request(const jsonmini::Value& root) {
  if (root.kind != jsonmini::Value::Object) throw std::runtime_error("request must be object");
  SolveRequest req;
  req.oop_range = require_string(root, "oop_range");
  req.ip_range = require_string(root, "ip_range");
  req.board = read_string_array(root, "board");
  req.starting_pot_bb = require_number(root, "starting_pot_bb", 0.0);
  req.effective_stack_bb = require_number(root, "effective_stack_bb", 0.0);
  req.chip_unit_bb = require_number(root, "chip_unit_bb", 100.0);
  req.bet_sizes = require_string(root, "bet_sizes");
  req.raise_sizes = require_string(root, "raise_sizes");
  req.max_iterations = static_cast<int>(require_number(root, "max_iterations", 10000));
  req.target_exploitability_pct = require_number(root, "target_exploitability_pct", 0.5);
  req.action_path = read_string_array(root, "action_path");
  if (req.board.size() < 3) throw std::runtime_error("board must have at least 3 cards");
  return req;
}

std::string build_texassolver_config(const SolveRequest& req, int thread_num) {
  const int pot_chips = static_cast<int>(req.starting_pot_bb * req.chip_unit_bb);
  const int stack_chips = static_cast<int>(req.effective_stack_bb * req.chip_unit_bb);
  const int raise_pct = parse_raise_pct(req.raise_sizes);

  std::ostringstream oss;
  oss << "set_pot " << pot_chips << '\n';
  oss << "set_effective_stack " << stack_chips << '\n';
  oss << "set_board " << req.board[0] << ',' << req.board[1] << ',' << req.board[2] << '\n';
  oss << "set_range_oop " << req.oop_range << '\n';
  oss << "set_range_ip " << req.ip_range << '\n';
  append_bet_lines(oss, raise_pct);
  oss << "set_allin_threshold 0.15\n";
  oss << "build_tree\n";
  oss << "set_thread_num " << thread_num << '\n';
  oss << "set_accuracy " << req.target_exploitability_pct << '\n';
  oss << "set_max_iteration " << req.max_iterations << '\n';
  oss << "set_print_interval 50\n";
  oss << "set_use_isomorphism 1\n";
  oss << "start_solve\n";
  oss << "set_dump_rounds 2\n";
  oss << "dump_result output_result.json\n";
  return oss.str();
}
