#include "response_adapter.hpp"

#include "json_minimal.hpp"

#include <algorithm>
#include <map>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {

std::string normalize_action_key(const std::string& raw) {
  std::string k = raw;
  std::transform(k.begin(), k.end(), k.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  if (k == "call") return "check";
  if (k.find("all") != std::string::npos) return "all_in";
  if (k.find("bet") != std::string::npos || k.find("raise") != std::string::npos) return "bet_0.50";
  if (k == "check" || k == "fold") return k;
  return raw;
}

void walk_actions(
    const jsonmini::Value& node,
    std::map<std::string, double>& dist,
    std::vector<std::string>& avail) {
  if (node.kind != jsonmini::Value::Object) return;
  auto acts_it = node.object.find("actions");
  if (acts_it != node.object.end() && acts_it->second.kind == jsonmini::Value::Object) {
    for (const auto& kv : acts_it->second.object) {
      std::string key = normalize_action_key(kv.first);
      if (std::find(avail.begin(), avail.end(), key) == avail.end()) {
        avail.push_back(key);
      }
      double mass = 0.0;
      int count = 0;
      if (kv.second.kind == jsonmini::Value::Number) {
        mass = kv.second.number;
        count = 1;
      } else if (kv.second.kind == jsonmini::Value::Object) {
        for (const auto& hand : kv.second.object) {
          if (hand.second.kind == jsonmini::Value::Number) {
            mass += hand.second.number;
            ++count;
          }
        }
      }
      if (count > 0) dist[key] += mass / static_cast<double>(count);
    }
  }
  auto children_it = node.object.find("childrens");
  if (children_it != node.object.end() && children_it->second.kind == jsonmini::Value::Object) {
    for (const auto& child : children_it->second.object) {
      walk_actions(child.second, dist, avail);
    }
  }
  auto children2_it = node.object.find("children");
  if (children2_it != node.object.end()) walk_actions(children2_it->second, dist, avail);
}

void normalize_distribution(std::map<std::string, double>& dist) {
  double sum = 0.0;
  for (const auto& kv : dist) sum += kv.second;
  if (sum <= 0.0) return;
  for (auto& kv : dist) kv.second /= sum;
}

jsonmini::Value num(double x) {
  return jsonmini::Value{jsonmini::Value::Number, x, false, "", {}, {}};
}

jsonmini::Value str(const std::string& s) {
  return jsonmini::Value{jsonmini::Value::String, 0.0, false, s, {}, {}};
}

jsonmini::Value boolean(bool b) {
  return jsonmini::Value{jsonmini::Value::Bool, 0.0, b, "", {}, {}};
}

}  // namespace

std::string build_error_response(const std::string& message, int max_iterations) {
  jsonmini::Value out;
  out.kind = jsonmini::Value::Object;
  out.object["exploitability"] = num(0.0);
  out.object["target_exploitability"] = num(0.0);
  out.object["exploitability_pct_of_pot"] = num(0.0);
  out.object["converged"] = boolean(false);
  out.object["iterations"] = num(static_cast<double>(max_iterations));
  out.object["iterations_used"] = num(0.0);
  out.object["available_actions"] = jsonmini::Value{jsonmini::Value::Array, 0.0, false, "", {}, {}};
  out.object["action_distribution"] = jsonmini::Value{jsonmini::Value::Object, 0.0, false, "", {}, {}};
  out.object["solver_version"] = str("texas-solver@console");
  out.object["error"] = str(message);
  return jsonmini::stringify(out);
}

std::string adapt_texassolver_output_to_contract(
    const std::string& output_json,
    double target_exploitability_pct,
    int max_iterations,
    int pot_chips) {
  jsonmini::Value root = jsonmini::parse(output_json);
  if (root.kind != jsonmini::Value::Object) {
    return build_error_response("TexasSolver output is not a JSON object", max_iterations);
  }

  double exploit_pct = 100.0;
  int iterations_used = 0;
  auto meta_it = root.object.find("meta");
  if (meta_it != root.object.end() && meta_it->second.kind == jsonmini::Value::Object) {
    auto exp_it = meta_it->second.object.find("exploitability");
    if (exp_it != meta_it->second.object.end() && exp_it->second.kind == jsonmini::Value::Number) {
      exploit_pct = exp_it->second.number;
    }
    auto iter_it = meta_it->second.object.find("iteration");
    if (iter_it != meta_it->second.object.end() && iter_it->second.kind == jsonmini::Value::Number) {
      iterations_used = static_cast<int>(iter_it->second.number);
    }
  }

  const double target_exploitability = pot_chips * (target_exploitability_pct / 100.0);
  const double exploitability = pot_chips * (exploit_pct / 100.0);
  const bool converged = exploit_pct <= target_exploitability_pct + 1e-6;

  std::map<std::string, double> dist;
  std::vector<std::string> avail;
  walk_actions(root, dist, avail);
  normalize_distribution(dist);

  jsonmini::Value out;
  out.kind = jsonmini::Value::Object;
  out.object["exploitability"] = num(exploitability);
  out.object["target_exploitability"] = num(target_exploitability);
  out.object["exploitability_pct_of_pot"] = num(exploit_pct);
  out.object["converged"] = boolean(converged);
  out.object["iterations"] = num(static_cast<double>(max_iterations));
  out.object["iterations_used"] = num(static_cast<double>(iterations_used));

  jsonmini::Value dist_obj;
  dist_obj.kind = jsonmini::Value::Object;
  for (const auto& kv : dist) dist_obj.object[kv.first] = num(kv.second);
  out.object["action_distribution"] = dist_obj;

  jsonmini::Value avail_arr;
  avail_arr.kind = jsonmini::Value::Array;
  for (const auto& a : avail) avail_arr.array.push_back(str(a));
  out.object["available_actions"] = avail_arr;
  out.object["solver_version"] = str("texas-solver@console");
  out.object["error"] = jsonmini::Value{jsonmini::Value::Null, 0.0, false, "", {}, {}};
  return jsonmini::stringify(out);
}
