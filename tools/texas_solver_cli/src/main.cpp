#include "json_minimal.hpp"
#include "request_adapter.hpp"
#include "response_adapter.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>

namespace fs = std::filesystem;

static std::string read_stdin() {
  std::ostringstream oss;
  oss << std::cin.rdbuf();
  return oss.str();
}

static std::string env_or(const char* key, const std::string& fallback) {
  if (const char* v = std::getenv(key)) return std::string(v);
  return fallback;
}

static int default_thread_count() {
  const char* env = std::getenv("TEXAS_SOLVER_THREADS");
  if (env) return std::max(1, std::atoi(env));
  const unsigned hc = std::thread::hardware_concurrency();
  return hc > 0 ? static_cast<int>(hc) : 8;
}

static fs::path resolve_console_binary() {
  std::string direct = env_or("TEXAS_SOLVER_CONSOLE", "");
  if (!direct.empty()) return fs::path(direct);
  std::string root = env_or("TEXAS_SOLVER_ROOT", "");
  if (root.empty()) return {};
#ifdef _WIN32
  return fs::path(root) / "console_solver.exe";
#else
  return fs::path(root) / "console_solver";
#endif
}

static int run_console_solver(const fs::path& console, const fs::path& workdir, const fs::path& config) {
#ifdef _WIN32
  std::ostringstream cmd;
  cmd << '"' << console.string() << "\" -i \"" << config.string() << '"';
  return std::system(cmd.str().c_str());
#else
  std::ostringstream cmd;
  cmd << '"' << console.string() << "\" -i \"" << config.string() << '"';
  return std::system(cmd.str().c_str());
#endif
}

int main() {
  const std::string input = read_stdin();
  if (input.empty()) {
    std::cout << build_error_response("empty stdin") << '\n';
    return 1;
  }

  SolveRequest req;
  try {
    jsonmini::Value root = jsonmini::parse(input);
    req = parse_solve_request(root);
  } catch (const std::exception& ex) {
    std::cout << build_error_response(std::string("request parse: ") + ex.what()) << '\n';
    return 1;
  }

  fs::path console = resolve_console_binary();
  if (console.empty() || !fs::exists(console)) {
    std::cout << build_error_response(
                     "TEXAS_SOLVER_CONSOLE or TEXAS_SOLVER_ROOT not set or console_solver missing",
                     req.max_iterations)
              << '\n';
    return 1;
  }

  fs::path workdir = fs::temp_directory_path() / "texas_solver_cli_work";
  std::error_code ec;
  fs::create_directories(workdir, ec);

  // TexasSolver expects resources/ adjacent to cwd or install root
  std::string root = env_or("TEXAS_SOLVER_ROOT", console.parent_path().string());
  fs::path resources = fs::path(root) / "resources";
  if (fs::exists(resources)) {
    fs::path link = workdir / "resources";
    if (!fs::exists(link)) {
      fs::create_directory_symlink(resources, link, ec);
    }
  }

  const fs::path config_path = workdir / "input.txt";
  const fs::path output_path = workdir / "output_result.json";
  {
    std::ofstream cfg(config_path);
    cfg << build_texassolver_config(req, default_thread_count());
  }

  const int rc = run_console_solver(console, workdir, config_path);
  if (rc != 0 || !fs::exists(output_path)) {
    std::cout << build_error_response("TexasSolver console_solver failed", req.max_iterations) << '\n';
    return 1;
  }

  std::string output_json;
  {
    std::ifstream in(output_path);
    std::ostringstream oss;
    oss << in.rdbuf();
    output_json = oss.str();
  }

  const int pot_chips = static_cast<int>(req.starting_pot_bb * req.chip_unit_bb);
  try {
    std::cout << adapt_texassolver_output_to_contract(
                     output_json, req.target_exploitability_pct, req.max_iterations, pot_chips)
              << '\n';
  } catch (const std::exception& ex) {
    std::cout << build_error_response(std::string("response adapt: ") + ex.what(), req.max_iterations) << '\n';
    return 1;
  }
  return 0;
}
