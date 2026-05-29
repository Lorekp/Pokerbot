#pragma once

#include <string>

std::string adapt_texassolver_output_to_contract(
    const std::string& output_json,
    double target_exploitability_pct,
    int max_iterations,
    int pot_chips);

std::string build_error_response(const std::string& message, int max_iterations = 0);
