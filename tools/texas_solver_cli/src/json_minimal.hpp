#pragma once

#include <map>
#include <string>
#include <vector>

namespace jsonmini {

struct Value {
  enum Kind { Null, Bool, Number, String, Array, Object } kind = Null;
  double number = 0.0;
  bool boolean = false;
  std::string str;
  std::vector<Value> array;
  std::map<std::string, Value> object;
};

Value parse(const std::string& text);
std::string stringify(const Value& v);

}  // namespace jsonmini
