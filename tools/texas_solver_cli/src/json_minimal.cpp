#include "json_minimal.hpp"

#include <cctype>
#include <cmath>
#include <stdexcept>

namespace jsonmini {
namespace {

struct Parser {
  const std::string& s;
  size_t i = 0;

  void skip_ws() {
    while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i;
  }

  char peek() const { return i < s.size() ? s[i] : '\0'; }

  char get() { return i < s.size() ? s[i++] : '\0'; }

  Value parse_value() {
    skip_ws();
    char c = peek();
    if (c == '{') return parse_object();
    if (c == '[') return parse_array();
    if (c == '"') return parse_string();
    if (c == 't' || c == 'f') return parse_bool();
    if (c == 'n') return parse_null();
    return parse_number();
  }

  Value parse_object() {
    Value v;
    v.kind = Value::Object;
    get();
    skip_ws();
    if (peek() == '}') {
      get();
      return v;
    }
    while (true) {
      skip_ws();
      if (peek() != '"') throw std::runtime_error("expected object key");
      std::string key = parse_string().str;
      skip_ws();
      if (get() != ':') throw std::runtime_error("expected ':'");
      v.object[key] = parse_value();
      skip_ws();
      char c = get();
      if (c == '}') break;
      if (c != ',') throw std::runtime_error("expected ',' or '}'");
    }
    return v;
  }

  Value parse_array() {
    Value v;
    v.kind = Value::Array;
    get();
    skip_ws();
    if (peek() == ']') {
      get();
      return v;
    }
    while (true) {
      v.array.push_back(parse_value());
      skip_ws();
      char c = get();
      if (c == ']') break;
      if (c != ',') throw std::runtime_error("expected ',' or ']'");
    }
    return v;
  }

  Value parse_string() {
    Value v;
    v.kind = Value::String;
    get();
    while (i < s.size()) {
      char c = get();
      if (c == '"') break;
      if (c == '\\') {
        char e = get();
        if (e == '"' || e == '\\' || e == '/') v.str.push_back(e);
        else if (e == 'n') v.str.push_back('\n');
        else if (e == 't') v.str.push_back('\t');
        else v.str.push_back(e);
      } else {
        v.str.push_back(c);
      }
    }
    return v;
  }

  Value parse_bool() {
    Value v;
    v.kind = Value::Bool;
    if (s.compare(i, 4, "true") == 0) {
      i += 4;
      v.boolean = true;
    } else if (s.compare(i, 5, "false") == 0) {
      i += 5;
      v.boolean = false;
    } else {
      throw std::runtime_error("invalid bool");
    }
    return v;
  }

  Value parse_null() {
    if (s.compare(i, 4, "null") != 0) throw std::runtime_error("invalid null");
    i += 4;
    Value v;
    v.kind = Value::Null;
    return v;
  }

  Value parse_number() {
    size_t start = i;
    if (peek() == '-') get();
    while (std::isdigit(static_cast<unsigned char>(peek()))) get();
    if (peek() == '.') {
      get();
      while (std::isdigit(static_cast<unsigned char>(peek()))) get();
    }
    if (peek() == 'e' || peek() == 'E') {
      get();
      if (peek() == '+' || peek() == '-') get();
      while (std::isdigit(static_cast<unsigned char>(peek()))) get();
    }
    Value v;
    v.kind = Value::Number;
    v.number = std::stod(s.substr(start, i - start));
    return v;
  }
};

void escape_str(std::string& out, const std::string& in) {
  out.push_back('"');
  for (char c : in) {
    if (c == '"' || c == '\\') out.push_back('\\');
    out.push_back(c);
  }
  out.push_back('"');
}

void stringify_impl(std::string& out, const Value& v) {
  switch (v.kind) {
    case Value::Null:
      out += "null";
      break;
    case Value::Bool:
      out += v.boolean ? "true" : "false";
      break;
    case Value::Number: {
      char buf[64];
      std::snprintf(buf, sizeof(buf), "%.17g", v.number);
      out += buf;
      break;
    }
    case Value::String:
      escape_str(out, v.str);
      break;
    case Value::Array:
      out += '[';
      for (size_t j = 0; j < v.array.size(); ++j) {
        if (j) out += ',';
        stringify_impl(out, v.array[j]);
      }
      out += ']';
      break;
    case Value::Object:
      out += '{';
      {
        bool first = true;
        for (const auto& kv : v.object) {
          if (!first) out += ',';
          first = false;
          escape_str(out, kv.first);
          out += ':';
          stringify_impl(out, kv.second);
        }
      }
      out += '}';
      break;
  }
}

}  // namespace

Value parse(const std::string& text) {
  Parser p{text};
  Value v = p.parse_value();
  p.skip_ws();
  if (p.i != text.size()) throw std::runtime_error("trailing json");
  return v;
}

std::string stringify(const Value& v) {
  std::string out;
  stringify_impl(out, v);
  return out;
}

}  // namespace jsonmini
