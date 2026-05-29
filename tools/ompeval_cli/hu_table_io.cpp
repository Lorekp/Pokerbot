#include "hu_table_io.hpp"

#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

std::string read_npy_header(std::ifstream& in) {
    char magic[6] = {};
    in.read(magic, 6);
    if (std::strncmp(magic, "\x93NUMPY", 6) != 0) {
        throw std::runtime_error("not a NumPy .npy file: " + std::string(magic, 6));
    }
    uint8_t ver_major = 0;
    uint8_t ver_minor = 0;
    in.read(reinterpret_cast<char*>(&ver_major), 1);
    in.read(reinterpret_cast<char*>(&ver_minor), 1);
    uint32_t header_len = 0;
    if (ver_major == 1) {
        uint16_t len16 = 0;
        in.read(reinterpret_cast<char*>(&len16), 2);
        header_len = len16;
    } else if (ver_major == 2) {
        in.read(reinterpret_cast<char*>(&header_len), 4);
    } else {
        throw std::runtime_error("unsupported npy version");
    }
    std::string header(header_len, '\0');
    in.read(header.data(), static_cast<std::streamsize>(header_len));
    return header;
}

bool header_has_substr(const std::string& header, const char* needle) {
    return header.find(needle) != std::string::npos;
}

std::size_t parse_shape_len(const std::string& header) {
    const auto pos = header.find("'shape'");
    if (pos == std::string::npos) {
        throw std::runtime_error("npy header missing shape");
    }
    const auto open = header.find('(', pos);
    const auto close = header.find(')', open);
    if (open == std::string::npos || close == std::string::npos) {
        throw std::runtime_error("npy header malformed shape");
    }
    const std::string shape = header.substr(open + 1, close - open - 1);
    if (shape.find(',') != std::string::npos) {
        const auto comma = shape.find(',');
        const std::string first = shape.substr(0, comma);
        if (first.empty()) {
            return static_cast<std::size_t>(std::stoull(shape.substr(comma + 1)));
        }
    }
    if (shape.empty()) {
        throw std::runtime_error("npy header empty shape");
    }
    return static_cast<std::size_t>(std::stoull(shape));
}

}  // namespace

KeyFile load_keys_file(const std::string& path) {
    KeyFile out{};
    if (path.size() >= 4 && path.substr(path.size() - 4) == ".npy") {
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            throw std::runtime_error("cannot open keys file: " + path);
        }
        const std::string header = read_npy_header(in);
        if (!header_has_substr(header, "'descr': '<u8'") &&
            !header_has_substr(header, "descr': '<u8'")) {
            throw std::runtime_error("unsupported npy dtype (expected 1-D <u8): " + path);
        }
        if (!header_has_substr(header, "'fortran_order': False") &&
            !header_has_substr(header, "fortran_order': False")) {
            throw std::runtime_error("unsupported npy fortran_order (expected False): " + path);
        }
        const std::size_t n = parse_shape_len(header);
        out.keys.resize(n);
        in.read(reinterpret_cast<char*>(out.keys.data()),
                static_cast<std::streamsize>(n * sizeof(uint64_t)));
        if (!in) {
            throw std::runtime_error("short read from npy payload: " + path);
        }
        return out;
    }

    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        throw std::runtime_error("cannot open keys file: " + path);
    }
    const auto size = in.tellg();
    if (size < 0 || static_cast<std::size_t>(size) % sizeof(uint64_t) != 0) {
        throw std::runtime_error("raw keys file size must be multiple of 8 bytes: " + path);
    }
    const std::size_t n = static_cast<std::size_t>(size) / sizeof(uint64_t);
    in.seekg(0, std::ios::beg);
    out.keys.resize(n);
    in.read(reinterpret_cast<char*>(out.keys.data()),
            static_cast<std::streamsize>(n * sizeof(uint64_t)));
    if (!in) {
        throw std::runtime_error("short read from raw keys file: " + path);
    }
    return out;
}

void write_scores_raw(const std::string& path, const uint32_t* scores, std::size_t n) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("cannot open output file: " + path);
    }
    out.write(reinterpret_cast<const char*>(scores),
              static_cast<std::streamsize>(n * sizeof(uint32_t)));
    if (!out) {
        throw std::runtime_error("failed writing output file: " + path);
    }
}
