#pragma once
#include <cstdlib>
#include <stdexcept>


using std::size_t;

namespace torchlet::index {

    struct Slice {
        size_t start;
        size_t end;

        Slice(size_t idx) : start(idx), end(idx+1){};
        Slice(size_t start, size_t end) : start(start), end(end) {
            if (end <= start) throw std::runtime_error("Slice end must be > start");
        };

        inline size_t range() const {return end - start;}

    };

}