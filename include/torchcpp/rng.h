#pragma once
#include <random>


class Generator {

    public:
        void manual_seed(size_t s) {m_engine.seed(s); m_seed = s;};

        Generator() {
            std::random_device rd;
            m_seed = rd();
            m_engine = std::mt19937(m_seed);
        }; 

        Generator(size_t s) {
            m_seed= s;
            m_engine = std::mt19937(m_seed);
        };

        std::mt19937&       engine()       noexcept { return m_engine; }  // for drawing
        const std::mt19937& engine() const noexcept { return m_engine; }  // read-only
    
    private:
        std::mt19937 m_engine;
        size_t m_seed;

        Generator(const Generator&) = delete;
        Generator(Generator&&) = delete;
        Generator& operator=(const Generator&) = delete;
        Generator& operator=(Generator&&) = delete;
};

