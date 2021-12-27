#pragma once

#include <chrono>
#include <vector>
#include <limits>

class Timer
{
public:
	Timer(const size_t nTimers = 1);
	void start();
	void stop();
	size_t getMin();
	size_t getMax();
	size_t getAvg();
private:
	void computeTime();
	size_t m_currentIndex = 0;
	std::vector<std::chrono::system_clock::time_point> m_starts;
	std::vector<std::chrono::system_clock::time_point> m_ends;

	bool m_isComputed = false;
	size_t m_min = std::numeric_limits<size_t>::max();
	size_t m_max = std::numeric_limits<size_t>::min();
	size_t m_avg = 0;
};

