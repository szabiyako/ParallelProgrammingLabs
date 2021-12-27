#include "timer.h"

#include <cassert>

Timer::Timer(const size_t nTimers)
	: m_starts(nTimers),
	  m_ends(nTimers)
{
	assert(nTimers > 0);
}

void Timer::start()
{
	m_starts[m_currentIndex] = std::chrono::system_clock::now();
}

void Timer::stop()
{
	m_ends[m_currentIndex++] = std::chrono::system_clock::now();
	m_isComputed = false;
}

size_t Timer::getMin()
{
	if (!m_isComputed)
		computeTime();
	return m_min;
}

size_t Timer::getMax()
{
	if (!m_isComputed)
		computeTime();
	return m_max;
}

size_t Timer::getAvg()
{
	if (!m_isComputed)
		computeTime();
	return m_avg;
}

void Timer::computeTime()
{
	assert(m_currentIndex > 0);
	m_min = std::numeric_limits<size_t>::max();
	m_max = std::numeric_limits<size_t>::min();
	m_avg = 0;
	for (size_t i = 0; i < m_currentIndex; ++i) {
		const size_t timeInNs = std::chrono::duration_cast<std::chrono::nanoseconds>(m_ends[i] - m_starts[i]).count();
		if (timeInNs < m_min)
			m_min = timeInNs;
		if (timeInNs > m_max)
			m_max = timeInNs;
		m_avg += timeInNs;
	}
	m_avg = m_avg / m_currentIndex;
	m_isComputed = true;
}
