#include "timer.h"

void diff(timespec* start, timespec* end, timespec* diff)
{
  if( (end->tv_nsec-start->tv_nsec) < 0 )
    {
      diff->tv_sec = end->tv_sec-start->tv_sec-1;
      diff->tv_nsec = 1000000000LL+end->tv_nsec-start->tv_nsec;
    } 
  else
    {
      diff->tv_sec = end->tv_sec-start->tv_sec;
      diff->tv_nsec = end->tv_nsec-start->tv_nsec;
    }

  return;
}

void get_time( timespec* ts )
{

#ifdef __MACH__ // OS X does not have clock_gettime, use clock_get_time
  clock_serv_t cclock;
  mach_timespec_t mts;
  host_get_clock_service(mach_host_self(), CALENDAR_CLOCK, &cclock);
  clock_get_time(cclock, &mts);
  mach_port_deallocate(mach_task_self(), cclock);
  ts->tv_sec = mts.tv_sec;
  ts->tv_nsec = mts.tv_nsec;
#else
  clock_gettime(CLOCK_MONOTONIC, ts);
#endif

  return;
}
