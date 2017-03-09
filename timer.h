#include <time.h>
#include <sys/time.h>

#ifdef __MACH__
#include <mach/clock.h>
#include <mach/mach.h>
#endif

extern"C" void diff(timespec* start, timespec* end, timespec* diff);

extern"C" void get_time( timespec* time );
