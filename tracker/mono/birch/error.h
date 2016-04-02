/*********************************************************************
 * error.h
 *********************************************************************/

#ifndef _ERROR_H_
#define _ERROR_H_

#include <stdio.h>
#include <stdarg.h>

void KLTError(char const*fmt, ...);
void KLTWarning(char const*fmt, ...);

#endif

