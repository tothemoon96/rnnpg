/*
 * AncChar.h
 *
 *  Created on: 2014年3月13日
 *      Author: xing
 */

#ifndef ANCCHAR_H_
#define ANCCHAR_H_

#include <string>
#include <iostream>
using namespace std;

class AncChar
{
public:
	bool isSameRhythm(const AncChar &another)
	{
		return tone == another.tone && numstr == another.numstr
				&& reprChar == another.reprChar;
	}

//	string toString()
//	{
//		char buf[1024];
//		snprintf(buf,sizeof(buf),"%s < -- %s:%s%s", ch, )
//	}

	string ch;//送/梦/...
	int tone;//3
	string numstr;//一
	string reprChar;//送
};


#endif /* ANCCHAR_H_ */
