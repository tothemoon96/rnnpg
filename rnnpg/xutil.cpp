/*
 * xutil.cpp
 *
 *  Created on: 24 Dec 2013
 *      Author: s1270921
 */

#include "xutil.h"
#include <iostream>
using namespace std;

FILE *xfopen(const char *infile, const char *mode, const char *msg)
{
	FILE *f = fopen(infile, mode);
	if(f == NULL)
	{
		//将错误信息打印到标准错误流
		fprintf(stderr, "open file '%s' failed! (%s)\n", infile, msg);
		exit(1);
	}
	return f;
}

FILE *xfopen(const char *infile, const char *mode)
{
	return xfopen(infile, mode, "");
}

void* xmalloc (size_t size, const char *msg)
{
	void* p = malloc(size);
	if(p == NULL)
	{
		//将错误信息打印到标准错误流
		fprintf(stderr, "alloc memory %d failed (%s)\n", size, msg);
		exit(1);
	}
	return p;
}

void* xmalloc(size_t size)
{
	return xmalloc(size, "");
}

void *xrealloc(void *mem_address, unsigned int newsize, const char *msg)
{
	void *p = realloc(mem_address, newsize);
	if(p == NULL)
	{
		fprintf(stderr, "realloc memory %d failed (%s)\n", newsize, msg);
		exit(1);
	}
	return p;
}

void *xrealloc(void *mem_address, unsigned int newsize)
{
	return xrealloc(mem_address, newsize, "");
}

int split(const char *str, const char *sep, vector<string>& fields)
{
	fields.clear();
	string word;
	int i = 0;
	while(str[i] != '\0')
	{
		while(str[i] != '\0' && strchr(sep, str[i]))
			i ++;
		word.clear();
		while(str[i] != '\0' && strchr(sep, str[i]) == NULL)
		{
			word.append(1, str[i]);
			i ++;
		}
		if(word.length() > 0)
			fields.push_back(word);
	}

	return fields.size();
}

/**
 * @brief
 * 将str字符串根据sep分隔符进行分割，将分割结果存入fields这个string容器中
 * @param str 待分割字符串
 * @param sep 分隔符
 * @param fields 存储分割结果的string的vector
 * @return int 返回存储分割结果的fields.size()
 */
int split(string str, const char *sep, vector<string>& fields)
{
	const char *cstr = str.c_str();
	return split(cstr, sep, fields);
}

/**
 * @brief
 * 设置fin指针到mark字符之后的一个字符，也就是说mark作为一个分隔符，要读取mark字符之后的内容
 * @param mark 分隔符
 * @param fin 读入文件的文件指针
 */
void skiputil(int mark, FILE *fin)
{
	int ch=0;

	while (ch!=mark) {
		ch=fgetc(fin);
		if (feof(fin)) {
			printf("Unexpected end of file (skiputil)\n");
			exit(1);
		}
	}
}

void xstrcpy(char *dst, int dstLen, const char *str)
{
	snprintf(dst, dstLen, "%s", str);
}

char *tolowerN(const char *str)
{
	char *tmp = new char[strlen(str) + 1];
	strcpy(tmp, str);
	for(char *ptr = tmp; *ptr != 0; ptr ++)
		*ptr = tolower(*ptr);

	return tmp;
}

bool atob(const char*str)
{
	char *ptr = tolowerN(str);
	bool ret = strcmp(ptr, "true") == 0;
	delete []ptr;
	return ret;
}

void printsvec(vector<string> &strs)
{
	for(size_t i = 0; i < strs.size(); i ++)
		cout << strs[i] << " ";
	cout << endl;
}
