/**
 * @file csv.h
 * @brief The definition of csv object
 *
 * csv type is defined in this file. csv object is used to read and write csv
 * file.
 *
 * @author Eugene Lin <lin.eugene.l.e@gmail.com>
 * @date 2021.5.5
 */
#ifndef __CSV_H__
#define __CSV_H__

#include <bits/stdint-uintn.h>
#include <cerrno>
#include <cstring>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include "common.h"

/**
 * @class csv
 * @breif csv type is used to read/write csv file.
 *
 * csv object is used to deel with csv file I/O. There are several csv file have
 * byte order remark. csv object is able to handle UTF-8 and UTF-16 byte order
 * remark.
 */
class csv
{
protected:
    /**
     * _hasBOM () - check if @b _text has UTF-@b bits kind of byte order mark.
     *
     * In _hasBOM, the first 4 characters are cast to be unsigned int type
     * variable and check if the variable has BOM.
     *
     * @var _text : first line in the file.
     * @var bom : byte order mark number.
     * @var bits : specify what kind of UTF
     *
     * @warning :
     * 1. Make sure @b _text has at least 4 bytes. If @b _text doesn't, the
     * behavior is undefined.
     * 2. The function only supports little endian hardward.
     */
    bool _hasBOM(char *_text, unsigned int bom, short bits);

protected:
    std::vector<std::vector<std::string> > _data;
    std::map<std::string, std::uint16_t> _head;
    std::string _mode;
    std::string _filename;
    FILE *_file;

public:
    csv();

    /**
     * csv - Constructor of csv object for reading file
     * 
     * This constructor is used to read csv file.
     */
    csv(std::string filename,
        std::string mode,
        bool read = true,
        bool head = true,
        int r1 = -1,
        int r2 = -1);

    void setMode(std::string mode);
    void setFileName(std::string filename);

    void setHeaders(std::map<std::string, std::string> mapping,
                    bool replace = false);
    void setHeaders(std::map<std::string, uint16_t> head);
    void setHeader(std::string old, std::string n, bool replace = false);


    bool write(std::string filename, std::string mode, bool head);
    bool read(std::string filename = "",
              std::string mode = "",
              bool head = true,
              int r1 = -1,
              int r2 = -1);
    bool read(bool head = true, int r1 = -1, int r2 = -1);
    void close();
    unsigned int size();


    std::vector<std::string> getRow(int row);
    std::string getElement(int col, int row);
    std::string getElement(std::string col, int row);
    std::map<std::string, std::string> getElements(int nrow);
    std::vector<std::vector<std::string> > getData();

    ~csv();
};

#endif
