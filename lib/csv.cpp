#include <include/csv.h>
#include <sys/types.h>
#include <clocale>
#include <cstdint>
#include <cstdio>
#include <locale>
#include <stdexcept>

csv::csv()
{
    _file = NULL;
}
csv::csv(std::string filename,
         std::string mode,
         bool r,
         bool head,
         int r1,
         int r2)
{
    _filename = filename;
    _mode = mode;
    _file = NULL;
    if (r) {
        read(filename, mode, head, r1, r2);
    }
}

csv::~csv()
{
    if (_file)
        fclose(_file);
}

bool csv::_hasBOM(char *_text, unsigned int bom, short bits)
{
    unsigned int result, *text;
    text = (unsigned int *) _text;
    result = *text ^ bom;
    return !(result << bits);
}

bool csv::read(bool head, int r1, int r2){
    return read(_filename, _mode, head, r1, r2);
}

bool csv::read(std::string filename,
               std::string mode,
               bool head,
               int r1,
               int r2)
{
    _data.clear();
    if (filename.compare("") == 0) {
        filename = _filename;
    }

    if (mode.compare("") == 0) {
        mode = _mode;
    }

    if (_file) {
        close();
    }

    _file = fopen(filename.c_str(), mode.c_str());

    if (errno != 0) {
        if (errno == ENOENT) {
            throw std::ios_base::failure("No such file or directory");
        } else if (errno == EINVAL) {
            std::string err = "\"" + mode + "\"is and invalid argument.";
            throw std::invalid_argument(err);
        } else {
            throw std::ios_base::failure("Error on opening file");
        }
    }

    size_t size = 0;
    char *line_ptr = NULL;
    std::vector<std::string> text;

    if (getline(&line_ptr, &size, _file) > 0) {
        // check byte order mark
        if (_hasBOM(line_ptr, 0x00BFBBEF,
                    8)) {  // EF BB BF, 0x00BFBBEF for little endian
            memmove(line_ptr, line_ptr + 3, size - 3);  // remove byte order
        } else if (_hasBOM(line_ptr, 0x0000FFFE,
                           16)) {  // FE FF, 0x0000FFFE for little endian
            memmove(line_ptr, line_ptr + 2, size - 2);  // remvoe byte order
        }
    } else {
        return false;
    }


    text = split(line_ptr, ',');
    if (head) {
        for (int i = 0, size = text.size(); i < size; ++i) {
            if (text[i].compare("") != 0) {
                _head[text[i]] = i;
            } else {
                return false;  // col head has empty head fail to set head
            }
        }
    } else {
        _data.push_back(text);
    }



    while (getline(&line_ptr, &size, _file) > 0) {
        text = split(line_ptr, ',');
        _data.push_back(text);
    }
    free(line_ptr);

    if (r1 >= 0 && r2 < 0) {
        _data.erase(_data.begin() + r1, _data.end());
    } else if (r1 >= 0 && r2 >= 0) {
        _data.erase(_data.begin() + r2, _data.end());
        _data.erase(_data.begin(), _data.begin() + r1);
    }

    // for(unsigned int i = 0; i < _data.size(); ++i){
    //     for(unsigned int j = 0; j < _data[i].size(); ++j){
    //         printf("%s ", _data[i][j].c_str());
    //     }
    //     printf("\n");
    // }

    return true;
}

void csv::setMode(std::string mode)
{
    _mode = mode;
}

void csv::setFileName(std::string filename)
{
    _filename = filename;
}

void csv::setHeader(std::string old, std::string n, bool replace)
{
    _head[n] = _head[old];
    if (replace) {
        _head.erase(old);
    }
}

void csv::setHeaders(std::map<std::string, std::uint16_t> head)
{
    _head = head;
}

void csv::setHeaders(std::map<std::string, std::string> mapping, bool replace)
{
    for (std::map<std::string, std::string>::iterator it = mapping.begin();
         it != mapping.end(); ++it) {
        setHeader(it->second, it->first, replace);
    }
}

void csv::close()
{
    if (_file) {
        fclose(_file);
        _file = NULL;
    }
}

bool csv::write(std::string filename, std::string mode, bool head)
{
    return true;
}


std::vector<std::string> csv::getRow(int row)
{
    int idx;
    if(row < 0){
        idx = (int)_data.size() + row;
    }else
        idx = row;
    return _data.at(idx);
}

std::string csv::getElement(std::string _col, int row)
{
    uint16_t col = _head[_col];
    return getRow(row).at(col);
}

std::string csv::getElement(int col, int row)
{
    return getRow(row).at(col);
}

std::map<std::string, std::string> csv::getElements(int row)
{
    std::map<std::string, std::string> data;

    for (std::map<std::string, std::uint16_t>::iterator it = _head.begin();
         it != _head.end(); ++it) {
        data[it->first] = _data[row][it->second];
    }

    return data;
}

unsigned int csv::nrows(){
    return _data.size();
}

std::vector<std::vector<std::string> > csv::getData(){
    return _data;
}
