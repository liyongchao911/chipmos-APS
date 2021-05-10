#include <include/csv.h>
#include <sys/types.h>
#include <clocale>
#include <cstdint>
#include <cstdio>
#include <locale>
#include <stdexcept>

csv_t::csv_t(csv_t &csv)
{
    this->_file = NULL;
    this->_data = csv._data;
    this->_head = csv._head;
    this->_mode = csv._mode;
    this->_filename = csv._filename;
}

csv_t::csv_t()
{
    _file = NULL;
}
csv_t::csv_t(std::string filename,
             std::string mode,
             bool r,
             bool head,
             int r1,
             int r2)
{
    _filename = filename;
    _mode = mode;
    _file = NULL;
    bool retval;
    if (r) {
        retval = read(filename, mode, head, r1, r2);
    }
}

void csv_t::trim(std::string text)
{
    size_t found;
    iter(_data, i)
    {
        iter(_data[i], j)
        {
            found = _data[i][j].find_last_not_of(text);
            if (found != std::string::npos)
                _data[i][j].erase(found + 1);
            else
                _data[i][j].clear();
        }
    }
}


csv_t::~csv_t()
{
    if (_file != NULL)
        fclose(_file);
    _file = NULL;
}

bool csv_t::_hasBOM(char *_text, unsigned int bom, short bits)
{
    unsigned int result, *text;
    text = (unsigned int *) _text;
    result = *text ^ bom;
    return !(result << bits);
}

bool csv_t::read(bool head, int r1, int r2)
{
    return read(_filename, _mode, head, r1, r2);
}

bool csv_t::read(std::string filename,
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


    int i = 0;
    while (getline(&line_ptr, &size, _file) > 0) {
        text = split(line_ptr, ',');
        ++i;
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

void csv_t::setMode(std::string mode)
{
    _mode = mode;
}

void csv_t::setFileName(std::string filename)
{
    _filename = filename;
}

void csv_t::setHeader(std::string old, std::string n, bool replace)
{
    _head[n] = _head[old];
    if (replace) {
        _head.erase(old);
    }
}

void csv_t::setHeaders(std::map<std::string, std::uint16_t> head)
{
    _head = head;
}

void csv_t::setHeaders(std::map<std::string, std::string> mapping, bool replace)
{
    for (std::map<std::string, std::string>::iterator it = mapping.begin();
         it != mapping.end(); ++it) {
        setHeader(it->second, it->first, replace);
    }
}

void csv_t::close()
{
    if (_file) {
        fclose(_file);
        _file = NULL;
    }
}

bool csv_t::write(std::string filename, std::string mode, bool head)
{
    return true;
}


std::vector<std::string> csv_t::getRow(int row)
{
    int idx;
    if (row < 0) {
        idx = (int) _data.size() + row;
    } else
        idx = row;
    return _data.at(idx);
}

std::string csv_t::getElement(std::string _col, int row)
{
    uint16_t col = _head[_col];
    return getRow(row).at(col);
}

std::string csv_t::getElement(int col, int row)
{
    return getRow(row).at(col);
}

std::map<std::string, std::string> csv_t::getElements(int row)
{
    std::map<std::string, std::string> data;

    for (std::map<std::string, std::uint16_t>::iterator it = _head.begin();
         it != _head.end(); ++it) {
        data[it->first] = _data[row][it->second];
    }

    return data;
}

unsigned int csv_t::nrows()
{
    return _data.size();
}

std::vector<std::vector<std::string> > csv_t::getData(int r1, int r2)
{
    std::vector<std::vector<std::string> > data;
    if ((r1 < 0 && r2 > 0) || (r1 > 0 && r2 < -1)) {
        throw std::invalid_argument("given r1 < 0 but r2 >0");
    } else if (r1 > 0 && r2 == -1) {
        r2 = _data.size();
    } else {
        return _data;
    }

    iter_range(_data, i, (unsigned int) r1, (unsigned int) r2)
    {
        data.push_back(_data[i]);
    }
    return data;
}


csv_t csv_t::filter(std::string head, std::string value)
{
    csv_t newcsv;

    std::vector<std::vector<std::string> > data;
    int idx = _head[head];
    iter(_data, i)
    {
        if (_data[i][idx].compare(value) == 0) {
            data.push_back(_data[i]);
        }
    }

    newcsv._data = data;
    newcsv._filename = _filename;
    newcsv._head = _head;
    newcsv._mode = _mode;

    return newcsv;
}

std::vector<std::string> csv_t::getColumn(std::string head)
{
    int idx = _head[head];
    std::vector<std::string> cols;

    iter(_data, i) { cols.push_back(_data[i][idx]); }

    return cols;
}
