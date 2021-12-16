#include "include/arg_parser.h"

using namespace std;

static string join(vector<string> str_list, string token)
{
    string ret;
    int size = str_list.size() - 1;
    if (size > 0) {
        for (int i = 0; i < size; ++i) {
            ret += str_list[i] + token;
        }
        ret += str_list[size];
    }
    return ret;
}

vector<string> argument_parser_t::_split_eq(string str)
{
    vector<string> tokens;
    size_t pos = 0, next_pos;
    next_pos = str.find("="s);
    tokens.push_back(str.substr(pos, next_pos));
    if (next_pos != string::npos) {
        ++next_pos;
        pos = next_pos;
        tokens.push_back(str.substr(pos));
    }
    return tokens;
}

void argument_parser_t::_check_and_output()
{
    set<arg_element_t *> error_elements;
    for (auto it = _arg_descritpions.begin(); it != _arg_descritpions.end();
         ++it) {
        if (it->second->_element &&
            error_elements.count(it->second->_element) == 0) {
            if (it->second->_type != ARG_ERROR &&
                it->second->_type != ARG_NONE &&
                it->second->_element->_value.length() == 0) {
                cerr << "The argument " << it->second->_element->_name
                     << " need a value\n";
                error_elements.insert(it->second->_element);
            }
        }
    }
}

void argument_parser_t::parse_argument_list(int argc, const char *argv[])
{
    map<string, string> flag_to_val;
    for (int i = 1; i < argc; ++i) {
        vector<string> tokens = _split_eq(argv[i]);
        if (tokens.size() >= 2) {
            string first = tokens[0];
            tokens.erase(tokens.begin());
            flag_to_val[first] = tokens[0];
        } else {
            flag_to_val[tokens[0]] = ""s;
        }
    }

    for (auto it = flag_to_val.begin(); it != flag_to_val.end(); ++it) {
        try {
            arg_descritpion_t *dscrpt = _arg_descritpions.at(it->first);
            if (dscrpt->_element) {
                cerr << "You previously set this argument and its value is "
                     << dscrpt->_element->_value << " The new value is "
                     << it->second;
            } else {
                dscrpt->_element = new arg_element_t();
                dscrpt->_element->_value = it->second;
                dscrpt->_element->_name = it->first;
            }
        } catch (out_of_range &e) {
            cerr << "Warnning : The argument " << it->first << " is invalid"
                 << endl;
        }
    }

    _check_and_output();
}

void argument_parser_t::print_arg_description()
{
    for (unsigned int i = 0; i < _description_list.size(); ++i) {
        printf("%s : \n\t %s\n", _description_list[i]->_name.c_str(),
               _description_list[i]->_description.c_str());
    }
}

void argument_parser_t::add_args(arg_descritpion_t dscrpt,
                                 vector<string> arg_name)
{
    arg_descritpion_t *dscrp = new arg_descritpion_t();
    *dscrp = dscrpt;
    _description_list.push_back(dscrp);
    string joined_arg_name = join(arg_name, " ,");
    dscrp->_name = joined_arg_name;
    dscrp->_element = NULL;

    for (auto name : arg_name) {
        if (_arg_descritpions.count(name) == 0) {
            _arg_descritpions[name] = dscrp;
            ++_number_of_set_arguments;
        } else {
            throw invalid_argument("You set the arg name " + name + " twice");
        }
    }
}

void argument_parser_t::_init()
{
    _number_of_set_arguments = 0;
}
