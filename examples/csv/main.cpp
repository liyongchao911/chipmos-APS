#include <execinfo.h>
#include <signal.h>
#include <include/common.h>
#include <include/csv.h>
#include <iostream>

using namespace std;

#define BUF_SIZE 100000

void *buf[BUF_SIZE];

void sig_handler(int signum){
    printf("Segmentation fault\n");
    char ** strings = NULL;
    int nptrs;
    nptrs = backtrace(buf, BUF_SIZE); 
    strings = backtrace_symbols(buf, nptrs);
    
    if(strings == NULL){
        perror("backtrace_symbols error\n");
    }

    for(int j = 0; j < nptrs; ++j){
        printf("%s\n", strings[j]); 
    }
    free(strings);
    exit(-1);
}

int main(int argc, char const *argv[]){
    // signal(SIGSEGV, sig_handler);
    if(argc < 3){
        printf("usage : CSVFILE row_idx\n");
        exit(-1);
    }
    csv file(argv[1], "r", false);
    file.read(false, 3, 14);

    vector<string> data = file.getRow(atoi(argv[2]));
    for(unsigned int i = 0; i < data.size(); ++i){
        cout<< data[i]<< " ";
    }
    cout<<endl;
    
    // file.setHeader("wlot_lot_number", "lot_id", true);

    // map<string, string> data = file.getElements(atoi(argv[2]));

    // for(map<string, string>::iterator it = data.begin(); it != data.end(); ++it){
    //     cout<<it->first<<" : "<<it->second<<endl; 
    // }
    
}
