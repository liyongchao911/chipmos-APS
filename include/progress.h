#ifndef __PROGRESS_H__
#define __PROGRESS_H__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef WIN32
#include<winsock.h>
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/select.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct __progresss_bar_thread_data_t {
    const char *ip;
    int port;
} progresss_bar_thread_data_t;

typedef struct __progress_bar_attr_t {
    int server_socket_fd;
    int number_of_connection;
    int *client_sockfds;
    fd_set *fdset;
    int nfds;
    struct sockaddr_in server_info, client_info;
} progress_bar_attr_t;

#define progress_bar_attr_initialization()                                \
    (progress_bar_attr_t)                                                 \
    {                                                                     \
        .number_of_connection = 0, .client_sockfds = NULL, .fdset = NULL, \
        .nfds = -1                                                        \
    }

progress_bar_attr_t *create_progress_bar_attr(int number_of_connection,
                                              const char *ip,
                                              int port);

void *run_progress_bar_server(void *data);


void *accept_connection(void *);

int create_client_connection(const char *ip, int port);

void delete_attr(progress_bar_attr_t **attr);

void setup_fdset(progress_bar_attr_t *attr);

#ifdef __cplusplus
}
#endif

#endif
