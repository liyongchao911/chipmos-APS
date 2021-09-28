#include "include/progress.h"
#include <asm-generic/socket.h>
#include <netinet/in.h>
#include <pthread.h>
#include <stdbool.h>
#include <sys/socket.h>
#include <unistd.h>

typedef struct entry_t {
    int channel_id;
    int total_times;
    int current_times;
    double progress;
    double value;
} entry_t;

void progress_bar(double progress)
{
    printf("[");
    for (int i = 0, max = progress - 2; i < max; i += 2) {
        printf("= ");
    }
    if (progress < 100.0)
        printf("=>");
    else
        printf("= ");
    for (int i = progress + 2; i <= 100; i += 2) {
        printf("· ");
    }

    printf("]%03.2f%%  ", progress);
}

void erase_last_line()
{
    printf(
        "\033[A\r                                                              "
        "                                                      \r");
}

void form_entry(char *buffer, entry_t *ent)
{
    sscanf(buffer, "%d/%d-%lf\n", &ent->current_times, &ent->total_times,
           &ent->value);
    ent->current_times += 1;
    ent->progress =
        (double) ent->current_times * 100 / (double) ent->total_times;
}

void *run_progress_bar_server(void *_data)
{
    progress_bar_attr_t *attr = (progress_bar_attr_t *) _data;
    char input_buffer[1024];
    entry_t *entries =
        (entry_t *) malloc(sizeof(entry_t) * attr->number_of_connection - 1);
    for (int i = 0; i < attr->number_of_connection; ++i) {
        entries[i] = (entry_t){.channel_id = i,
                               .total_times = 1,
                               .current_times = 0,
                               .progress = 0,
                               .value = 0};
    }

    for (int i = 0; i < attr->number_of_connection - 1; ++i)
        printf("\n");

    struct timeval timeout;
    bool running = true;
    while (running) {
        setup_fdset(attr);
        timeout.tv_sec = 10;
        timeout.tv_usec = 0;

        // printf("Wait for select\n");
        int ret = select(attr->nfds + 1, attr->fdset, NULL, NULL, &timeout);
        if (ret < 0) {
            perror("Select error ");
        } else if (ret == 0) {
            free(entries);
            pthread_exit(0);
        } else {
            for (int i = 0; i < attr->number_of_connection; ++i) {
                if (FD_ISSET(attr->client_sockfds[i], attr->fdset)) {
                    ret = recv(attr->client_sockfds[i], input_buffer, 1024, 0);
                    if (ret < 0) {
                        perror("recv error");
                    } else {
                        if (strncmp(input_buffer, "close", 5) == 0) {
                            running = false;
                            // printf("Close the server\n");
                        } else {
                            form_entry(input_buffer, entries + i);
                        }
                    }
                }
            }
            for (int i = 0; i < attr->number_of_connection - 1; ++i) {
                erase_last_line();
            }
            for (int i = 0; i < attr->number_of_connection - 1; ++i) {
                printf("[%d]", entries[i].channel_id);
                progress_bar(entries[i].progress);
                printf("(%.3lf)\n", entries[i].value);
            }
        }
    }
    printf("Close the server\n");
    pthread_exit(NULL);
}

void setup_fdset(progress_bar_attr_t *attr)
{
    FD_ZERO(attr->fdset);
    for (int i = 0; i < attr->number_of_connection; ++i) {
        FD_SET(attr->client_sockfds[i], attr->fdset);
    }
}

void delete_attr(progress_bar_attr_t **_attr)
{
    progress_bar_attr_t *attr = *_attr;
    for (int i = 0; i < attr->number_of_connection; ++i) {
        close(attr->client_sockfds[i]);
    }
    close(attr->server_socket_fd);
    free(attr->client_sockfds);
    free(attr->fdset);
    free(attr);

    *_attr = NULL;
}

progress_bar_attr_t *create_progress_bar_attr(int number_of_connection,
                                              const char *ip,
                                              int port)
{
    progress_bar_attr_t *attr;
    attr = (progress_bar_attr_t *) malloc(sizeof(progress_bar_attr_t));

    *attr = progress_bar_attr_initialization();

    // check number of connection > 1024 ?
    attr->number_of_connection = number_of_connection;

    attr->client_sockfds = (int *) malloc(sizeof(int) * number_of_connection);

    attr->fdset = (fd_set *) malloc(sizeof(fd_set) * number_of_connection);

    memset(&attr->server_info, 0, sizeof(struct sockaddr_in));
    memset(&attr->client_info, 0, sizeof(struct sockaddr_in));

    attr->server_socket_fd = socket(AF_INET, SOCK_STREAM, 0);
    int optval = 1;
    setsockopt(attr->server_socket_fd, SOL_SOCKET, SO_REUSEADDR, &optval,
               sizeof(optval));

    attr->server_info = (struct sockaddr_in){
        .sin_family = AF_INET,
        .sin_port = htons(port),
    };
    attr->server_info.sin_addr.s_addr = inet_addr(ip);

    int ret =
        bind(attr->server_socket_fd, (struct sockaddr *) &(attr->server_info),
             sizeof attr->server_info);
    if (ret == -1) {
        perror("Bind server error\n");
    }
    listen(attr->server_socket_fd, number_of_connection);

    return attr;
}

void *accept_connection(void *_attr)
{
    progress_bar_attr_t *attr = (progress_bar_attr_t *) _attr;
    for (int i = 0; i < attr->number_of_connection; ++i) {
        int addrlen = 0;
        int client_fd =
            accept(attr->server_socket_fd,
                   (struct sockaddr *) &attr->client_info, &addrlen);
        attr->client_sockfds[i] = client_fd;
    }

    for (int i = 0; i < attr->number_of_connection; ++i) {
        if (attr->client_sockfds[i] > attr->nfds) {
            attr->nfds = attr->client_sockfds[i];
        }
    }
    char message[] = "Accepting all connections\n";
    write(1, message, sizeof(message));
    return NULL;
}


int create_client_connection(const char *ip, int port)
{
    int sockfd = 0;
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in info;
    memset(&info, 0, sizeof(info));
    info.sin_family = AF_INET;
    info.sin_addr.s_addr = inet_addr(ip);
    info.sin_port = htons(port);

    int err = connect(sockfd, (struct sockaddr *) &info, sizeof(info));
    if (err == -1) {
        printf("Connection error\n");
    }
    return sockfd;
}
