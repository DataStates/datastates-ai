#include "server.hpp"

#include <getopt.h>

#define __DEBUG
#include "debug.hpp"

static struct option long_ops[] = {
    {"connection", required_argument, 0, 'c'},
    {"provider", required_argument, 0, 'p'},
    {"threads", required_argument, 0, 't'},
    {"buffer-size", required_argument, 0, 'b'},
    {0, 0, 0, 0}
};

void exit_with_usage() {
    std::cerr << "Usage: launcher --connection <conn_string> [--provider <id> (default 0)] [--threads <thread_no> (default 1)] [--buffer_size <buff_size> (default 1 GiB)]" << std::endl
    << "Note: shortcuts (-c, -p, -t, -b) are also allowed" << std::endl
    << "See the Thallium documentation for more details." << std::endl;
    exit(-1);
}

int main(int argc, char **argv) {
    std::string thallium_conn;
    unsigned int provider_id = 0, thread_no = 1;
    size_t buff_size = dstates::ai::DEFAULT_BUFFER_SIZE;

    int ret, args_set = 0;
    while ((ret = getopt_long(argc, argv, "c:p:t:", long_ops, NULL)) != -1)
        if (ret == 'c')
	    thallium_conn = optarg;
	else if (ret == 'p' && sscanf(optarg, "%u", &provider_id) != 1)
	    exit_with_usage();
	else if (ret == 't' && sscanf(optarg, "%u", &thread_no) != 1)
	    exit_with_usage();
	else if (ret == 'b' && sscanf(optarg, "%lu", &buff_size) != 1)
	    exit_with_usage();

    if (thallium_conn.empty())
	exit_with_usage();

    tl::engine model_server_engine(thallium_conn, THALLIUM_SERVER_MODE);
    dstates::ai::model_server_t model_server(model_server_engine, provider_id, thread_no, buff_size);
    INFO("Model server listening at: " << model_server_engine.self());

    return 0;
}
