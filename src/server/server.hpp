#ifndef __DSTATES_AI_SERVER_HPP
#define __DSTATES_AI_SERVER_HPP

#include "dstates/ai/types.hpp"

#include <memory_resource>
#include <thallium.hpp>

namespace tl = thallium;
using namespace std::chrono;

namespace dstates::ai {

class model_server_t : public tl::provider<model_server_t> {
    struct model_info_t {
	std::vector<digraph_t>::iterator index;
	composition_t composition;
	float val_acc;
	model_info_t(const std::vector<digraph_t>::iterator &idx, const composition_t &comp,
                     const float &acc)
        : index(idx), composition(comp), val_acc(acc) {}
    };

    struct rdma_buffer_t {
	tl::mutex buffer_lock;
	char *buffer;
	std::unique_ptr<std::pmr::memory_resource> buffer_resource, pool;
    };

    struct layer_t {
	segment_t segment;
	size_t ref_count;
	layer_t(size_t size, void *ptr) : segment{ptr, size}, ref_count(0) {}
    };

    struct layer_info_t {
	tl::mutex layer_lock;
	std::unordered_map<model_id_t, layer_t> owner_map;
    };

    tl::managed<tl::pool> request_pool;
    std::vector<tl::managed<tl::xstream>> ess;
    std::vector<digraph_t> graph_store;
    std::unordered_map<uint64_t, model_info_t> graph_info;
    std::unordered_map<vertex_t, layer_info_t> layer_store;
    rdma_buffer_t rdma_segments;
    tl::mutex store_lock;
    std::vector<tl::remote_procedure> procedures;
    std::string policy;
    size_t pinned_buffer_size;

public:
    model_server_t(tl::engine &e, uint16_t provider_id = 0, uint32_t num_procs = 1,
		   size_t buffer_size = DEFAULT_BUFFER_SIZE,
                   std::string const &server_policy = std::string("map"));
    ~model_server_t();
    bool store_meta(const digraph_t &g, const composition_t &comp, const float val_acc);
    prefix_t get_prefix(const digraph_t &child);
    composition_t get_composition(const model_id_t &id);
    void store_layers(const tl::request &req, const model_id_t &id, const vertex_list_t &layer_id,
                      const std::vector<size_t> &layer_size, tl::bulk &layer_bulk);
    void read_layers(const tl::request &req, const vertex_list_t &layer_id, const model_id_t &owner,
                     tl::bulk &layer_bulk);
    bool update_ref_counter(const model_id_t &owner, const vertex_list_t &layer_id, int value);
    int shutdown();
    void rdma_buffers_init(tl::engine &e);
};
} // namespace dstates::ai

#endif //__DSTATES_AI_SERVER
