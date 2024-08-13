#ifndef __DSTATES_AI_PYCLIENT_HPP
#define __DSTATES_AI_PYCLIENT_HPP

#include "dstates/ai/client.hpp"

#include <cstdlib>
#include <memory>
#include <memory_resource>
#include <string>
#include <nanobind/ndarray.h>

using tensor_list_t = std::vector<nanobind::ndarray<>>;
using uint64_list_t = std::vector<uint64_t>;
using string_list_t = std::vector<std::string>;

namespace dstates::ai {
class py_backend {
    std::unique_ptr<char[]> mem_buffer;
    std::unique_ptr<std::pmr::memory_resource> buffer_resource, pool;
    std::unique_ptr<rpc_client> client;

public:
    py_backend(const std::string &thallium_cfg, const std::vector<std::string> &servers,
	       size_t buffer_size = DEFAULT_BUFFER_SIZE);

    bool save_layers(tensor_list_t &tensors, uint64_t model_id, uint64_list_t &layer_ids);
    bool load_layers(tensor_list_t &tensors, uint64_t model_id, uint64_list_t &layer_ids,
                    uint64_list_t &layer_owners);
    bool store_meta(uint64_t id, uint64_list_t &edges, uint64_list_t &layer_ids,
                    uint64_list_t &layer_owners, uint64_list_t &sizes, const float val_acc);
    composition_t get_composition(uint64_t model_id);
    prefix_t get_prefix(uint64_list_t &edges);
    bool update_ref_counter(uint64_t id, int value);
    int shutdown();
};
} // namespace dstates::ai

#endif
