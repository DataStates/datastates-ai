#ifndef DSTATESAI_MODEL_SERVER_H
#define DSTATESAI_MODEL_SERVER_H

#include "dstatesai/types.h"
#include <memory_resource>
#include <thallium.hpp>

namespace tl = thallium;
using namespace std::chrono;

namespace dstatesai {

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
    std::vector<std::unique_ptr<std::pmr::memory_resource>> dbg;
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

public:
  model_server_t(tl::engine &e, uint16_t provider_id = 0, uint32_t num_procs = 1,
                 std::string const &server_policy = std::string("map"));
  ~model_server_t();
  timestamp_t store_meta(const digraph_t &g, const composition_t &comp, const float val_acc);
  std::pair<prefix_t, timestamp_t> get_prefix(const digraph_t &child);
  std::pair<composition_t, timestamp_t> get_composition(const model_id_t &id);
  void store_layers(const tl::request &req, const model_id_t &id, const vertex_list_t &layer_id,
                    const std::vector<size_t> &layer_size, tl::bulk &layer_bulk);
  void read_layers(const tl::request &req, const vertex_list_t &layer_id, const model_id_t &owner,
                   tl::bulk &layer_bulk);
  bool update_ref_counter(const model_id_t &owner, const vertex_list_t &layer_id, int value);
  bool clear_timestamps(const model_id_t &id);
  int shutdown();
  void rdma_buffers_init(tl::engine &e);
  std::vector<uint64_t> get_timestamps(const model_id_t &id);
};
} // namespace dstatesai

#endif //__MODEL_SERVER
