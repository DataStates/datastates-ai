#ifndef __LIB_CLIENT_BACKEND_HPP
#define __LIB_CLIENT_BACKEND_HPP
#include <cstdlib>
#include <memory>
#include <memory_resource>
#include <string>

#include <dstatesai/client-cpp.h>
#include <nanobind/ndarray.h>
using tensor_list_t = std::vector<nanobind::ndarray<>>;
using uint64_list_t = std::vector<uint64_t>;

namespace dstatesai {
class py_backend {

  std::unique_ptr<rpc_client> client;
  std::vector<std::unique_ptr<std::pmr::memory_resource>> mem_resources;
  std::unique_ptr<char[]> mem_buffer;

public:
  py_backend(std::string servers, size_t size);

  inline int getTimeStamps(uint64_t *ts);
  inline int getTimeStampsByKey(uint64_t *ts, char *function_string);
  inline int getNumTimeStamps();
  inline int getNumTimeStampsByKey(char *function_string);
  inline void clearTimeStamps();
  inline bool clearTimeStampsByKey(char *function_string);

  int save_layers(tensor_list_t &tensors, uint64_t &model_ids, uint64_list_t &layer_ids);
  int load_layers(tensor_list_t &tensors, uint64_t &model_ids, uint64_list_t &layer_ids,
                  uint64_list_t &layer_owners);
  bool store_meta(uint64_t id, uint64_list_t &edges, int m, uint64_list_t &layer_ids,
                  uint64_list_t &layer_owners, uint64_list_t &sizes, int n, const float val_acc);
  int get_composition(uint64_t id, uint64_list_t &layer_ids, uint64_list_t &layer_owners, int n);
  int get_prefix(uint64_list_t &edges, int n, uint64_list_t &result);
  bool update_ref_counter(uint64_t id, int value);
  int shutdown();
};
} // namespace dstatesai
#endif
