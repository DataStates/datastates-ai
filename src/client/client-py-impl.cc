#include "client-py-impl.h"
#include "debug.hpp"
#include "debug_memory_resource.h"

#include "dstatesai/client-cpp.h"
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>
#include <map>
#include <nanobind/nanobind.h>
#include <numeric>
#include <sstream>
#include <vector>

namespace dstatesai {
namespace nb = nanobind;

timestamp_map_t profile_time_stamps;

static std::vector<std::string> split(std::string s, char c) {
  std::stringstream test(s);
  std::string segment;
  std::vector<std::string> seglist;
  while (std::getline(test, segment, c))
    seglist.push_back(segment);
  INFO("NUM items: " << seglist.size());
  return seglist;
}

inline int py_backend::getTimeStamps(uint64_t *ts) {
  int k = 0;
  for (auto &kv : profile_time_stamps) {
    std::vector<uint64_t> ts_temp = kv.second;
    for (int i = 0; i < ts_temp.size(); ++i) {
      ts[k] = ts_temp[i];
      k++;
    }
  }
  return k;
}
inline int py_backend::getTimeStampsByKey(uint64_t *ts, char *function_string) {
  if (profile_time_stamps.find(std::string(function_string)) == profile_time_stamps.end()) {
    return -1;
  }

  std::vector<uint64_t> ts_temp = profile_time_stamps[std::string(function_string)];
  int k = 0;
  for (int i = 0; i < ts_temp.size(); ++i) {
    ts[k] = ts_temp[i];
    ++k;
  }
  return k;
}
inline int py_backend::getNumTimeStamps() {
  int count = 0;
  for (auto &kv : profile_time_stamps)
    count += kv.second.size();
  return count;
}
inline int py_backend::getNumTimeStampsByKey(char *function_string) {
  if (profile_time_stamps.find(std::string(function_string)) == profile_time_stamps.end())
    return -1;
  return profile_time_stamps[std::string(function_string)].size();
}
inline void py_backend::clearTimeStamps() {
  for (auto &kv : profile_time_stamps) {
    kv.second.clear();
  }
}
inline bool py_backend::clearTimeStampsByKey(char *function_string) {
  if (profile_time_stamps.find(std::string(function_string)) == profile_time_stamps.end())
    return false;
  profile_time_stamps[std::string(function_string)].clear();
  return true;
}

py_backend::py_backend(std::string servers, size_t size) {
  std::vector<int> providers;
  providers.resize(size);
  std::iota(providers.begin(), providers.end(), 0);
  std::vector<std::string> global_servers;
  try {
    global_servers = split(servers, '[');
  } catch (...) {
    ERROR("ERROR");
  }
  INFO("client(cpp)");
  for (auto i : global_servers) {
    INFO(i);
  }
  INFO("");

  client = std::make_unique<rpc_client>(global_servers, providers);
  mem_buffer = std::make_unique<char[]>(client_pinned_buffer_size);
  mem_resources.emplace_back(std::make_unique<std::pmr::monotonic_buffer_resource>(
      mem_buffer.get(), client_pinned_buffer_size, std::pmr::null_memory_resource()));
  mem_resources.emplace_back(
      std::make_unique<std::pmr::unsynchronized_pool_resource>(mem_resources.back().get()));
  mem_resources.emplace_back(
      std::make_unique<debug_resource>("pool", mem_resources.back().get(), mem_buffer.get()));
}

int py_backend::save_layers(tensor_list_t &tensors, uint64_t &model_id, uint64_list_t &layer_ids) {
  std::pmr::vector<char> temp{mem_resources.back().get()};
  std::pmr::vector<char> ptrs[tensors.size()];
  for (int i = 0; i < tensors.size(); ++i)
    ptrs[i] = temp;

  std::vector<segment_t> segments;
  if (tensors[0].device_type() == nb::device::cpu::value) {
    for (auto &t : tensors)
      segments.emplace_back((void *)t.data(), (size_t)(t.size() * sizeof(t.dtype())));
  } else {
    int iter = 0;
    for (auto &t : tensors) {
      auto size = (size_t)(t.size() * sizeof(t.dtype()));
      ptrs[iter].resize(size);
      cudaMemcpy((char *)ptrs[iter].data(), (char *)t.data(), size, cudaMemcpyDeviceToHost);
      segments.emplace_back((void *)ptrs[iter].data(), size);
      iter++;
    }
  }
  auto ret = client->store_layers(model_id, layer_ids, segments, profile_time_stamps);
  return ret;
}

int py_backend::load_layers(tensor_list_t &tensors, uint64_t &model_id, uint64_list_t &layer_ids,
                            uint64_list_t &layer_owners) {
  std::pmr::vector<char> temp{mem_resources.back().get()};
  std::pmr::vector<char> ptrs[tensors.size()];
  std::vector<segment_t> segments;
  for (int i = 0; i < tensors.size(); ++i)
    ptrs[i] = temp;
  bool is_gpu = (tensors[0].device_type() != nb::device::cpu::value);
  if (!is_gpu) {
    for (auto &t : tensors)
      segments.emplace_back((void *)t.data(), (size_t)(t.size() * sizeof(t.dtype())));
  } else {
    int iter = 0;
    for (auto &t : tensors) {
      auto size = (size_t)(t.size() * sizeof(t.dtype()));
      ptrs[iter].resize(size);
      segments.emplace_back((void *)ptrs[iter].data(), size);
      iter++;
    }
  }
  int ret = client->read_layers(model_id, layer_ids, segments, layer_owners, profile_time_stamps);
  if (is_gpu) {
    int iter = 0;
    for (auto &t : tensors) {
      cudaMemcpy((char *)t.data(), (char *)segments[iter].first, segments[iter].second,
                 cudaMemcpyHostToDevice);
      ++iter;
    }
  }
  return ret;
}

bool py_backend::store_meta(uint64_t id, uint64_list_t &edges, int m, uint64_list_t &layer_ids,
                            uint64_list_t &layer_owners, uint64_list_t &sizes, int n,
                            const float val_acc) {

  if (n < 2)
    return false;

  digraph_t g;
  g.root = edges[0];
  g.id = id;
  for (int i = 0; i < m; i += 2) {
    g.out_edges[edges[i]].insert(edges[i + 1]);
    g.in_degree[edges[i + 1]]++;
  }
  composition_t comp;
  for (int i = 0; i < n; i++)
    comp.emplace(layer_ids[i], std::make_pair(layer_owners[i], sizes[i]));

  bool ret = client->store_meta(g, comp, val_acc, profile_time_stamps);
  return ret;
}

int py_backend::get_composition(uint64_t id, uint64_list_t &layer_ids, uint64_list_t &layer_owners,
                                int n) {
  auto &comp = client->get_composition(id, profile_time_stamps);
  int count = 0;
  for (int i = 0; i < n; i++) {
    auto it = comp.find(layer_ids[i]);
    if (it != comp.end()) {
      layer_owners[i] = it->second.first;
      count++;
    } else {
      layer_owners[i] = 0;
    }
  }
  return count;
}

int py_backend::get_prefix(uint64_list_t &edges, int n, uint64_list_t &result) {
  if (n < 2) {
    return 0;
  }
  digraph_t g;
  g.root = edges[0];
  for (int i = 0; i < n; i += 2) {
    g.out_edges[edges[i]].insert(edges[i + 1]);
    g.in_degree[edges[i + 1]]++;
  }
  prefix_t reply = client->get_prefix(g, profile_time_stamps);
  uint64_t id = reply.first;
  std::copy(reply.second.begin(), reply.second.end(), result.data());
  result[reply.second.size()] = id;
  return reply.second.size();
}

bool py_backend::update_ref_counter(uint64_t id, int value) {
  return client->update_ref_counter(id, value);
}

int py_backend::shutdown() { return client->shutdown(); }
} // namespace dstatesai
