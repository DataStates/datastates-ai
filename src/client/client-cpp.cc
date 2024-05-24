#include "dstatesai/client-cpp.h"
#include <thallium/serialization/stl/pair.hpp>
#include <thallium/serialization/stl/string.hpp>
#include <thallium/serialization/stl/unordered_map.hpp>
#include <thallium/serialization/stl/unordered_set.hpp>
#include <thallium/serialization/stl/vector.hpp>

#define __DEBUG
#include "debug.hpp"
using namespace std::chrono;

namespace dstatesai {

rpc_client::rpc_client(const std::vector<std::string> &servers,
                       const std::vector<int> &provider_ids)
    : // TODO gracefully fall back or provide an error if this environment variable needs to be set
      engine(getenv("THALLIUM_NETWORK"), THALLIUM_CLIENT_MODE) {

  // create RPC handles, these can be used with any provider
  _store_meta = engine.define("store_meta");
  _get_prefix = engine.define("get_prefix");
  _get_composition = engine.define("get_composition");
  _store_layers = engine.define("store_layers");
  _read_layers = engine.define("read_layers");
  _update_ref_counter = engine.define("update_ref_counter");
  _shutdown = engine.define("shutdown");

  // create the providers handles
  int iter = 0;
  for (auto &server : servers) {
    tl::endpoint endp = engine.lookup(server);
    providers.emplace_back(tl::provider_handle(endp, provider_ids[iter]));
    iter++;
    INFO("client connected " << server);
  }
}

void rpc_client::append_to_timestamp_map(timestamp_map_t &profile_time_stamps,
                                         timestamp_t &timestamp, std::string &function_string) {
  if (profile_time_stamps.find(function_string) == profile_time_stamps.end())
    profile_time_stamps[function_string] = std::vector<uint64_t>{timestamp.first, timestamp.second};
  else
    profile_time_stamps[function_string].insert(profile_time_stamps[function_string].end(),
                                                {timestamp.first, timestamp.second});
}

void rpc_client::append_to_timestamp_map(timestamp_map_t &profile_time_stamps,
                                         std::vector<uint64_t> &timestamp,
                                         std::string &function_string) {
  if (profile_time_stamps.find(function_string) == profile_time_stamps.end())
    profile_time_stamps[function_string] =
        std::vector<uint64_t>(timestamp.begin(), timestamp.end());
  else
    profile_time_stamps[function_string].insert(profile_time_stamps[function_string].end(),
                                                timestamp.begin(), timestamp.end());
}

bool rpc_client::store_meta(const digraph_t &g, const composition_t &comp, const float val_acc,
                            timestamp_map_t &profile_time_stamps) {
  timestamp_t ret = _store_meta.on(get_provider(g.id))(g, comp, val_acc);
  std::string function_str = std::string(__func__);
  append_to_timestamp_map(profile_time_stamps, ret, function_str);
  return true;
}

bool rpc_client::store_layers(const model_id_t &id, const vertex_list_t &layer_id,
                              std::vector<segment_t> &segments,
                              timestamp_map_t &profile_time_stamps) {

  std::vector<size_t> layer_size;
  for (auto &segment : segments)
    layer_size.emplace_back(segment.second);

  tl::bulk bulk = engine.expose(segments, tl::bulk_mode::read_write);
  std::vector<uint64_t> ts = _store_layers.on(get_provider(id))(id, layer_id, layer_size, bulk);
  std::string function_str = std::string(__func__);
  append_to_timestamp_map(profile_time_stamps, ts, function_str);
  return true;
}

composition_t &rpc_client::get_composition(const model_id_t &id,
                                           timestamp_map_t &profile_time_stamps) {
  std::unique_lock<tl::mutex> lock(cache_lock);
  auto it = comp_cache.find(id);
  std::pair<composition_t, timestamp_t> comp_result = _get_composition.on(get_provider(id))(id);
  if (it == comp_cache.end())
    it = comp_cache.emplace_hint(it, id, comp_result.first);
  else if (it->second.size() == 0) {
    it->second = comp_result.first;
  }
  std::string function_str = std::string(__func__);
  append_to_timestamp_map(profile_time_stamps, comp_result.second, function_str);
  return it->second;
}

bool rpc_client::read_layers(const model_id_t &id, const vertex_list_t &layer_id,
                             std::vector<segment_t> &segment_list, std::vector<uint64_t> &owners,
                             timestamp_map_t &profile_time_stamps) {
  struct req_info_t {
    vertex_list_t layer_id;
    std::vector<segment_t> segments;
  };
  std::unordered_map<model_id_t, req_info_t> owner_map;
  std::vector<tl::bulk> bulks;
  std::vector<tl::async_response> reps;
  /*auto &comp = get_composition(id, profile_time_stamps);
  if (comp.empty()){
      return false;
  }*/
  for (int i = 0; i < layer_id.size(); i++) {
    auto owner = owners[i];
    // auto [owner, size] = comp[layer_id[i]];
    auto &e = owner_map[owner];
    e.layer_id.emplace_back(layer_id[i]);
    e.segments.emplace_back(segment_list[i]);
  }
  for (auto &e : owner_map) {
    bulks.emplace_back(engine.expose(e.second.segments, tl::bulk_mode::write_only));
    reps.emplace_back(
        _read_layers.on(get_provider(e.first)).async(e.second.layer_id, e.first, bulks.back()));
  }

  std::string function_str = std::string(__func__);
  bool result = true;
  for (auto &rep : reps) {
    std::vector<uint64_t> temp = rep.wait();
    append_to_timestamp_map(profile_time_stamps, temp, function_str);
    // result = result && rep.wait();
  }
  return result;
}

bool rpc_client::update_ref_counter(const model_id_t &id, int value) {
  std::unordered_map<model_id_t, vertex_list_t> req_args;
  std::vector<tl::async_response> reps;
  timestamp_map_t temp;
  auto &comp = get_composition(id, temp);
  if (comp.empty())
    return false;
  for (auto &e : comp)
    req_args[e.second.first].emplace_back(e.first);
  for (auto &e : req_args)
    reps.push_back(_update_ref_counter.on(get_provider(id)).async(e.first, e.second, value));
  bool result = true;
  for (auto &rep : reps)
    result = result && rep.wait();
  return result;
}

prefix_t rpc_client::get_prefix(const digraph_t &child, timestamp_map_t &profile_time_stamps) {
  std::string function_str = std::string(__func__);
  prefix_t max_result;
  std::vector<tl::async_response> requests;
  for (auto &provider : providers)
    requests.emplace_back(_get_prefix.on(provider).async(child));
  for (auto &request : requests) {
    std::pair<prefix_t, timestamp_t> result = request.wait();
    if (result.first.second.size() > max_result.second.size())
      std::swap(result.first, max_result);
    append_to_timestamp_map(profile_time_stamps, result.second, function_str);
  }
  return max_result;
}

int rpc_client::shutdown() {
  INFO("client issued shutdown!!");
  for (auto const &i : providers) {
    engine.shutdown_remote_engine(i);
  }
  return 0;
}

} // namespace dstatesai
