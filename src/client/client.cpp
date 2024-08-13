#include "dstates/ai/client.hpp"
#include <thallium/serialization/stl/pair.hpp>
#include <thallium/serialization/stl/string.hpp>
#include <thallium/serialization/stl/unordered_map.hpp>
#include <thallium/serialization/stl/unordered_set.hpp>
#include <thallium/serialization/stl/vector.hpp>

#define __DEBUG
#include "debug.hpp"

using namespace std::chrono;
logger_state_t logger_state;

namespace dstates::ai {
    rpc_client::rpc_client(const std::string &thallium_cfg,
			   const std::vector<std::string> &servers,
			   const std::vector<int> &provider_ids) : engine(thallium_cfg, THALLIUM_CLIENT_MODE) {
    // create RPC handles, these can be used with any provider
    _store_meta = engine.define("store_meta");
    _get_prefix = engine.define("get_prefix");
    _get_composition = engine.define("get_composition");
    _store_layers = engine.define("store_layers");
    _read_layers = engine.define("read_layers");
    _update_ref_counter = engine.define("update_ref_counter");
    _shutdown = engine.define("shutdown");

    // create the providers handles
    for (int i = 0; i < servers.size(); i++) {
	tl::endpoint endp = engine.lookup(servers[i]);
	providers.emplace_back(tl::provider_handle(endp, provider_ids[i]));
	INFO("client connected " << servers[i]);
    }
}

bool rpc_client::store_meta(const digraph_t &g, const composition_t &comp, const float val_acc) {
    return _store_meta.on(get_provider(g.id))(g, comp, val_acc);
}

bool rpc_client::store_layers(const model_id_t &id, const vertex_list_t &layer_id,
			      const std::vector<segment_t> &segments) {
    std::vector<size_t> layer_size(segments.size());
    for (int i = 0; i < segments.size(); i++)
	layer_size[i] = segments[i].second;

    tl::bulk bulk = engine.expose(segments, tl::bulk_mode::read_write);
    return _store_layers.on(get_provider(id))(id, layer_id, layer_size, bulk);
}

composition_t &rpc_client::get_composition(const model_id_t &id) {
    std::unique_lock lock(cache_lock);
    auto it = comp_cache.find(id);
    if (it != comp_cache.end() && !it->second.empty())
	return it->second;
    lock.unlock();
    composition_t comp = _get_composition.on(get_provider(id))(id);
    lock.lock();
    it = comp_cache.emplace_hint(it, id, comp);
    return it->second;
}

bool rpc_client::read_layers(const model_id_t &id, const vertex_list_t &layer_id,
			     std::vector<segment_t> &segment_list, std::vector<uint64_t> &owners) {
    struct req_info_t {
	vertex_list_t layer_id;
	std::vector<segment_t> segments;
    };
    std::unordered_map<model_id_t, req_info_t> owner_map;
    std::vector<tl::bulk> bulks;
    std::vector<tl::async_response> reps;

    for (int i = 0; i < layer_id.size(); i++) {
	auto owner = owners[i];
	// auto [owner, size] = comp[layer_id[i]];
	auto &e = owner_map[owner];
	e.layer_id.emplace_back(layer_id[i]);
	e.segments.emplace_back(segment_list[i]);
    }
    for (auto &e : owner_map) {
	bulks.emplace_back(engine.expose(e.second.segments, tl::bulk_mode::write_only));
	reps.emplace_back(_read_layers.on(get_provider(e.first)).async(e.second.layer_id, e.first, bulks.back()));
    }
    bool result = true;
    for (auto &rep : reps)
	result = result && rep.wait();
    return result;
}

bool rpc_client::update_ref_counter(const model_id_t &id, int value) {
    std::unordered_map<model_id_t, vertex_list_t> req_args;
    std::vector<tl::async_response> reps;
    auto &comp = get_composition(id);
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

prefix_t rpc_client::get_prefix(const digraph_t &child) {
    prefix_t max_result;
    std::vector<tl::async_response> requests;
    for (auto &provider : providers)
	requests.emplace_back(_get_prefix.on(provider).async(child));
    for (auto &request : requests) {
	prefix_t result = request.wait();
	if (result.second.size() > max_result.second.size())
	    std::swap(result, max_result);
    }
    return max_result;
}

int rpc_client::shutdown() {
	INFO("client issued shutdown");
	for (auto const &i : providers) {
	    engine.shutdown_remote_engine(i);
	}
	return 0;
    }

} // namespace dstates::ai
