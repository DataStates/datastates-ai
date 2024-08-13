#include "server.hpp"

#include <chrono>
#include <deque>
#include <thallium/serialization/stl/pair.hpp>
#include <thallium/serialization/stl/string.hpp>
#include <thallium/serialization/stl/unordered_map.hpp>
#include <thallium/serialization/stl/unordered_set.hpp>
#include <thallium/serialization/stl/vector.hpp>

#define __DEBUG
#include "debug.hpp"

logger_state_t logger_state;

namespace dstates::ai {
model_server_t::model_server_t(tl::engine &e, uint16_t provider_id, uint32_t num_procs,
			       size_t buffer_size, std::string const &server_policy)
: tl::provider<model_server_t>(e, provider_id),
request_pool(tl::pool::create(tl::pool::access::spmc)), pinned_buffer_size(buffer_size) {
    rdma_buffers_init(e);
    policy = std::move(server_policy);
    for (int i = 0; i < num_procs; i++)
	ess.emplace_back(tl::xstream::create(tl::scheduler::predef::deflt, *request_pool));
    procedures.emplace_back(define("store_meta", &model_server_t::store_meta, *request_pool));
    procedures.emplace_back(define("get_prefix", &model_server_t::get_prefix, *request_pool));
    procedures.emplace_back(define("get_composition", &model_server_t::get_composition, *request_pool));
    procedures.emplace_back(define("store_layers", &model_server_t::store_layers, *request_pool));
    procedures.emplace_back(define("read_layers", &model_server_t::read_layers, *request_pool));
    procedures.emplace_back(define("update_ref_counter", &model_server_t::update_ref_counter, *request_pool));
    procedures.emplace_back(define("shutdown", &model_server_t::shutdown, *request_pool));
    get_engine().push_finalize_callback(this, [p = this] { delete p; });
}

model_server_t::~model_server_t() {
    get_engine().wait_for_finalize();
    for (auto &proc : procedures)
	proc.deregister();
    for (int i = 0; i < ess.size(); i++)
	ess[i]->join();
    get_engine().pop_finalize_callback(this);
}

bool model_server_t::store_meta(const digraph_t &g, const composition_t &comp,
                                const float val_acc) {
    std::unique_lock lock(store_lock);
    graph_store.emplace_back(g);
    graph_info.try_emplace(g.id, model_info_t(std::prev(graph_store.end()), comp, val_acc));
    return true;
}

bool model_server_t::update_ref_counter(const model_id_t &owner, const vertex_list_t &layer_id, int value) {
    for (int i = 0; i < layer_id.size(); i++) {
	std::unique_lock lock(store_lock);
	auto &li = layer_store[layer_id[i]];
	lock.unlock();
	lock = std::unique_lock(li.layer_lock);
	auto it = li.owner_map.find(owner);
	if (it == li.owner_map.end())
	    return false;
	it->second.ref_count += value;
	if (it->second.ref_count <= 0) {
	    delete (unsigned char *)it->second.segment.first;
	    li.owner_map.erase(it);
	}
    }
    if (value < 0) {
	std::unique_lock lock(store_lock);
	auto it = graph_info.find(owner);
	if (it != graph_info.end()) {
	    graph_store.erase(it->second.index);
	    graph_info.erase(it);
	    DBG("retired model " << owner);
	}
    }
    return true;
}

void model_server_t::store_layers(const tl::request &req, const model_id_t &id,
                                  const vertex_list_t &layer_id,
                                  const std::vector<size_t> &layer_size, tl::bulk &bulk) {
    std::vector<layer_t> layers;
    std::vector<segment_t> segments;
    for (int i = 0; i < layer_id.size(); i++) {
	void *ptr;
	try {
	    std::unique_lock lock(rdma_segments.buffer_lock);
	    ptr = rdma_segments.pool->allocate(layer_size[i], alignof(std::max_align_t));
	} catch (std::bad_alloc &e) {
	    for (int j = 0; j < i; j++)
		rdma_segments.pool->deallocate(segments[j].first, segments[j].second);
	    req.respond(false);
	    return;
	}
	layers.emplace_back(layer_t(layer_size[i], ptr));
	segments.emplace_back(layers.back().segment);
    }

    tl::bulk local = get_engine().expose(segments, tl::bulk_mode::read_write);
    tl::endpoint ep = req.get_endpoint();
    bulk.on(ep) >> local;
    for (int i = 0; i < layer_id.size(); i++) {
	std::unique_lock lock(store_lock);
	auto &lid = layer_store[layer_id[i]];
	lock.unlock();
	lock = std::unique_lock(lid.layer_lock);
	auto it = lid.owner_map.find(id);
	if (it != lid.owner_map.end()) {
	    auto segment = it->second.segment;
	    it->second.segment = layers[i].segment;
	    std::unique_lock lock(rdma_segments.buffer_lock);
	    rdma_segments.pool->deallocate(segment.first, segment.second);
	} else
	    lid.owner_map.emplace_hint(it, id, layers[i]);
    }
    req.respond(true);
}

void model_server_t::read_layers(const tl::request &req, const vertex_list_t &layer_id,
                                 const model_id_t &owner, tl::bulk &layer_bulk) {
    std::vector<segment_t> segments;
    for (int i = 0; i < layer_id.size(); i++) {
	std::unique_lock lock(store_lock);
	auto &lid = layer_store[layer_id[i]];
	lock.unlock();
	lock = std::unique_lock(lid.layer_lock);
	auto it = lid.owner_map.find(owner);
	if (it != lid.owner_map.end())
	    segments.emplace_back(it->second.segment);
	else {
	    DBG("cannot find layer " << layer_id[i]);
	    req.respond(false);
	    return;
	}
    }
    tl::bulk local = get_engine().expose(segments, tl::bulk_mode::read_write);
    tl::endpoint ep = req.get_endpoint();
    layer_bulk.on(ep) << local;
    req.respond(true);
}

composition_t model_server_t::get_composition(const model_id_t &id) {
    std::unique_lock lock(store_lock);
    auto it = graph_info.find(id);
    if (it == graph_info.end())
	return composition_t();
    else
	return it->second.composition;
}

prefix_t model_server_t::get_prefix(const digraph_t &child) {
    vertex_list_t max_prefix;
    float max_accuracy = 0;
    uint64_t max_id = 0;
    for (auto &parent : graph_store) {
	std::deque<vertex_t> frontier{child.root};
	std::unordered_map<vertex_t, int> visits;
	vertex_list_t prefix;
	auto it_graph = graph_info.find(parent.id);
	float model_accuracy = it_graph->second.val_acc;
	while (frontier.size() > 0) {
	    uint64_t u = frontier.front();
	    frontier.pop_front();
	    prefix.push_back(u);
	    auto c_it = child.out_edges.find(u);
	    if (c_it == child.out_edges.end())
		continue;
	    auto p_it = parent.out_edges.find(u);
	    if (p_it == parent.out_edges.end())
		continue;
	    for (auto const &v : c_it->second) {
		if (p_it->second.count(v)) {
		    visits[v]++;
		    if (visits[v] == std::max(child.in_degree.at(v), parent.in_degree.at(v)))
			frontier.push_back(v);
		}
	    }
	}
	auto prefix_len = prefix.size();
	auto max_prefix_len = max_prefix.size();
	if ((prefix_len > max_prefix_len) ||
            ((prefix_len == max_prefix_len) && (model_accuracy > max_accuracy))) {
	    std::swap(prefix, max_prefix);
	    max_accuracy = model_accuracy;
	    max_id = parent.id;
	}
    }
    return std::make_pair(max_id, max_prefix);
}

int model_server_t::shutdown() {
    get_engine().finalize();
    return 0;
}

void model_server_t::rdma_buffers_init(tl::engine &e) {
    rdma_segments.buffer = new char[pinned_buffer_size];
    std::vector<std::pair<void *, std::size_t>> segments(1);
    segments[0].first = (void *)(&rdma_segments.buffer[0]);
    segments[0].second = pinned_buffer_size;
    e.expose(segments, tl::bulk_mode::read_write);

    rdma_segments.buffer_resource =
	std::make_unique<std::pmr::monotonic_buffer_resource>(rdma_segments.buffer,
							      pinned_buffer_size, std::pmr::null_memory_resource());
    rdma_segments.pool =
	std::make_unique<std::pmr::unsynchronized_pool_resource>(rdma_segments.buffer_resource.get());
}
} // namespace dstates::ai
