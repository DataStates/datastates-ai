#include "client-py-impl.hpp"
#include "dstates/ai/client.hpp"

#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>
#include <map>
#include <nanobind/nanobind.h>
#include <numeric>
#include <sstream>
#include <vector>

#define __DEBUG
#include "debug.hpp"

namespace dstates::ai {
namespace nb = nanobind;

py_backend::py_backend(const std::string &thallium_cfg, const std::vector<std::string> &servers, size_t buffer_size) {
    std::vector<int> providers(servers.size());
    std::iota(providers.begin(), providers.end(), 0);
    mem_buffer = std::make_unique<char[]>(buffer_size);
    buffer_resource =
	std::make_unique<std::pmr::monotonic_buffer_resource>(mem_buffer.get(), buffer_size,
							      std::pmr::null_memory_resource());
    pool = std::make_unique<std::pmr::unsynchronized_pool_resource>(buffer_resource.get());
    client = std::make_unique<rpc_client>(thallium_cfg, servers, providers);
}

bool py_backend::save_layers(tensor_list_t &tensors, uint64_t model_id, uint64_list_t &layer_ids) {
    std::pmr::vector<char> temp{pool.get()};
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
    return client->store_layers(model_id, layer_ids, segments);
}

bool py_backend::load_layers(tensor_list_t &tensors, uint64_t model_id, uint64_list_t &layer_ids,
                            uint64_list_t &layer_owners) {
    std::pmr::vector<char> temp{pool.get()};
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
    bool ret = client->read_layers(model_id, layer_ids, segments, layer_owners);
    if (is_gpu) {
	int iter = 0;
	for (auto &t : tensors) {
	    cudaMemcpy((char *)t.data(), (char *)segments[iter].first, segments[iter].second, cudaMemcpyHostToDevice);
	    ++iter;
	}
    }
    return ret;
}

bool py_backend::store_meta(uint64_t id, uint64_list_t &edges, uint64_list_t &layer_ids,
                            uint64_list_t &layer_owners, uint64_list_t &sizes,
                            const float val_acc) {
    if (edges.size() < 2 || edges.size() % 2 != 0 ||
	layer_ids.size() != layer_owners.size() || layer_ids.size() != sizes.size())
	return false;
    digraph_t g;
    g.root = edges[0];
    g.id = id;
    for (int i = 0; i < edges.size(); i += 2) {
	g.out_edges[edges[i]].insert(edges[i + 1]);
	g.in_degree[edges[i + 1]]++;
    }
    composition_t comp;
    for (int i = 0; i < layer_ids.size(); i++)
	comp.emplace(layer_ids[i], std::make_pair(layer_owners[i], sizes[i]));

    return client->store_meta(g, comp, val_acc);
}

composition_t py_backend::get_composition(uint64_t model_id) {
    return client->get_composition(model_id);
}

prefix_t py_backend::get_prefix(uint64_list_t &edges) {
    if (edges.size() < 2 || edges.size() % 2 != 0)
	return prefix_t();
    digraph_t g;
    g.root = edges[0];
    for (int i = 0; i < edges.size(); i += 2) {
	g.out_edges[edges[i]].insert(edges[i + 1]);
	g.in_degree[edges[i + 1]]++;
    }
    return client->get_prefix(g);
}

bool py_backend::update_ref_counter(uint64_t id, int value) {
    return client->update_ref_counter(id, value);
}

int py_backend::shutdown() {
    return client->shutdown();
}
} // namespace dstates::ai
