#ifndef DSTATESAI_CLIENT_CPP_H
#define DSTATESAI_CLIENT_CPP_H

#include "dstatesai/types.h"
#include <vector>
#include <string>
#include <thallium.hpp>

namespace dstatesai {
namespace tl = thallium;

/**
 * C++ client for Dstates-AI, it performs RPC calls and RMDA to communicate with the server
 */
class rpc_client {
    tl::remote_procedure _store_meta, _get_prefix, _get_composition, _store_layers, _read_layers, _update_ref_counter, _shutdown;
    std::vector<tl::provider_handle> providers;
    std::unordered_map<model_id_t, composition_t> comp_cache;
    tl::mutex cache_lock;
    tl::engine engine;

public:
    /**
     * we load balance across providers
     */
    inline tl::provider_handle &get_provider(model_id_t id) {
        return providers[id % providers.size()];
    }
    /**
     * Create an RPC client
     *
     * \param[in] servers the list of servers that can provide this request.  Addresses are expected to be valid Mochi addresses
     * \param[in] provider_ids numeric ids associated with each provider. TODO remove this from the interface
     */
    rpc_client(const std::vector<std::string> &servers, const std::vector<int>&provider_ids);
    /**
     * Store the metadata for model
     *
     * \param[in] g the graph describing the model to store
     * \param[in] comp the composition of the model in terms of layers
     */
    bool store_meta(const digraph_t &g, const composition_t &comp, float val_acc, timestamp_map_t &timestamps);
    /**
     * get the model that has the most closely matching prefix for the model, breaking ties on accuracy
     */
    prefix_t get_prefix(const digraph_t &child, timestamp_map_t &timestamps);
    /**
     * get the composition of a model in terms of layers
     */
    composition_t& get_composition(const model_id_t &id, timestamp_map_t &timestamps);
    /**
     * store the layers on the server
     *
     * \param[in] id model id to store
     * \param[in] layer_id ids of all of the layers to store
     * \param[in] segments memory for all of the segments to send 
     * \param[in,out] timestamps appends timestamps produced by storing the layers
     *
     * TODO segments is only marked as non-const here because of thalliums API; we can get around this with a const_cast
     *
     */
    bool store_layers(const model_id_t &id, const vertex_list_t &layer_id,
                      std::vector<segment_t> &segments, timestamp_map_t &timestamps);
    /**
     * read the layers from the server
     *
     * \param[in] id model id to store
     * \param[in] layer_id ids of all of the layers to store
     * \param[out] segment_list where to write all of the segments from the model
     * \param[in,out] timestamps appends timestamps produced by storing the layers
     *
     * TODO segments is only marked as non-const here because of thalliums API; we can get around this with a const_cast
     *
     */
    bool read_layers(const model_id_t &id, const vertex_list_t &layer_id, std::vector<segment_t>&segment_list, std::vector<uint64_t> &owners, timestamp_map_t &timestamps);

    /**
     * change the ref counter for id via +=value
     */
    bool update_ref_counter(const model_id_t &id, int value);

    /**
     * clear the timestamp map for id
     */
    bool clear_timestamps(const model_id_t &id);
    /**
     * append a timestamp for a model
     */
    void append_to_timestamp_map(timestamp_map_t &profile_time_stamps, timestamp_t &timestamp, std::string &function_string);
    /**
     * append a series of timestamps for a model
     */
    void append_to_timestamp_map(timestamp_map_t &profile_time_stamps, std::vector<uint64_t> &timestamp, std::string &function_string);
    /**
     * return the list of timestamps for a model
     */
    std::vector<uint64_t> get_timestamps(const model_id_t &id);
    /**
     * indicate that shutdown will occur and give thallium time to cleanup
     */
    int shutdown();
};
}

#endif
