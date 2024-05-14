#ifndef DSTATESAI_TYPES_H
#define DSTATESAI_TYPES_H

#include <utility>
#include <string>
#include <vector>
#include <chrono>
#include <unordered_set>
#include <unordered_map>
#include <map>

namespace dstatesai {

/*
 * TODO we need to extend this datamodel to account for the notion of versions and allow us to walk the history of versions
 */

/**
 * stores a pointer and a size to describe a memory segment
 */
typedef std::pair<void *, size_t> segment_t;
/**
 * ID for a vertex/layer
 */
typedef uint64_t vertex_t;
/**
 * ID for a model
 */
typedef uint64_t model_id_t;
/**
 * collection of vertices
 *
 * TODO these are not std::list -> should we rename these
 */
typedef std::vector<vertex_t> vertex_list_t;
/**
 * collection of models
 *
 * TODO these are not std::list -> should we rename these
 */
typedef std::vector<model_id_t> model_id_list_t;
/**
 * maps a model_id to the vertices in the model
 *
 * TODO should this be a model -> (vertex, size)
 */
typedef std::unordered_map<model_id_t, std::pair<model_id_t, size_t>> composition_t;
/**
 * maps a model_id to list of verticies in the generalized longest common prefix
 */
typedef std::pair<model_id_t, vertex_list_t> prefix_t;
/**
 * begin and end timestamp for a model
 */
typedef std::pair<uint64_t, uint64_t> timestamp_t;
/**
 * map of events names to a list of the durations for this event
 */
typedef std::map<std::string, std::vector<uint64_t>> timestamp_map_t;

/**
 * records the access of a particular tensor
 *
 * TODO appears to be unused...
 */
struct tensor_access_t 
{
    /** how many times was this tensor accessed*/
    uint64_t count;
    /** how long were the accesses to the tensor */
    std::vector<double> elapsed_timestamps;
    /** when was it first created? */
    std::chrono::steady_clock::time_point first_timestamp; 
     template<typename A> void serialize(A& ar) {
         ar & count;
         ar & elapsed_timestamps;
     }
};
/**
 *
 * TODO this needs a better name in the context of the program
 */
struct digraph_t {
    typedef std::unordered_set<vertex_t> vset_t;

    /// the id of this model
    model_id_t id = 0;
    /// what is the "global" root vertex; in graphs with multiple verteices with no in-edges, this is a "super vertex" that connects all of them.
    vertex_t root;
    /// what are the out edges from each vertex
    std::unordered_map<vertex_t, vset_t> out_edges;
    /// what is the in-degree of each vertex, used to facilitate LCP
    std::unordered_map<vertex_t, int> in_degree;

    template<typename A> void serialize(A& ar) {
        ar & id;
        ar & root;
        ar & out_edges;
        ar & in_degree;
    }
};

/**
 *  this appears to be unused now
 */
inline constexpr size_t rdma_transfer_size=400000000;
/**
 * memory allocated upfront by the client
 *
 * TODO isn't pinned for now; should we rename?
 * TODO should this be dynamic?
 */
inline constexpr size_t client_pinned_buffer_size=5000000000;
/**
 * memory allocated upfront by the server
 *
 * TODO should this be dynamic?
 */
inline constexpr size_t server_pinned_buffer_size=50000000000;

}

#endif
