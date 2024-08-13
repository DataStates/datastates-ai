#ifndef __DSTATES_AI_TYPES_HPP
#define __DSTATES_AI_TYPES_HPP

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_set>
#include <unordered_map>

namespace dstates::ai {

static const size_t DEFAULT_BUFFER_SIZE = 1 << 30;
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
 * maps a model_id to the owner  in the model
 *
 * TODO should this be a model -> (vertex, size)
 */
typedef std::unordered_map<vertex_t, std::pair<model_id_t, size_t>> composition_t;
/**
 * maps a model_id to list of verticies in the generalized longest common prefix
 */
typedef std::pair<model_id_t, vertex_list_t> prefix_t;
/**
 *
 * TODO this needs a better name in the context of the program
 */
struct digraph_t {
    typedef std::unordered_set<vertex_t> vset_t;

    /// the id of this model
    model_id_t id = 0;
    /// what is the "global" root vertex; in graphs with multiple verteices with no in-edges,
    /// this is a "super vertex" that connects all of them.
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
}

#endif
