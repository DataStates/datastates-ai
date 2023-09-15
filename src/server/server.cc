#include <utility>
#include <vector>
using digraph_t = std::pair<int,int>;
using prefix_t = std::vector<digraph_t>;

struct metrics_interface {
    virtual void get_prefix_begin(digraph_t const&)=0;
    virtual void get_prefix_end(digraph_t const&, prefix_t const& retval)=0;
};

struct server_interface {
    virtual prefix_t get_prefix(digraph_t)=0;
};

struct server_base : public server_interface {
    virtual prefix_t get_prefix_impl(digraph_t d)=0;

    prefix_t get_prefix(digraph_t d) final {
        m->get_prefix_begin(d);
        auto p = get_prefix_impl(d);
        m->get_prefix_end(d,p);
        return p;
    }
    metrics_interface* m;
};


struct server_impl : public server_base {

    prefix_t get_prefix_impl(digraph_t d);
};

template <class M>
struct server_template_common {
    prefix_t get_prefix(digraph_t d) {
        this->get_prefix_impl(d);
    }
};

struct server_template : public server_template_common<server_template> {

    prefix_t get_prefix_impl(digraph_t d) {

        return prefix_t{};
    }
};
