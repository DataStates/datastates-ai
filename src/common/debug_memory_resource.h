#include <string>
#include <memory_resource>
#include <stdlib.h>
#include <stdio.h>

class debug_resource : public std::pmr::memory_resource {
	std::string _name;
	std::pmr::memory_resource* _upstream;
	char *start;
	size_t curr_pos;
public:
	explicit debug_resource(std::string name, std::pmr::memory_resource* up = std::pmr::get_default_resource(), char* begin_ptr=NULL): _name{ std::move(name) }, _upstream{ up }, start{begin_ptr} {}

	void *do_allocate(size_t bytes, size_t alignment) override {
		  void* ret = _upstream->allocate(bytes, alignment);
		  char* temp = (char*)ret;
		  curr_pos = temp - start;
		  return ret;
	}
	void do_deallocate(void *ptr, size_t bytes, size_t alignment) override{
		 _upstream->deallocate(ptr, bytes, alignment);
	}

	bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override {
		return this == &other;
	}
	size_t get_pos_in_buffer(){
		return curr_pos;
	}
};

