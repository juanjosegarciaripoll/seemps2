#pragma once
#include <iterator>
#include "core.h"

namespace nanobind {

#if 1
class python_list_iterator {
  list list_;
  size_t index_;
  using iterator = python_list_iterator;

public:
  using iterator_category = std::forward_iterator_tag;
  using value_type = object;
  using difference_type = std::ptrdiff_t;
  using pointer = object *;

  class reference {
    list list_;
    size_t index_;

  public:
    reference(list list, size_t index) : list_{list}, index_{index} {}
    auto operator=(object p) {
      list_[index_] = p;
      return p;
    }
    auto operator=(const reference &r) {
      list_[index_] = r.list_[r.index_];
      return r;
    }
    operator object() { return list_[index_]; }
  };

  python_list_iterator(const python_list_iterator &) = default;
  python_list_iterator(python_list_iterator &&) = default;
  python_list_iterator &operator=(const python_list_iterator &) = default;
  python_list_iterator &operator=(python_list_iterator &&) = default;
  python_list_iterator(list &list, size_t index) : list_{list}, index_{index} {}
  ~python_list_iterator() = default;

  bool operator==(const iterator &other) const {
    return index_ == other.index_;
  }

  bool operator!=(const iterator &other) const {
    return index_ != other.index_;
  }

  iterator &operator++() {
    ++index_;
    return *this;
  }

  iterator operator++(int) {
    auto retval = *this;
    ++index_;
    return retval;
  }

  reference operator*() { return reference(list_, index_); }
};

class python_list_const_iterator {
  list list_;
  size_t index_;
  using iterator = python_list_const_iterator;

public:
  using iterator_category = std::forward_iterator_tag;
  using value_type = object;
  using difference_type = std::ptrdiff_t;
  using pointer = object *;
  using reference = object &;

  python_list_const_iterator(const python_list_const_iterator &) = default;
  python_list_const_iterator(python_list_const_iterator &&) = default;
  python_list_const_iterator &operator=(const python_list_const_iterator &it) {
    list_ = it.list_;
    index_ = it.index_;
    return *this;
  }
  python_list_const_iterator &operator=(python_list_const_iterator &&it) {
    list_ = std::move(it.list_);
    index_ = it.index_;
    return *this;
  }
  python_list_const_iterator(const list &list, size_t index)
      : list_{list}, index_{index} {}
  ~python_list_const_iterator() = default;

  bool operator==(const python_list_const_iterator &other) const {
    return index_ == other.index_;
  }

  bool operator!=(const python_list_const_iterator &other) const {
    return index_ != other.index_;
  }

  python_list_const_iterator &operator++() {
    ++index_;
    return *this;
  }

  python_list_const_iterator operator++(int) {
    auto return_value = *this;
    ++index_;
    return return_value;
  }

  auto operator*() const { return list_[index_]; }
};
#endif

#if 0
inline auto begin(list &l) { return l.begin(); }
inline auto end(list &l) { return l.end(); }

inline auto begin(const list &l) { return l.begin(); }
inline auto end(const list &l) { return l.end(); }

inline auto begin_const(list &l) { return l.begin(); }
inline auto end_const(list &l) { return l.end(); }
#else
inline auto begin(list &l) { return python_list_iterator(l, 0); }
inline auto end(list &l) { return python_list_iterator(l, l.size()); }

inline auto begin(const list &l) { return python_list_const_iterator(l, 0); }
inline auto end(const list &l) {
  return python_list_const_iterator(l, l.size());
}

inline auto begin_const(list &l) { return python_list_const_iterator(l, 0); }
inline auto end_const(list &l) {
  return python_list_const_iterator(l, l.size());
}
#endif

list copy(const list &l);

list rescale(const object &factor, const list &b);

object conj(const object &w);

object real(const object &w);

double abs(const object &w);

inline bool iscomplex(const object &w) { return PyComplex_Check(w.ptr()); }

bool is_true(const object &o);

template <class iterator> list copy_to_list(iterator begin, iterator end) {
  // TODO: Presize list
  list output;
  for (; begin != end; ++begin) {
    output.append(*begin);
  }
  return output;
}

template <rv_policy policy = rv_policy::automatic, typename... Args>
list make_list(Args &&...args) {
  auto result = steal<list>(PyList_New((Py_ssize_t)sizeof...(Args)));

  size_t nargs = 0;
  PyObject *o = result.ptr();

  (NB_LIST_SET_ITEM(o, nargs++,
                    detail::make_caster<Args>::from_cpp(
                        (detail::forward_t<Args>)args, policy, nullptr)
                        .ptr()),
   ...);

  return result;
}

} // namespace nanobind
